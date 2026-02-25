# llm_eval/metrics/extraction_scoring.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from llm_eval.metrics.common import quantile

# -------------------------
# Types
# -------------------------

FieldName = str


@dataclass(frozen=True)
class ExtractAttempt:
    """
    One model/service attempt for a single document.

    You can build these records in your extraction runner from:
      - dataset example (id, expected, schema_id)
      - API response on success OR error payload on failure
      - timing, caching, etc.
    """
    doc_id: str
    schema_id: str

    expected: Dict[str, Any]  # ground truth
    predicted: Optional[Dict[str, Any]]  # None if request failed (422/500/etc.)

    # Service signals (fill what you have)
    ok: bool  # True if response returned schema-valid JSON (HTTP 200)
    status_code: Optional[int] = None  # HTTP status for failures
    error_code: Optional[str] = None  # e.g. "invalid_json", "schema_validation_failed"
    error_stage: Optional[str] = None  # optional: parse/validate/repair_parse/repair_validate

    repair_attempted: bool = False  # from API response on success OR inferred from error path
    cached: bool = False  # from API response
    cache_layer: Optional[str] = None  # "redis" / "db" / None (if you track)

    latency_ms: Optional[float] = None  # end-to-end (client) or server if exposed


# -------------------------
# Normalization utilities
# -------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_CURRENCY_RE = re.compile(r"[€$£¥₹₩₽₺₫₴₦₲₱₡₵₸₮₭₤₳₠₢₣₥₧₯₰₶₷₺₻₼₾₿]")


def _is_empty(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    return False


def norm_text_basic(s: str) -> str:
    """
    Conservative normalization:
    - strip
    - casefold
    - collapse whitespace
    """
    s = s.strip().casefold()
    s = _WS_RE.sub(" ", s)
    return s


def norm_text_strict(s: str) -> str:
    """
    Stronger normalization for noisy OCR:
    - basic normalization
    - remove most punctuation
    """
    s = norm_text_basic(s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def norm_company(s: str) -> str:
    """
    Company names suffer OCR punctuation + casing.
    Use strict normalization and remove common noise tokens.
    """
    s = norm_text_strict(s)

    # Optionally remove trailing legal suffixes for robustness
    # (keep conservative; don't over-normalize)
    suffixes = ["inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "co"]
    toks = s.split()
    # Remove suffix if it is the last token
    if toks and toks[-1] in suffixes:
        toks = toks[:-1]
    return " ".join(toks).strip()


def norm_address(s: str) -> str:
    """
    Address is usually the noisiest; strict normalization helps.
    """
    return norm_text_strict(s)


def _parse_amount_to_float(s: str) -> Optional[float]:
    """
    Parse currency-ish amount strings to float.
    Handles:
      "$1,234.56", "1.234,56", "1234.56", "TOTAL 12.34"
    We keep it best-effort.
    """
    if not isinstance(s, str):
        s = str(s)

    t = s.strip()
    if not t:
        return None

    # Remove currency symbols and spaces
    t = _CURRENCY_RE.sub("", t)
    t = t.replace(" ", "")

    # Keep only digits, separators, minus
    t = re.sub(r"[^0-9,.\-]", "", t)
    if not t or t in {"-", ".", ",", "-.", "-,"}:
        return None

    # Heuristic: if both '.' and ',' exist, decide which is decimal by last separator
    if "." in t and "," in t:
        last_dot = t.rfind(".")
        last_com = t.rfind(",")
        if last_dot > last_com:
            # dot decimal, remove commas
            t = t.replace(",", "")
        else:
            # comma decimal, remove dots, swap comma -> dot
            t = t.replace(".", "")
            t = t.replace(",", ".")
    else:
        # Only one separator type: if comma used, treat as decimal if last group length != 3
        if "," in t and "." not in t:
            parts = t.split(",")
            if len(parts) == 2 and len(parts[1]) in (1, 2):
                t = parts[0].replace(".", "") + "." + parts[1]
            else:
                # commas are thousands separators
                t = t.replace(",", "")

    try:
        return float(t)
    except Exception:
        return None


def norm_total(s: str) -> str:
    """
    Normalize totals by parsing to float when possible; otherwise fall back to strict text.
    Represent floats in a canonical form.
    """
    f = _parse_amount_to_float(s)
    if f is None:
        return norm_text_strict(s)
    # canonical numeric format (avoid scientific notation)
    return f"{f:.2f}"


def norm_date_loose(s: str) -> str:
    """
    Light date normalization:
    - basic normalization
    - remove punctuation
    Keeps digits/letters but standardizes separators.
    This is *not* full date parsing; it's robust across formats.
    """
    t = norm_text_basic(s)
    # replace separators with single dash
    t = re.sub(r"[./\\_]", "-", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -------------------------
# Field comparators
# -------------------------

def default_field_normalizer(field: FieldName):
    if field == "company":
        return norm_company
    if field == "address":
        return norm_address
    if field == "total":
        return norm_total
    if field == "date":
        return norm_date_loose
    # fallback
    return norm_text_strict


def field_equal(
    field: FieldName,
    pred_val: Any,
    exp_val: Any,
    *,
    allow_none_match: bool = False,
) -> bool:
    """
    Compare field values with normalization.

    allow_none_match:
      - If True: None == None counts as correct.
      - If False: if expected is non-empty, predicted must match it; None is incorrect.
    """
    if _is_empty(exp_val):
        return allow_none_match and _is_empty(pred_val)

    if _is_empty(pred_val):
        return False

    p = str(pred_val)
    e = str(exp_val)

    norm = default_field_normalizer(field)
    return norm(p) == norm(e)


# -------------------------
# Core per-document scoring
# -------------------------

@dataclass(frozen=True)
class DocFieldScore:
    doc_id: str
    schema_id: str
    ok: bool
    repair_attempted: bool
    cached: bool
    latency_ms: Optional[float]
    status_code: Optional[int]
    error_code: Optional[str]
    error_stage: Optional[str]
    field_correct: Dict[str, Optional[bool]]
    required_all_correct: Optional[bool]
    required_present_non_null: Optional[bool]


def score_document(
    attempt: ExtractAttempt,
    *,
    fields: Sequence[FieldName],
    required_fields: Sequence[FieldName],
    ignore_if_expected_missing: bool = True,
    allow_none_match_when_expected_missing: bool = False,
) -> DocFieldScore:
    """
    Returns per-document correctness by field + rollups.

    ignore_if_expected_missing:
      If expected[field] is empty/missing, we return None for that field (not scorable).
    """
    field_correct: Dict[str, Optional[bool]] = {}

    if not attempt.ok or attempt.predicted is None:
        for f in fields:
            field_correct[f] = None
        return DocFieldScore(
            doc_id=attempt.doc_id,
            schema_id=attempt.schema_id,
            ok=False,
            repair_attempted=attempt.repair_attempted,
            cached=attempt.cached,
            latency_ms=attempt.latency_ms,
            status_code=attempt.status_code,
            error_code=attempt.error_code,
            error_stage=attempt.error_stage,
            field_correct=field_correct,
            required_all_correct=None,
            required_present_non_null=None,
        )

    pred = attempt.predicted
    exp = attempt.expected

    for f in fields:
        exp_val = exp.get(f)
        pred_val = pred.get(f)

        if ignore_if_expected_missing and _is_empty(exp_val):
            field_correct[f] = None
        else:
            field_correct[f] = field_equal(
                f,
                pred_val,
                exp_val,
                allow_none_match=allow_none_match_when_expected_missing,
            )

    required_present = True
    for rf in required_fields:
        if _is_empty(pred.get(rf)):
            required_present = False
            break

    req_scores: List[bool] = []
    for rf in required_fields:
        v = field_correct.get(rf)
        if v is None:
            continue
        req_scores.append(bool(v))

    required_all_correct: Optional[bool]
    if req_scores:
        required_all_correct = all(req_scores)
    else:
        required_all_correct = None

    return DocFieldScore(
        doc_id=attempt.doc_id,
        schema_id=attempt.schema_id,
        ok=True,
        repair_attempted=attempt.repair_attempted,
        cached=attempt.cached,
        latency_ms=attempt.latency_ms,
        status_code=attempt.status_code,
        error_code=attempt.error_code,
        error_stage=attempt.error_stage,
        field_correct=field_correct,
        required_all_correct=required_all_correct,
        required_present_non_null=required_present,
    )


# -------------------------
# Aggregation helpers
# -------------------------

def _percent(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return 100.0 * (num / den)


_TIMEOUT_STAGES = {
    "timeout",
    "connect_timeout",
    "read_timeout",
    "write_timeout",
    "pool_timeout",
}


def _is_timeout_attempt(a: ExtractAttempt) -> bool:
    """
    Timeout detection aligned to HttpEvalClient (error_code="timeout" and/or error_stage in a known set).
    """
    if a.error_code and str(a.error_code).strip().lower() == "timeout":
        return True
    if a.error_stage:
        st = str(a.error_stage).strip().lower()
        if st in _TIMEOUT_STAGES:
            return True
    return False


def wilson_ci_low(num: int, den: int, z: float = 1.96) -> Optional[float]:
    """
    Wilson score interval lower bound for a binomial proportion.

    Returns a proportion in [0,1]. None if den == 0.
    """
    if den <= 0:
        return None
    if num < 0:
        num = 0
    if num > den:
        num = den

    n = float(den)
    p = float(num) / n
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = p + (z2 / (2.0 * n))
    margin = z * math.sqrt((p * (1.0 - p) + (z2 / (4.0 * n))) / n)
    low = (center - margin) / denom

    if low < 0.0:
        low = 0.0
    if low > 1.0:
        low = 1.0
    return low


def _rate_for_gating(*, percent: Optional[float], num: int, den: int, n_strict: int) -> Optional[float]:
    """
    Phase 1.3 rule:
      - if den < n_strict: gate on Wilson ci95 low
      - else: gate on point estimate
    All values are in percent units [0,100].
    """
    if percent is None or den <= 0:
        return None
    if den < int(n_strict):
        low = wilson_ci_low(int(num), int(den), z=1.96)
        return (100.0 * float(low)) if low is not None else None
    return float(percent)


# -------------------------
# Aggregate metrics
# -------------------------

@dataclass(frozen=True)
class ExtractionScoreSummary:
    n_total: int

    # validity
    n_ok: int
    schema_validity_rate: float
    schema_validity_counts: Dict[str, int]  # {"num": n_ok, "den": n_total}

    # repair
    n_invalid_initial: int
    n_repair_attempted: int
    n_repair_success: int
    repair_success_rate: float  # among attempted repairs

    # cache
    n_cached: int
    cache_hit_rate: float

    # system health (derived)
    non_200_rate: float
    non_200_counts: Dict[str, int]
    http_5xx_rate: float
    http_5xx_counts: Dict[str, int]
    timeout_rate: float
    timeout_counts: Dict[str, int]

    passed_system: bool
    passed_quality: bool
    passed: bool

    # errors (raw breakdowns)
    error_code_counts: Dict[str, int]
    status_code_counts: Dict[str, int]
    error_stage_counts: Dict[str, int]

    # field scoring
    field_exact_match_rate: Dict[str, float]
    field_exact_match_counts: Dict[str, Dict[str, int]]  # field -> {"num": ..., "den": ...}

    doc_required_exact_match_rate: Optional[float]
    doc_required_exact_match_counts: Optional[Dict[str, int]]  # {"num": ..., "den": ...}

    required_present_rate: Optional[float]
    required_present_counts: Optional[Dict[str, int]]  # {"num": ..., "den": ...}

    # uncertainty + gate values (Phase 1.3)
    schema_validity_ci95_low: Optional[float]
    required_present_ci95_low: Optional[float]
    doc_required_exact_match_ci95_low: Optional[float]

    schema_validity_gate: float
    required_present_gate: Optional[float]
    doc_required_exact_match_gate: Optional[float]

    # latency
    latency_p50_ms: Optional[float]
    latency_p95_ms: Optional[float]
    latency_p99_ms: Optional[float]


def summarize_extraction(
    attempts: Sequence[ExtractAttempt],
    *,
    fields: Sequence[FieldName],
    required_fields: Sequence[FieldName],
    ignore_if_expected_missing: bool = True,
    # System error budgets (keep local + conservative for demo; can be moved to config later)
    max_non_200_rate: float = 10.0,
    max_http_5xx_rate: float = 5.0,
    max_timeout_rate: float = 5.0,
    # Quality gating is intentionally minimal here (demo hardening, not “academic eval”)
    min_schema_validity_rate: float = 0.0,
    # Phase 1.3: small-sample conservatism
    n_strict: int = 200,
) -> ExtractionScoreSummary:
    """
    Compute the full summary for a run.

    Phase 1 hardening additions:
      - Split system health vs quality (prevents mis-blame)
      - Add effective denominators for rates (prevents “lying with percents”)
      - Add one uncertainty measure (Wilson ci95 low) and gate-on-ci for small den (prevents noisy flips)
    """
    n_total = len(attempts)
    n_ok = sum(1 for a in attempts if a.ok)

    # Cache stats (only meaningful on ok responses)
    n_cached = sum(1 for a in attempts if a.ok and a.cached)

    # Latencies
    latencies = [float(a.latency_ms) for a in attempts if a.latency_ms is not None]
    p50 = quantile(latencies, 0.50)
    p95 = quantile(latencies, 0.95)
    p99 = quantile(latencies, 0.99)

    # Error counts (existing behavior)
    error_code_counts: Dict[str, int] = {}
    status_code_counts: Dict[str, int] = {}
    error_stage_counts: Dict[str, int] = {}

    for a in attempts:
        if a.ok:
            continue
        if a.error_code:
            error_code_counts[a.error_code] = error_code_counts.get(a.error_code, 0) + 1
        if a.status_code is not None:
            k = str(a.status_code)
            status_code_counts[k] = status_code_counts.get(k, 0) + 1
        if a.error_stage:
            error_stage_counts[a.error_stage] = error_stage_counts.get(a.error_stage, 0) + 1

    # Repair stats
    n_repair_attempted = sum(1 for a in attempts if a.repair_attempted)
    n_repair_success = sum(1 for a in attempts if a.ok and a.repair_attempted)

    n_invalid_initial = n_repair_success + sum(
        1
        for a in attempts
        if (not a.ok) and (a.error_stage in {"parse", "validate", "repair_parse", "repair_validate"})
    )

    repair_success_rate = (n_repair_success / n_repair_attempted) if n_repair_attempted else 0.0

    # -------------------------
    # System health metrics (1.1)
    # -------------------------
    den_all = n_total

    non_200_num = 0
    http_5xx_num = 0
    timeout_num = 0

    for a in attempts:
        # non-200: any failure OR any explicit non-200 code
        sc = a.status_code
        if (not a.ok) or (sc is not None and int(sc) != 200):
            non_200_num += 1

        if sc is not None:
            try:
                sc_i = int(sc)
                if 500 <= sc_i <= 599:
                    http_5xx_num += 1
            except Exception:
                pass

        if _is_timeout_attempt(a):
            timeout_num += 1

    non_200_rate = _percent(non_200_num, den_all)
    http_5xx_rate = _percent(http_5xx_num, den_all)
    timeout_rate = _percent(timeout_num, den_all)

    passed_system = (
        non_200_rate <= float(max_non_200_rate)
        and http_5xx_rate <= float(max_http_5xx_rate)
        and timeout_rate <= float(max_timeout_rate)
    )

    # -------------------------
    # Quality metrics (existing) + denominators (1.2)
    # -------------------------
    doc_scores = [
        score_document(
            a,
            fields=fields,
            required_fields=required_fields,
            ignore_if_expected_missing=ignore_if_expected_missing,
        )
        for a in attempts
    ]

    field_exact_match_rate: Dict[str, float] = {}
    field_exact_match_counts: Dict[str, Dict[str, int]] = {}

    for f in fields:
        den = 0
        num = 0
        for ds in doc_scores:
            v = ds.field_correct.get(f)
            if v is None:
                continue
            den += 1
            if v:
                num += 1
        field_exact_match_rate[f] = _percent(num, den)
        field_exact_match_counts[f] = {"num": int(num), "den": int(den)}

    doc_req_den = 0
    doc_req_num = 0
    for ds in doc_scores:
        if ds.required_all_correct is None:
            continue
        doc_req_den += 1
        if ds.required_all_correct:
            doc_req_num += 1
    doc_required_exact_match_rate = _percent(doc_req_num, doc_req_den) if doc_req_den else None
    doc_required_exact_match_counts = {"num": int(doc_req_num), "den": int(doc_req_den)} if doc_req_den else None

    pres_den = 0
    pres_num = 0
    for ds in doc_scores:
        if not ds.ok:
            continue
        if ds.required_present_non_null is None:
            continue
        pres_den += 1
        if ds.required_present_non_null:
            pres_num += 1
    required_present_rate = _percent(pres_num, pres_den) if pres_den else None
    required_present_counts = {"num": int(pres_num), "den": int(pres_den)} if pres_den else None

    schema_validity_rate = _percent(n_ok, n_total)
    schema_validity_counts = {"num": int(n_ok), "den": int(n_total)}

    # -------------------------
    # Phase 1.3: CI lows + gate values
    # -------------------------
    sv_num = int(schema_validity_counts["num"])
    sv_den = int(schema_validity_counts["den"])

    schema_validity_ci95_low: Optional[float] = None
    if sv_den > 0:
        low = wilson_ci_low(sv_num, sv_den, z=1.96)
        schema_validity_ci95_low = (100.0 * float(low)) if low is not None else None

    rp_num = int(required_present_counts["num"]) if required_present_counts else 0
    rp_den = int(required_present_counts["den"]) if required_present_counts else 0
    required_present_ci95_low: Optional[float] = None
    if required_present_rate is not None and rp_den > 0:
        low = wilson_ci_low(rp_num, rp_den, z=1.96)
        required_present_ci95_low = (100.0 * float(low)) if low is not None else None

    dr_num = int(doc_required_exact_match_counts["num"]) if doc_required_exact_match_counts else 0
    dr_den = int(doc_required_exact_match_counts["den"]) if doc_required_exact_match_counts else 0
    doc_required_exact_match_ci95_low: Optional[float] = None
    if doc_required_exact_match_rate is not None and dr_den > 0:
        low = wilson_ci_low(dr_num, dr_den, z=1.96)
        doc_required_exact_match_ci95_low = (100.0 * float(low)) if low is not None else None

    schema_validity_gate = float(
        _rate_for_gating(percent=schema_validity_rate, num=sv_num, den=sv_den, n_strict=int(n_strict)) or 0.0
    )
    required_present_gate = _rate_for_gating(
        percent=required_present_rate, num=rp_num, den=rp_den, n_strict=int(n_strict)
    )
    doc_required_exact_match_gate = _rate_for_gating(
        percent=doc_required_exact_match_rate, num=dr_num, den=dr_den, n_strict=int(n_strict)
    )

    # Minimal “quality pass” signal (kept deliberately tiny for Phase 1) but uses gate value (1.3)
    passed_quality = schema_validity_gate >= float(min_schema_validity_rate)

    # Composite pass (can be ignored by policy if desired)
    passed = bool(passed_system and passed_quality)

    return ExtractionScoreSummary(
        n_total=int(n_total),
        n_ok=int(n_ok),
        schema_validity_rate=float(schema_validity_rate),
        schema_validity_counts=schema_validity_counts,
        n_invalid_initial=int(n_invalid_initial),
        n_repair_attempted=int(n_repair_attempted),
        n_repair_success=int(n_repair_success),
        repair_success_rate=100.0 * repair_success_rate if n_repair_attempted else 0.0,
        n_cached=int(n_cached),
        cache_hit_rate=_percent(n_cached, n_ok) if n_ok else 0.0,
        non_200_rate=float(non_200_rate),
        non_200_counts={"num": int(non_200_num), "den": int(den_all)},
        http_5xx_rate=float(http_5xx_rate),
        http_5xx_counts={"num": int(http_5xx_num), "den": int(den_all)},
        timeout_rate=float(timeout_rate),
        timeout_counts={"num": int(timeout_num), "den": int(den_all)},
        passed_system=bool(passed_system),
        passed_quality=bool(passed_quality),
        passed=bool(passed),
        error_code_counts=error_code_counts,
        status_code_counts=status_code_counts,
        error_stage_counts=error_stage_counts,
        field_exact_match_rate=field_exact_match_rate,
        field_exact_match_counts=field_exact_match_counts,
        doc_required_exact_match_rate=doc_required_exact_match_rate,
        doc_required_exact_match_counts=doc_required_exact_match_counts,
        required_present_rate=required_present_rate,
        required_present_counts=required_present_counts,
        schema_validity_ci95_low=schema_validity_ci95_low,
        required_present_ci95_low=required_present_ci95_low,
        doc_required_exact_match_ci95_low=doc_required_exact_match_ci95_low,
        schema_validity_gate=float(schema_validity_gate),
        required_present_gate=required_present_gate,
        doc_required_exact_match_gate=doc_required_exact_match_gate,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
    )


# -------------------------
# Convenience: pretty printing
# -------------------------

def format_summary(s: ExtractionScoreSummary) -> str:
    lines: List[str] = []
    lines.append(f"n_total={s.n_total}")

    # validity (+ denominators)
    sv_num = s.schema_validity_counts.get("num", s.n_ok)
    sv_den = s.schema_validity_counts.get("den", s.n_total)
    lines.append(f"schema_validity_rate={s.schema_validity_rate:.2f}% ({sv_num}/{sv_den})")

    # Phase 1.3: show conservatism explicitly
    if s.schema_validity_ci95_low is not None:
        lines.append(f"schema_validity_ci95_low={s.schema_validity_ci95_low:.2f}%")
    lines.append(f"schema_validity_gate={s.schema_validity_gate:.2f}% (n_strict gating)")

    # system health
    lines.append(
        f"system: non_200_rate={s.non_200_rate:.2f}% ({s.non_200_counts.get('num', 0)}/{s.non_200_counts.get('den', 0)}), "
        f"http_5xx_rate={s.http_5xx_rate:.2f}% ({s.http_5xx_counts.get('num', 0)}/{s.http_5xx_counts.get('den', 0)}), "
        f"timeout_rate={s.timeout_rate:.2f}% ({s.timeout_counts.get('num', 0)}/{s.timeout_counts.get('den', 0)})"
    )
    lines.append(f"passed_system={s.passed_system} passed_quality={s.passed_quality} passed={s.passed}")

    # repair/cache
    lines.append(
        f"repair_success_rate={s.repair_success_rate:.2f}% (success={s.n_repair_success}, attempted={s.n_repair_attempted})"
    )
    lines.append(f"cache_hit_rate={s.cache_hit_rate:.2f}% (cached={s.n_cached}/{s.n_ok})")

    # required present / doc em (+ denominators + ci low + gate)
    if s.required_present_rate is not None:
        if s.required_present_counts:
            lines.append(
                f"required_present_rate={s.required_present_rate:.2f}% ({s.required_present_counts.get('num', 0)}/{s.required_present_counts.get('den', 0)})"
            )
        else:
            lines.append(f"required_present_rate={s.required_present_rate:.2f}%")
        if s.required_present_ci95_low is not None:
            lines.append(f"required_present_ci95_low={s.required_present_ci95_low:.2f}%")
        if s.required_present_gate is not None:
            lines.append(f"required_present_gate={s.required_present_gate:.2f}% (n_strict gating)")

    if s.doc_required_exact_match_rate is not None:
        if s.doc_required_exact_match_counts:
            lines.append(
                f"doc_required_exact_match_rate={s.doc_required_exact_match_rate:.2f}% ({s.doc_required_exact_match_counts.get('num', 0)}/{s.doc_required_exact_match_counts.get('den', 0)})"
            )
        else:
            lines.append(f"doc_required_exact_match_rate={s.doc_required_exact_match_rate:.2f}%")
        if s.doc_required_exact_match_ci95_low is not None:
            lines.append(f"doc_required_exact_match_ci95_low={s.doc_required_exact_match_ci95_low:.2f}%")
        if s.doc_required_exact_match_gate is not None:
            lines.append(f"doc_required_exact_match_gate={s.doc_required_exact_match_gate:.2f}% (n_strict gating)")

    # field scoring (+ denominators)
    lines.append("field_exact_match_rate:")
    for k, v in s.field_exact_match_rate.items():
        c = s.field_exact_match_counts.get(k) or {}
        den = c.get("den")
        num = c.get("num")
        if isinstance(num, int) and isinstance(den, int) and den > 0:
            lines.append(f"  - {k}: {v:.2f}% ({num}/{den})")
        else:
            lines.append(f"  - {k}: {v:.2f}%")

    # latency
    if s.latency_p50_ms is not None:
        lines.append(f"latency_p50_ms={s.latency_p50_ms:.1f}")
    if s.latency_p95_ms is not None:
        lines.append(f"latency_p95_ms={s.latency_p95_ms:.1f}")
    if s.latency_p99_ms is not None:
        lines.append(f"latency_p99_ms={s.latency_p99_ms:.1f}")

    # raw breakdowns
    if s.status_code_counts:
        lines.append(f"status_code_counts={s.status_code_counts}")
    if s.error_code_counts:
        lines.append(f"error_code_counts={s.error_code_counts}")
    if s.error_stage_counts:
        lines.append(f"error_stage_counts={s.error_stage_counts}")

    return "\n".join(lines)