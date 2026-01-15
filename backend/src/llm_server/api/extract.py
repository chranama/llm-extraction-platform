# src/llm_server/api/extract.py
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from llm_server.api.deps import get_api_key, get_llm
from llm_server.core.config import settings
from llm_server.core.errors import AppError
from llm_server.core.metrics import (
    EXTRACTION_CACHE_HITS,
    EXTRACTION_REPAIR,
    EXTRACTION_REQUESTS,
    EXTRACTION_VALIDATION_FAILURES,
)
from llm_server.core.redis import get_redis_from_request, redis_get, redis_set
from llm_server.core.schema_registry import (
    SchemaLoadError,
    SchemaNotFoundError,
    list_schemas,
    load_schema,
)
from llm_server.core.validation import (
    DependencyMissingError,
    JSONSchemaValidationError,
    validate_jsonschema,
)
from llm_server.db.models import CompletionCache, InferenceLog
from llm_server.db.session import async_session_maker
from llm_server.services.llm import MultiModelManager

router = APIRouter()

REDIS_TTL_SECONDS = 3600

# Delimiters help good models be deterministic, but we don't rely on them.
_JSON_BEGIN = "<<<JSON>>>"
_JSON_END = "<<<END>>>"


# -------------------------------
# Schemas
# -------------------------------


class ExtractRequest(BaseModel):
    schema_id: str = Field(..., description="Schema id (e.g. ticket_v1, invoice_v1, receipt_v1)")
    text: str = Field(..., description="Raw text or OCR text to extract from")

    # Optional model override
    model: str | None = None

    # Generation params
    max_new_tokens: int | None = 512
    temperature: float | None = 0.0

    # Platform behavior
    cache: bool = True
    repair: bool = True  # one repair attempt if invalid JSON or schema validation fails


class ExtractResponse(BaseModel):
    schema_id: str
    model: str
    data: dict[str, Any]
    cached: bool
    repair_attempted: bool


# -------------------------------
# LLM routing
# -------------------------------


def resolve_model(llm: Any, model_override: str | None) -> tuple[str, Any]:
    allowed = settings.all_model_ids

    if model_override is None:
        model_id = settings.model_id
    else:
        model_id = model_override
        if model_id not in allowed:
            raise AppError(
                code="model_not_allowed",
                message=f"Model '{model_id}' not allowed.",
                status_code=status.HTTP_400_BAD_REQUEST,
                extra={"allowed": allowed},
            )

    if isinstance(llm, MultiModelManager):
        if model_id not in llm:
            raise AppError(
                code="model_missing",
                message=f"Model '{model_id}' not found in LLM registry",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return model_id, llm[model_id]

    if isinstance(llm, dict):
        if model_id not in llm:
            raise AppError(
                code="model_missing",
                message=f"Model '{model_id}' not found in LLM registry",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        return model_id, llm[model_id]

    return model_id, llm


# -------------------------------
# Helpers
# -------------------------------


def _hash_text(schema_id: str, text: str) -> str:
    payload = f"{schema_id}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]


def _fingerprint_params(body: ExtractRequest) -> str:
    params = body.model_dump(exclude={"text", "model", "cache", "repair"}, exclude_none=True)
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def _make_redis_key(model_id: str, prompt_hash: str, params_fp: str) -> str:
    return f"llm:extract:{model_id}:{prompt_hash}:{params_fp}"


def _strip_wrapping_code_fences(s: str) -> str:
    s = s.strip()
    if not s.startswith("```"):
        return s
    s = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n?", "", s)
    s = re.sub(r"\n?\s*```\s*$", "", s)
    return s.strip()


def _schema_summary(schema: dict[str, Any]) -> str:
    """
    Compact schema representation to avoid prompting with a huge JSON schema.
    We include:
      - required fields
      - for each property: type + enum (if present)
    """
    required = schema.get("required") or []
    props = schema.get("properties") or {}

    lines = []
    if required:
        lines.append(f"REQUIRED_FIELDS: {', '.join(required)}")

    lines.append("FIELDS:")
    for k, v in props.items():
        if not isinstance(v, dict):
            continue
        t = v.get("type", "any")
        enum = v.get("enum")
        pat = v.get("pattern")
        desc = v.get("description")

        pieces = [f"- {k}: {t}"]
        if enum:
            pieces.append(f"enum={enum}")
        if pat:
            pieces.append(f"pattern={pat}")
        if desc:
            # keep it short; small models tend to echo long text
            pieces.append(f"desc={str(desc)[:80]}")
        lines.append("  " + " | ".join(pieces))

    # also mention additionalProperties if strict
    ap = schema.get("additionalProperties", None)
    if ap is False:
        lines.append("CONSTRAINT: additionalProperties=false (no extra keys).")

    return "\n".join(lines)


def _build_extraction_prompt(schema_id: str, schema: dict[str, Any], text: str) -> str:
    summary = _schema_summary(schema)
    return (
        "You are a structured information extraction engine.\n"
        "Return ONLY a JSON object that matches the contract below.\n"
        "No markdown. No code fences. No commentary.\n"
        "If a value is unknown: omit the field unless it is REQUIRED.\n"
        "If a REQUIRED field is missing in the text: set it to null.\n\n"
        f"OUTPUT FORMAT:\n{_JSON_BEGIN}\n<JSON_OBJECT>\n{_JSON_END}\n\n"
        f"SCHEMA_ID: {schema_id}\n"
        f"{summary}\n\n"
        f"INPUT_TEXT:\n{text}\n"
    )


def _build_repair_prompt(
    schema_id: str,
    schema: dict[str, Any],
    text: str,
    bad_output: str,
    error_hint: str,
) -> str:
    summary = _schema_summary(schema)
    return (
        "Your previous output did NOT match the contract.\n"
        "Fix it. Return ONLY the corrected JSON object.\n"
        "No markdown. No code fences. No commentary.\n\n"
        f"OUTPUT FORMAT:\n{_JSON_BEGIN}\n<JSON_OBJECT>\n{_JSON_END}\n\n"
        f"SCHEMA_ID: {schema_id}\n"
        f"{summary}\n\n"
        f"INPUT_TEXT:\n{text}\n\n"
        f"PREVIOUS_OUTPUT:\n{bad_output}\n\n"
        f"ERROR_HINT:\n{error_hint}\n"
    )


def _iter_json_objects(raw: str) -> list[dict[str, Any]]:
    """
    Extract ALL JSON objects from a messy string using JSONDecoder.raw_decode.

    This is robust to prompt-echo because we validate candidates against the schema.
    """
    s = _strip_wrapping_code_fences(raw)
    dec = json.JSONDecoder()

    objs: list[dict[str, Any]] = []
    i = 0
    n = len(s)

    while i < n:
        # find next plausible start of a JSON object
        j = s.find("{", i)
        if j == -1:
            break
        try:
            obj, end = dec.raw_decode(s[j:])
            if isinstance(obj, dict):
                objs.append(obj)
            i = j + max(end, 1)
        except Exception:
            i = j + 1

    return objs


def _validate_first_matching(schema: dict[str, Any], raw_output: str) -> dict[str, Any]:
    """
    Try:
      1) if delimiters exist, decode inside them first
      2) otherwise scan all objects and choose first that validates
    """
    if raw_output is None:
        raise AppError(
            code="invalid_json",
            message="Model output was empty.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    s = raw_output.strip()

    # 1) Prefer delimited JSON if present (best signal)
    if _JSON_BEGIN in s and _JSON_END in s:
        try:
            inner = s.split(_JSON_BEGIN, 1)[1].split(_JSON_END, 1)[0].strip()
            inner = _strip_wrapping_code_fences(inner)
            obj = json.loads(inner)
            if not isinstance(obj, dict):
                raise ValueError("Delimited JSON was not an object")
            validate_jsonschema(schema, obj)
            return obj
        except DependencyMissingError as e:
            raise AppError(code=e.code, message=e.message, status_code=500) from e
        except JSONSchemaValidationError as e:
            raise AppError(
                code=e.code,
                message=e.message,
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                extra={"errors": e.errors, "raw_preview": (raw_output or "")[:500]},
            ) from e
        except Exception:
            # fall through to scanning
            pass

    # 2) Scan candidates and validate
    candidates = _iter_json_objects(s)
    if not candidates:
        raise AppError(
            code="invalid_json",
            message="Model output did not contain any JSON object.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            extra={"raw_preview": (raw_output or "")[:500]},
        )

    last_validation_error: Optional[JSONSchemaValidationError] = None

    for obj in candidates:
        try:
            validate_jsonschema(schema, obj)
            return obj
        except DependencyMissingError as e:
            raise AppError(code=e.code, message=e.message, status_code=500) from e
        except JSONSchemaValidationError as e:
            last_validation_error = e
            continue
        except Exception:
            continue

    # No candidate validated
    extra: dict[str, Any] = {"raw_preview": (raw_output or "")[:500], "candidates_found": len(candidates)}
    if last_validation_error is not None:
        extra["errors"] = last_validation_error.errors

    raise AppError(
        code="schema_validation_failed",
        message="No JSON object in the model output conformed to the schema.",
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        extra=extra,
    )


def _failure_stage_for_app_error(e: AppError, *, is_repair: bool) -> str | None:
    if getattr(e, "status_code", None) != status.HTTP_422_UNPROCESSABLE_ENTITY:
        return None

    # if we never found JSON at all, treat as parse
    if e.code == "invalid_json":
        return "repair_parse" if is_repair else "parse"
    if e.code == "schema_validation_failed":
        return "repair_validate" if is_repair else "validate"

    return "repair_validate" if is_repair else "validate"


# -------------------------------
# Routes
# -------------------------------


@router.get("/v1/schemas")
async def schemas_index(api_key=Depends(get_api_key)):
    return [{"schema_id": s.schema_id, "title": s.title, "description": s.description} for s in list_schemas()]


@router.get("/v1/schemas/{schema_id}", response_model=dict)
async def schema_detail(schema_id: str, api_key=Depends(get_api_key)):
    """
    Return the full JSON schema for a given schema_id.
    Used by the UI schema inspector + debugging tools.
    """
    try:
        schema = load_schema(schema_id)
    except SchemaNotFoundError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=status.HTTP_404_NOT_FOUND,
            extra={"schema_id": e.schema_id},
        ) from e
    except SchemaLoadError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            extra={"schema_id": e.schema_id},
        ) from e

    # Ensure it's JSON-serializable and returned as JSON
    return JSONResponse(content=schema)


@router.post("/v1/extract", response_model=ExtractResponse)
async def extract(
    request: Request,
    body: ExtractRequest,
    api_key=Depends(get_api_key),
    llm: Any = Depends(get_llm),
):
    model_id, model = resolve_model(llm, body.model)

    request.state.route = "/v1/extract"
    request.state.model_id = model_id
    request_id = getattr(request.state, "request_id", None)

    EXTRACTION_REQUESTS.labels(schema_id=body.schema_id, model_id=model_id).inc()

    try:
        schema = load_schema(body.schema_id)
    except SchemaNotFoundError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=status.HTTP_404_NOT_FOUND,
            extra={"schema_id": e.schema_id},
        ) from e
    except SchemaLoadError as e:
        raise AppError(
            code=e.code,
            message=e.message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            extra={"schema_id": e.schema_id},
        ) from e

    prompt_hash = _hash_text(body.schema_id, body.text)
    params_fp = _fingerprint_params(body)
    redis_key = _make_redis_key(model_id, prompt_hash, params_fp)

    start = time.time()
    redis = get_redis_from_request(request)

    async with async_session_maker() as session:
        # ---- 0) Redis cache ----
        if redis is not None and body.cache:
            raw = await redis_get(redis, redis_key, model_id=model_id, kind="single")
            if raw is not None:
                try:
                    payload = json.loads(raw)
                    data = payload.get("data")
                    if not isinstance(data, dict):
                        raise ValueError("Bad cache payload")

                    validate_jsonschema(schema, data)

                    EXTRACTION_CACHE_HITS.labels(schema_id=body.schema_id, model_id=model_id, layer="redis").inc()

                    request.state.cached = True
                    latency_ms = (time.time() - start) * 1000

                    session.add(
                        InferenceLog(
                            api_key=api_key.key,
                            request_id=request_id,
                            route="/v1/extract",
                            client_host=request.client.host if request.client else None,
                            model_id=model_id,
                            params_json={"schema_id": body.schema_id, "cache": True, "repair": body.repair},
                            prompt=body.text,
                            output=json.dumps(data, ensure_ascii=False),
                            latency_ms=latency_ms,
                            prompt_tokens=None,
                            completion_tokens=None,
                        )
                    )
                    await session.commit()

                    return ExtractResponse(
                        schema_id=body.schema_id,
                        model=model_id,
                        data=data,
                        cached=True,
                        repair_attempted=False,
                    )
                except DependencyMissingError as e:
                    raise AppError(code=e.code, message=e.message, status_code=500) from e
                except JSONSchemaValidationError:
                    # schema mismatch => treat as miss
                    pass
                except Exception:
                    pass

        # ---- 1) DB CompletionCache ----
        if body.cache:
            res = await session.execute(
                select(CompletionCache).where(
                    CompletionCache.model_id == model_id,
                    CompletionCache.prompt_hash == prompt_hash,
                    CompletionCache.params_fingerprint == params_fp,
                )
            )
            cached = res.scalar_one_or_none()
            if cached is not None:
                try:
                    data = json.loads(cached.output)
                    if not isinstance(data, dict):
                        raise ValueError("Expected object")

                    validate_jsonschema(schema, data)

                    EXTRACTION_CACHE_HITS.labels(schema_id=body.schema_id, model_id=model_id, layer="db").inc()

                    request.state.cached = True
                    latency_ms = (time.time() - start) * 1000

                    if redis is not None:
                        try:
                            await redis_set(redis, redis_key, json.dumps({"data": data}), ex=REDIS_TTL_SECONDS)
                        except Exception:
                            pass

                    session.add(
                        InferenceLog(
                            api_key=api_key.key,
                            request_id=request_id,
                            route="/v1/extract",
                            client_host=request.client.host if request.client else None,
                            model_id=model_id,
                            params_json={"schema_id": body.schema_id, "cache": True, "repair": body.repair},
                            prompt=body.text,
                            output=json.dumps(data, ensure_ascii=False),
                            latency_ms=latency_ms,
                            prompt_tokens=None,
                            completion_tokens=None,
                        )
                    )
                    await session.commit()

                    return ExtractResponse(
                        schema_id=body.schema_id,
                        model=model_id,
                        data=data,
                        cached=True,
                        repair_attempted=False,
                    )
                except DependencyMissingError as e:
                    raise AppError(code=e.code, message=e.message, status_code=500) from e
                except JSONSchemaValidationError:
                    pass
                except Exception:
                    pass

        # ---- 2) Run model ----
        prompt = _build_extraction_prompt(body.schema_id, schema, body.text)
        result = model.generate(
            prompt=prompt,
            max_new_tokens=body.max_new_tokens,
            temperature=body.temperature,
        )
        output = result if isinstance(result, str) else str(result)

        repair_attempted = False

        try:
            data = _validate_first_matching(schema, output)
        except AppError as e:
            stage = _failure_stage_for_app_error(e, is_repair=False)
            if stage is not None:
                EXTRACTION_VALIDATION_FAILURES.labels(schema_id=body.schema_id, model_id=model_id, stage=stage).inc()

            if not body.repair:
                raise

            repair_attempted = True
            EXTRACTION_REPAIR.labels(schema_id=body.schema_id, model_id=model_id, outcome="attempted").inc()

            error_hint = json.dumps(
                {"code": e.code, "message": e.message, **(e.extra or {})},
                ensure_ascii=False,
            )

            repair_prompt = _build_repair_prompt(
                body.schema_id,
                schema,
                body.text,
                bad_output=output,
                error_hint=error_hint,
            )
            repair_result = model.generate(
                prompt=repair_prompt,
                max_new_tokens=body.max_new_tokens,
                temperature=0.0,
            )
            repaired = repair_result if isinstance(repair_result, str) else str(repair_result)

            try:
                data = _validate_first_matching(schema, repaired)
                EXTRACTION_REPAIR.labels(schema_id=body.schema_id, model_id=model_id, outcome="success").inc()
            except AppError as e2:
                stage2 = _failure_stage_for_app_error(e2, is_repair=True)
                if stage2 is not None:
                    EXTRACTION_VALIDATION_FAILURES.labels(
                        schema_id=body.schema_id, model_id=model_id, stage=stage2
                    ).inc()
                EXTRACTION_REPAIR.labels(schema_id=body.schema_id, model_id=model_id, outcome="failure").inc()
                raise

        latency_ms = (time.time() - start) * 1000
        request.state.cached = False

        # ---- 3) Save cache (DB + Redis) ----
        if body.cache:
            session.add(
                CompletionCache(
                    model_id=model_id,
                    prompt=body.text,
                    prompt_hash=prompt_hash,
                    params_fingerprint=params_fp,
                    output=json.dumps(data, ensure_ascii=False),
                )
            )

            try:
                await session.flush()
            except IntegrityError:
                await session.rollback()

            if redis is not None:
                try:
                    await redis_set(redis, redis_key, json.dumps({"data": data}), ex=REDIS_TTL_SECONDS)
                except Exception:
                    pass

        # ---- 4) Save log ----
        session.add(
            InferenceLog(
                api_key=api_key.key,
                request_id=request_id,
                route="/v1/extract",
                client_host=request.client.host if request.client else None,
                model_id=model_id,
                params_json={"schema_id": body.schema_id, "cache": body.cache, "repair": body.repair},
                prompt=body.text,
                output=json.dumps(data, ensure_ascii=False),
                latency_ms=latency_ms,
                prompt_tokens=None,
                completion_tokens=None,
            )
        )
        await session.commit()

        return ExtractResponse(
            schema_id=body.schema_id,
            model=model_id,
            data=data,
            cached=False,
            repair_attempted=repair_attempted,
        )