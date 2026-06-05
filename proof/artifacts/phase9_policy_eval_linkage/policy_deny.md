== Provenance ==
eval_run_dir: /Users/chranama/career/llm-extraction-platform/proof/artifacts/phase9_policy_eval_linkage/eval_fail
deployment_key: async-proof--fake-extract
deployment: {"profile": "async-proof", "provider": "fake"}
# Policy Decision: `extract_enablement`

| Field | Value |
|---|---|
| ok | `False` |
| status | `DecisionStatus.deny` |
| thresholds_profile | `extract/default` |
| enable_extract | `False` |

## Provenance

| Field | Value |
|---|---|
| deployment_key | `async-proof--fake-extract` |
| deployment | `{"profile": "async-proof", "provider": "fake"}` |
| task | `extract` |
| run_id | `eval_fail` |
| run_dir | `/Users/chranama/career/llm-extraction-platform/proof/artifacts/phase9_policy_eval_linkage/eval_fail` |

## Reasons
- **system_unhealthy** - http_5xx_rate=5.000% exceeds budget max_http_5xx_rate=1.000%
- **system_unhealthy** - timeout_rate=5.000% exceeds budget max_timeout_rate=1.000%
- **system_unhealthy** - non_200_rate=10.000% exceeds budget max_non_200_rate=5.000%
- **schema_validity_too_low** - schema_validity_rate(point)=60.000% < min=90.000%

## Warnings
- **insufficient_sample_size** - n_total=2 below min_n_total=20; decision is low-confidence

## Metrics

| Metric | Value |
|---|---|
| `deployment` | `{'provider': 'fake', 'profile': 'async-proof'}` |
| `deployment_key` | `async-proof--fake-extract` |
| `field_exact_match_rate` | `{}` |
| `http_5xx_rate` | `5.0` |
| `n_ok` | `1` |
| `n_total` | `2` |
| `non_200_rate` | `10.0` |
| `run_dir` | `/Users/chranama/career/llm-extraction-platform/proof/artifacts/phase9_policy_eval_linkage/eval_fail` |
| `run_id` | `eval_fail` |
| `schema_validity_rate` | `60.0` |
| `schema_validity_rate__gate_source` | `point` |
| `schema_validity_rate__gate_value` | `60.0` |
| `task` | `extract` |
| `timeout_rate` | `5.0` |
