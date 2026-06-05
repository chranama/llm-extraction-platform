== Provenance ==
eval_run_dir: /Users/chranama/career/llm-extraction-platform/proof/artifacts/phase9_policy_eval_linkage/eval_pass
deployment_key: async-proof--fake-extract
deployment: {"profile": "async-proof", "provider": "fake"}
# Policy Decision: `extract_enablement`

| Field | Value |
|---|---|
| ok | `True` |
| status | `DecisionStatus.allow` |
| thresholds_profile | `extract/default` |
| enable_extract | `True` |

## Provenance

| Field | Value |
|---|---|
| deployment_key | `async-proof--fake-extract` |
| deployment | `{"profile": "async-proof", "provider": "fake"}` |
| task | `extract` |
| run_id | `eval_pass` |
| run_dir | `/Users/chranama/career/llm-extraction-platform/proof/artifacts/phase9_policy_eval_linkage/eval_pass` |

## Warnings
- **insufficient_sample_size** - n_total=2 below min_n_total=20; decision is low-confidence

## Metrics

| Metric | Value |
|---|---|
| `deployment` | `{'provider': 'fake', 'profile': 'async-proof'}` |
| `deployment_key` | `async-proof--fake-extract` |
| `field_exact_match_rate` | `{}` |
| `http_5xx_rate` | `0.0` |
| `n_ok` | `2` |
| `n_total` | `2` |
| `non_200_rate` | `0.0` |
| `run_dir` | `/Users/chranama/career/llm-extraction-platform/proof/artifacts/phase9_policy_eval_linkage/eval_pass` |
| `run_id` | `eval_pass` |
| `schema_validity_rate` | `99.0` |
| `schema_validity_rate__gate_source` | `point` |
| `schema_validity_rate__gate_value` | `99.0` |
| `task` | `extract` |
| `timeout_rate` | `0.0` |
