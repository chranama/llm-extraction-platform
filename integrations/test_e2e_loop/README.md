# End-to-End Loop Lane

Repo-level workflow tests spanning multiple packages. Current coverage includes:

- `eval` CLI run artifact generation
- `policy` CLI runtime decision fed from that eval run
- Cross-artifact linkage assertions (`eval_run_id`, `eval_task`, `eval_run_dir`)
