# Policy Job Lane

Repo-level integration tests that run the `policy` job as an external CLI and
assert `policy_decision_v2` artifact behavior for:

- `extract_only` happy path
- `extract_only` fail-closed behavior
- `generate_clamp_only` shaping-only behavior
