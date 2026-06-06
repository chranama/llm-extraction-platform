# Joint Resilience Proof

This proof captures bounded degradation and operator-driven recovery for the local containerized LLMEP plus inference-serving-gateway stack.

Checks:
- baseline_sync_extract_succeeds: pass
- baseline_async_extract_succeeds: pass
- backend_timeout_is_bounded: pass
- backend_unavailable_is_bounded: pass
- backend_recovery_succeeds: pass
- worker_failure_preserves_job_state: pass
- worker_recovery_completes_job: pass
- redis_failure_is_observable: pass
- redis_recovery_succeeds: pass
- postgres_failure_is_observable: pass
- postgres_recovery_succeeds: pass
- gateway_metrics_capture_failures: pass
- backend_metrics_capture_recovery: pass
- traces_capture_recovery_flow: pass

Scope: local resilience evidence only. This does not claim HA, autoscaling, zero downtime, cloud failover, or production incident response.
