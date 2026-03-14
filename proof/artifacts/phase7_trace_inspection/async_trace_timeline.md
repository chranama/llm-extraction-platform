# Async Trace Timeline

- Trace ID: `953598d1326201d2b4d54dbcaeb84fda`
- Status: `completed`
- Request Kind: `async_extract`

| Time | Event | Stage | Status | Job | Model |
| --- | --- | --- | --- | --- | --- |
| 2026-03-14T15:53:39.200305Z | extract_job.submitted | submitted | accepted |  |  |
| 2026-03-14T15:53:39.209842Z | extract_job.persisted | persisted | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:39.213548Z | extract_job.queued | queued | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:39.236731Z | extract_job.status_polled | status_poll | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:39.758857Z | extract_job.status_polled | status_poll | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.280251Z | extract_job.status_polled | status_poll | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.551086Z | extract_job.worker_claimed | claim_job | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.558719Z | extract_job.execution_started | execution_started | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.561825Z | extract.accepted | start | accepted | 85c7dcbaadf547f5b2e53980a883f95f |  |
| 2026-03-14T15:53:40.565028Z | extract.model_resolved | resolve_model | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.568195Z | extract.cache_lookup | cache_read | miss | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.571336Z | extract.generate_completed | model_generate | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.574241Z | extract.validation_completed | validate_output | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.668677Z | extract.cache_written | cache_write | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.682343Z | extract.logged | log_uncached | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.691623Z | extract.completed | complete | completed | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.702385Z | extract_job.completed | complete_job | completed | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
| 2026-03-14T15:53:40.805697Z | extract_job.status_polled | status_poll | ok | 85c7dcbaadf547f5b2e53980a883f95f | fake-extract |
