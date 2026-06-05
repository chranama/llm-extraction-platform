# Async Trace Timeline

- Trace ID: `63535f7bdb966c4d1c2902cd275ec54e`
- Status: `completed`
- Request Kind: `async_extract`

| Time | Event | Stage | Status | Job | Model |
| --- | --- | --- | --- | --- | --- |
| 2026-06-05T20:16:00.563974Z | extract_job.submitted | submitted | accepted |  |  |
| 2026-06-05T20:16:00.570608Z | extract_job.persisted | persisted | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:00.573265Z | extract_job.queued | queued | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:00.593913Z | extract_job.status_polled | status_poll | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:01.121175Z | extract_job.status_polled | status_poll | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:01.637571Z | extract_job.status_polled | status_poll | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:01.690667Z | extract_job.worker_claimed | claim_job | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:01.697399Z | extract_job.execution_started | execution_started | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:01.700634Z | extract.accepted | start | accepted | 2a8cab05d34e4f27ae590828bf67cecc |  |
| 2026-06-05T20:16:01.703630Z | extract.model_resolved | resolve_model | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:01.707009Z | extract.cache_lookup | cache_read | miss | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:01.710113Z | extract.generate_completed | model_generate | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:01.712966Z | extract.validation_completed | validate_output | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:02.042815Z | extract.cache_written | cache_write | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:02.064026Z | extract.logged | log_uncached | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:02.071513Z | extract.completed | complete | completed | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:02.083033Z | extract_job.completed | complete_job | completed | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
| 2026-06-05T20:16:02.160031Z | extract_job.status_polled | status_poll | ok | 2a8cab05d34e4f27ae590828bf67cecc | fake-extract |
