# Joint Edge Controls

Checks:
- readyz_passes: pass
- allowed_extract_succeeds: pass
- invalid_api_key_reaches_backend_auth: pass
- unsupported_generate_is_gateway_owned: pass
- extract_route_disabled_by_gateway: pass
- extract_jobs_route_disabled_by_gateway: pass
- oversized_extract_rejected_by_gateway: pass
- gateway_metrics_captured: pass

This proof uses the deterministic backend and isolates gateway-owned behavior.
