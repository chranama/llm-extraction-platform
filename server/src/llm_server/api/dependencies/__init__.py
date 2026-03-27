"""FastAPI dependency surfaces for route modules only."""

from llm_server.api.dependencies.admin import ensure_admin
from llm_server.api.dependencies.auth import clear_rate_limit_state, get_api_key

__all__ = ["ensure_admin", "clear_rate_limit_state", "get_api_key"]
