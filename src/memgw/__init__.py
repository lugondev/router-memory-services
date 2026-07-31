"""memgw -- one API in front of any AI memory provider."""

from memgw.errors import GatewayError
from memgw.types import (
    Episode,
    HealthStatus,
    MemoryRecord,
    Message,
    ProviderMemory,
    Scope,
    SearchQuery,
)

__all__ = [
    "Episode",
    "GatewayError",
    "HealthStatus",
    "MemoryRecord",
    "Message",
    "ProviderMemory",
    "Scope",
    "SearchQuery",
]

__version__ = "0.1.0"
