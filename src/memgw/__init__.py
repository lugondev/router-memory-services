"""memgw -- one API in front of any AI memory provider."""

from memgw.capabilities import Capabilities
from memgw.client import Memory, ScopeHandle
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
    "Capabilities",
    "Episode",
    "GatewayError",
    "HealthStatus",
    "Memory",
    "MemoryRecord",
    "Message",
    "ProviderMemory",
    "Scope",
    "ScopeHandle",
    "SearchQuery",
]

__version__ = "0.1.0"
