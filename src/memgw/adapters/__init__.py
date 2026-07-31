"""The adapter registry.

Built-ins register themselves on first use. An adapter whose optional dependency
is missing simply never appears in :func:`available` -- ``mem0`` is inert until
``pip install memgw[mem0]``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from memgw.adapters.base import MemoryAdapter
from memgw.errors import InvalidRequest

_FACTORIES: dict[str, Callable[..., MemoryAdapter]] = {}
_LOADED = False


def register(name: str, factory: Callable[..., MemoryAdapter]) -> None:
    _FACTORIES[name] = factory


def _load_builtins() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    # Every shipped adapter belongs here. An adapter absent from this list exists,
    # imports, and is entirely invisible to available() and get() -- which is how the
    # Zep adapter spent an afternoon reachable through the environment and through no
    # other route at all. tests/test_registry.py checks the list against what ships.
    for module in ("memgw.adapters.pgvector", "memgw.adapters.mem0", "memgw.adapters.zep"):
        try:
            __import__(module)
        except ImportError:
            # An optional dependency is absent. The provider is then simply not on
            # offer, which is exactly what available() should report.
            continue


def available() -> list[str]:
    _load_builtins()
    return sorted(_FACTORIES)


def get(name: str) -> Callable[..., MemoryAdapter]:
    _load_builtins()
    try:
        return _FACTORIES[name]
    except KeyError:
        raise InvalidRequest(
            f"unknown provider {name!r}",
            code="unknown_provider",
            details={"available": available()},
        ) from None


def build(name: str, **config: Any) -> MemoryAdapter:
    return get(name)(**config)


__all__ = ["MemoryAdapter", "available", "build", "get", "register"]
