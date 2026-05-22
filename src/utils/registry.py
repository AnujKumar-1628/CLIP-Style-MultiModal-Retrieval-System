"""Minimal component registry utility."""

from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """Simple name -> object registry."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, T] = {}

    def register(self, key: str, value: T | None = None) -> T | Callable[[T], T]:
        """
        Register an item by key.

        Supports both direct and decorator styles:
            registry.register("x", obj)
            @registry.register("x")
            class X: ...
        """
        if not key or not isinstance(key, str):
            raise ValueError("registry key must be a non-empty string")

        if value is not None:
            self._do_register(key, value)
            return value

        def decorator(obj: T) -> T:
            self._do_register(key, obj)
            return obj

        return decorator

    def _do_register(self, key: str, value: T) -> None:
        if key in self._items:
            raise KeyError(f"'{key}' is already registered in '{self.name}' registry")
        self._items[key] = value

    def get(self, key: str) -> T:
        if key not in self._items:
            available = ", ".join(sorted(self._items.keys())) or "<empty>"
            raise KeyError(
                f"'{key}' is not registered in '{self.name}' registry. "
                f"Available: {available}"
            )
        return self._items[key]

    def has(self, key: str) -> bool:
        return key in self._items

    def list_keys(self) -> list[str]:
        return sorted(self._items.keys())

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __len__(self) -> int:
        return len(self._items)
