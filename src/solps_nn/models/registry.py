# Authors: Abdourahmane (Abdou) Diaw - diawa@ornl.gov
# SPDX-License-Identifier: Apache-2.0
"""Model registry: checkpoint bundles name architectures by registry key,
never by python path (docs/specs/checkpoint_spec.md)."""

from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register_model(name: str):
    def wrap(cls):
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"model name already registered: {name}")
        _REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return wrap


def get_model(name: str) -> type:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}") from None


def build_model(name: str, config: dict):
    return get_model(name)(**config)
