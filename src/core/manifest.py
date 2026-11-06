"""Канонический отпечаток входа. Один и тот же набор даёт один SHA-256."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from core.status import DomainError


def _freeze(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise DomainError("Ключ манифеста должен быть строкой")
        return {key: _freeze(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_freeze(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise DomainError("Манифест содержит неконечное число")
        return number
    if isinstance(value, str) or value is None:
        return value
    raise DomainError("Тип %s в манифест не входит" % type(value).__name__)


def input_manifest(payload: dict) -> dict:
    """SHA-256 канонического JSON. Порядок ключей на отпечаток не влияет."""
    if not isinstance(payload, dict):
        raise DomainError("Манифест собирается из словаря")
    frozen = _freeze(payload)
    encoded = json.dumps(frozen, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "schema_version": 1}
