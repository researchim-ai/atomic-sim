"""Independent precision policy is explicit, bounded and part of the IVP cache key."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts.primary_steam_references import REFERENCE_POLICY, _cached_reference, reference


@pytest.fixture
def spec():
    raw = json.loads(Path("scenarios/ai05_primary_steam.json").read_text())
    raw.update(duration_s=0.02, timeline=[])
    return raw


def test_precision_is_part_of_cache_key(spec):
    at, _, _ = reference(spec)
    before = _cached_reference.cache_info()
    precision = dict(REFERENCE_POLICY, max_step_s=REFERENCE_POLICY["max_step_s"] / 2)
    tighter, _, _ = reference(spec, precision=precision)
    assert _cached_reference.cache_info().misses == before.misses + 1
    assert np.max(abs(at(0.02) - tighter(0.02))) < 1e-5
    before = _cached_reference.cache_info()
    reference(deepcopy(spec), precision=precision)
    assert _cached_reference.cache_info().hits == before.hits + 1


@pytest.mark.parametrize(
    "defect", ["unknown", "missing", "bool", "nan", "zero", "below_scipy_floor"]
)
def test_invalid_reference_precision_is_rejected(spec, defect):
    policy = deepcopy(REFERENCE_POLICY)
    if defect == "unknown":
        policy["unused"] = 1
    elif defect == "missing":
        policy.pop("rtol")
    elif defect == "bool":
        policy["rtol"] = True
    elif defect == "nan":
        policy["max_step_s"] = float("nan")
    elif defect == "zero":
        policy["mechanical_atol"] = 0
    else:
        policy["rtol"] = 1e-30
    with pytest.raises(ValueError, match="reference policy"):
        reference(spec, precision=policy)
