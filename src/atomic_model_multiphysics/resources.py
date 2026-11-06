"""Pinned real transport output for offline import and cross-checks."""

import hashlib
import json
from importlib.resources import files

REFERENCE_SHA256 = "4372e136e11d298d5a65ab5bf314c55d3bbbd2f752d51d051123a376d9db1185"


def reference_mgxs():
    raw = files("atomic_model_multiphysics").joinpath("data/mgxs_reference_v1.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REFERENCE_SHA256:
        raise ValueError("Pinned evaluated MGXS reference changed")
    return json.loads(raw)
