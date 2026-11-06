"""Strict JSON admission, physical split rules and evidence boundaries."""

import hashlib
import json
import math
import os
import stat
from pathlib import Path

from atomic_model_workspace.common import digest
from sdk.api import validate_commands

TIERS = {
    "training": dict(rtol=1e-6, atol=1e-5),
    "reference": dict(rtol=1e-7, atol=1e-6),
}
MAX_INPUT = 2 * 1024 * 1024


def read_json(path, limit=MAX_INPUT):
    # O_NONBLOCK also prevents a FIFO from hanging before fstat admission.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Input must be a bounded regular file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Input exceeds byte budget")
        return json.loads(
            raw, parse_constant=lambda value: _reject(value), object_pairs_hook=_unique_keys
        )
    finally:
        os.close(fd)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject(value):
    raise ValueError("Non-finite JSON: " + value)


def exact(obj, fields):
    if not isinstance(obj, dict) or set(obj) != set(fields):
        raise ValueError("Unexpected fields; expected " + str(sorted(fields)))


def integer(value, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Integer budget outside domain")
    return value


def admit(case):
    exact(case, {"id", "family", "split", "seed", "tier", "steps"})
    from copy import deepcopy

    case = deepcopy(case)
    for key in ("id", "family"):
        value = case[key]
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 80
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in value
            )
        ):
            raise ValueError("Invalid case/family identifier")
    if case["split"] not in ("calibration", "validation") or case["tier"] not in TIERS:
        raise ValueError("Unknown split or fidelity tier")
    integer(case["seed"], 0, 2**32 - 1)
    if not isinstance(case["steps"], list) or not 1 <= len(case["steps"]) <= 256:
        raise ValueError("Episode step budget 1..256")
    for step in case["steps"]:
        exact(step, {"duration_s", "commands"})
        dt = step["duration_s"]
        if type(dt) not in (int, float) or not math.isfinite(dt) or not 0 < dt <= 5:
            raise ValueError("Step domain (0,5] seconds")
        if not isinstance(step["commands"], dict):
            raise ValueError("Policy commands must be an object")
        try:
            step["commands"] = validate_commands(step["commands"])
        except (OverflowError, TypeError) as exc:
            raise ValueError("Invalid command value") from exc
    return json.loads(json.dumps(case, allow_nan=False))


def admit_campaign(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= 128:
        raise ValueError("Campaign case budget 1..128")
    cases = [admit(c) for c in cases]
    ids, partitions, trajectories = set(), {}, {}
    for c in cases:
        if c["id"] in ids:
            raise ValueError("Duplicate episode id")
        ids.add(c["id"])
        # Both declared family and canonical control schedule bind partitions.
        # Duration/tier/seed cannot make a duplicate command curve independent.
        schedule = []
        effective = {}
        for step in c["steps"]:
            effective = dict(effective, **step["commands"])
            if not schedule or schedule[-1] != effective:
                schedule.append(effective)
        physical = digest(schedule)
        for group, index in ((c["family"], partitions), (physical, trajectories)):
            if group in index and index[group] != c["split"]:
                raise ValueError("Physical-case leakage across calibration/validation")
            index[group] = c["split"]
    return cases


def episode_seed(root_seed, case):
    integer(root_seed, 0, 2**32 - 1)
    return int.from_bytes(
        hashlib.sha256(f"{root_seed}:{case['id']}:{case['seed']}".encode()).digest()[:4], "big"
    )


def promotion(kind, requested):
    if kind not in ("numerical", "intercode", "experimental"):
        raise ValueError("Unknown evidence kind")
    if requested not in ("implemented", "verified"):
        raise ValueError("No admitted whole-plant experimental/reference protocol")
    return dict(
        kind=kind,
        status=requested,
        scope="controlled reduced PWR software",
        validated=False,
        plant_referenced=False,
    )


def atomic_json(path, payload):
    path = Path(path)
    raw = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    tmp = path.with_name(path.name + ".pending")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
