"""Frozen public research corpus; splits differ in parameters, not just seeds."""

import hashlib
import json
import re

from agent_envs.config import EnvConfig

FAMILIES = ("hold", "tracking", "recovery", "diagnosis")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def generate_pack():
    splits = {}
    for split, count, offset, tau in (
        ("train", 8, 10000, 0.1),
        ("dev", 4, 20000, 0.3),
        ("test", 20, 30000, 0.5),
    ):
        cases = []
        for family_index, family in enumerate(FAMILIES):
            for i in range(count):
                amplitude = (offset / 10000 + i / 100) * 1e-4
                cfg = EnvConfig(
                    seed=offset + family_index * 100 + i,
                    episode_horizon_s=2.0,
                    action_dt_s=0.25,
                    target_power_fraction=1 - amplitude if family == "tracking" else 1.0,
                    sensor_tau_s=tau + i * 0.001,
                    sensor_noise=1e-6,
                    timeline=({"t": 0.875, "commands": {"throttle": 1 - amplitude}},)
                    if family == "recovery"
                    else (),
                    sensor_faults=(
                        ("open" if i % 4 == 1 else "stuck")
                        if family == "diagnosis" and i % 2
                        else "",
                        "",
                        "",
                        "",
                    ),
                    power_error_limit=0.001,
                )
                config = cfg.as_dict()
                group = dict(config)
                group.pop("seed")
                cases.append(
                    dict(
                        id=f"{split}-{family}-{i:02}",
                        family=family,
                        group_id=digest(group),
                        config=config,
                        diagnosis_label="sensor_fault" if config["sensor_faults"][0] else "healthy",
                    )
                )
        splits[split] = cases
    result = dict(
        schema_version=1,
        pack_id="pwr-ai-research-v1",
        public_corpus=True,
        blind_test=False,
        physical_scope="Reduced VVER, near-nominal two-second research tasks; no new plant qualification",
        splits=splits,
    )
    validate_pack(result)
    return result


def validate_pack(pack):
    if (
        not isinstance(pack, dict)
        or pack.get("schema_version") != 1
        or set(pack.get("splits", {})) != {"train", "dev", "test"}
    ):
        raise ValueError("Invalid benchmark pack")
    if pack.get("public_corpus") is not True or pack.get("blind_test") is not False:
        raise ValueError("Research corpus v1 is public, never a blind test")
    ids = set()
    groups = {}
    seeds = set()
    for split, cases in pack["splits"].items():
        if not cases:
            raise ValueError("Empty split")
        groups[split] = set()
        for case in cases:
            identifier = case.get("id")
            if (
                not isinstance(identifier, str)
                or re.fullmatch(r"[a-z]+-[a-z]+-[0-9]{2}", identifier) is None
            ):
                raise ValueError("Case ID must be a safe split-family-index identifier")
            if not identifier.startswith(split + "-" + str(case.get("family")) + "-"):
                raise ValueError("Case ID split/family mismatch")
            if case["id"] in ids or case["family"] not in FAMILIES:
                raise ValueError("Duplicate case ID or unknown family")
            ids.add(case["id"])
            cfg = EnvConfig.from_dict(case["config"])
            if cfg.action_channels != ("rod_command",) or cfg.task != "hold_power":
                raise ValueError("Benchmark v1 requires rod-only full-horizon tasks")
            if cfg.seed is None or cfg.seed in seeds:
                raise ValueError("Missing/overlapping episode seed")
            seeds.add(cfg.seed)
            group = cfg.as_dict()
            group.pop("seed")
            if case["group_id"] != digest(group):
                raise ValueError("Group signature mismatch")
            groups[split].add(case["group_id"])
            label = "sensor_fault" if cfg.sensor_faults[0] else "healthy"
            if case["diagnosis_label"] != label:
                raise ValueError("Diagnosis label mismatch")
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if groups[a] & groups[b]:
            raise ValueError("Configuration leakage between splits")
    if len(pack["splits"]["test"]) != 80 or any(
        sum(c["family"] == f for c in pack["splits"]["test"]) != 20 for f in FAMILIES
    ):
        raise ValueError("Test corpus requires 20 cases per family")
    return pack
