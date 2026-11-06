"""Verify full-horizon receipts and generate only admitted agent scenarios.

Reports with a failed prefix, changed code, changed trace, missing checks, or
short horizon are rejected. An admission applies to one exact configuration.
"""

import argparse
import hashlib
import json
from pathlib import Path

from atomic_model_plant_extension.qualification import CASES, CRITERIA, protocol
from atomic_model_plant_extension.session import identity
from atomic_model_workspace.common import digest


def envelope(path):
    saved = json.loads(Path(path).read_text())
    if set(saved) != {"payload", "sha256"} or digest(saved["payload"]) != saved["sha256"]:
        raise ValueError("Evidence envelope mismatch:" + str(path))
    return saved["payload"]


def verify_run(folder):
    folder = Path(folder)
    r = envelope(folder / "result.json")
    declared = envelope(folder / "protocol.json")
    if (
        declared != json.loads(json.dumps(protocol()))
        or r["source"] != identity()
        or r["protocol_sha256"] != digest(declared)
    ):
        raise ValueError("Plant source/protocol changed")
    if r["trace_sha256"] != hashlib.sha256((folder / "trace.jsonl").read_bytes()).hexdigest():
        raise ValueError("Trajectory trace changed")
    c = envelope(folder / "checkpoint-envelope.json")
    if c["sha256"] != r["checkpoint_sha256"] or c["payload"]["revision"] != r["steps"]:
        raise ValueError("Checkpoint identity mismatch")
    if digest(c["payload"]) != c["sha256"]:
        raise ValueError("Checkpoint content changed")
    name = r["case"]
    if name not in CASES or r["requested_horizon_s"] != CASES[name]["duration_s"]:
        raise ValueError("Requested case/horizon changed")
    required = {
        "energy_relative",
        "mass_relative",
        "boron_relative",
        "feed_nonnegative",
        "primary_scope",
    }
    if name == "nominal":
        required.add("stationary_fission")
    if name in ("scram", "coastdown"):
        required.add("shutdown_fission")
    if set(r["checks"]) != required or not all(type(v) is bool for v in r["checks"].values()):
        raise ValueError("Missing/invalid qualification checks")
    numerical = all(
        r["ledger"][k] <= CRITERIA[k]
        for k in ["energy_relative", "mass_relative", "boron_relative"]
    )
    numerical &= r["extrema"]["minimum_feed_flow_kg_s"] >= 0.0
    numerical &= r["extrema"]["maximum_primary_quality"] <= CRITERIA["maximum_primary_quality"]
    if name == "nominal":
        numerical &= r["extrema"]["maximum_fission_drift"] <= CRITERIA["nominal_power_drift"]
    if name in ("scram", "coastdown"):
        numerical &= r["final_state"][0] < CRITERIA["scram_final_relative_fission"]
    accepted = (
        r["failure"] is None
        and r["completed_horizon_s"] == CASES[name]["duration_s"]
        and numerical
        and all(r["checks"].values())
    )
    if r["full_horizon_accepted"] != accepted:
        raise ValueError("Reported acceptance differs from recalculated decision")
    return r


def scenarios(folders, replay_receipts=()):
    admitted = []
    candidates = []
    rejected = []
    replayed = set()
    for receipt in replay_receipts:
        replay = envelope(receipt)
        if replay.get("replay_equal") is not True or not isinstance(
            replay.get("checkpoint_sha256"), str
        ):
            raise ValueError("Invalid full-history replay receipt")
        replayed.add(replay["checkpoint_sha256"])
    for folder in folders:
        r = verify_run(folder)
        if not r["full_horizon_accepted"]:
            rejected.append(dict(case=r["case"], reason=r["failure"] or r["checks"]))
            continue
        target = admitted if r["checkpoint_sha256"] in replayed else candidates
        target.append(
            dict(
                case=r["case"],
                duration_s=r["completed_horizon_s"],
                commands=CASES[r["case"]]["changes"],
                configuration="pwr-reduced-seven-bank",
                solver=r["solver"],
                source=r["source"],
                evidence_sha256=digest(r),
                scope="Exact qualified command program only; no arbitrary perturbation envelope",
            )
        )
    return dict(
        schema_version=1,
        admitted=admitted,
        candidates_awaiting_full_history_replay=candidates,
        rejected=rejected,
        experimental_validation=False,
        natural_circulation_admitted=False,
        legacy_vver_failures_superseded=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replays", nargs="*", type=Path, default=[])
    a = p.parse_args()
    result = scenarios(a.runs, a.replays)
    with a.output.open("x") as stream:
        json.dump(dict(payload=result, sha256=digest(result)), stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
