"""Recalculate controlled-program gates and admit only fully replayed histories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from atomic_model_controlled_plant.qualification import CASES, CRITERIA, protocol, publish
from atomic_model_controlled_plant.session import Session, identity
from atomic_model_workspace.common import digest
from scripts.extended_plant_release import envelope


def verify_run(folder):
    folder = Path(folder)
    r = envelope(folder / "result.json")
    declared = envelope(folder / "protocol.json")
    if (
        declared != json.loads(json.dumps(protocol()))
        or r["source"] != identity()
        or r["protocol_sha256"] != digest(declared)
    ):
        raise ValueError("Controlled source/protocol changed")
    if hashlib.sha256((folder / "trace.jsonl").read_bytes()).hexdigest() != r["trace_sha256"]:
        raise ValueError("Controlled trajectory changed")
    c = envelope(folder / "checkpoint-envelope.json")
    if (
        digest(c["payload"]) != c["sha256"]
        or c["sha256"] != r["checkpoint_sha256"]
        or c["payload"]["revision"] != r["steps"]
    ):
        raise ValueError("Controlled checkpoint content/revision mismatch")
    if c["payload"]["model_id"] != "pwr-reduced-flow-controlled" or c["payload"]["config"][
        "implementation_sha256"
    ] != digest(identity()):
        raise ValueError("Controlled checkpoint implementation mismatch")
    name = r["case"]
    if name not in CASES or r["requested_horizon_s"] != CASES[name]["duration_s"]:
        raise ValueError("Controlled horizon changed")
    checks = {
        k: bool(0 <= r["ledger"][k] <= CRITERIA[k])
        for k in ["energy_relative", "mass_relative", "boron_relative"]
    }
    checks["feed_nonnegative"] = bool(r["extrema"]["minimum_feed_flow_kg_s"] >= 0.0)
    checks["primary_scope"] = bool(
        r["extrema"]["maximum_primary_quality"] <= CRITERIA["maximum_primary_quality"]
    )
    if name == "nominal":
        checks["stationary_fission"] = bool(
            r["extrema"]["maximum_fission_drift"] <= CRITERIA["nominal_power_drift"]
        )
    if name in ("scram", "coastdown"):
        checks["shutdown_fission"] = bool(
            r["final_state"][0] < CRITERIA["scram_final_relative_fission"]
        )
    if name.startswith("natural"):
        rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
        checks["zero_shaft_and_head"] = all(row["omega_rad_s"] == 0.0 for row in rows)
        checks["positive_primary_flow"] = all(row["primary_flow_kg_s"] > 0.0 for row in rows)
        h = r["initial_natural_heads"]
        checks["buoyancy_drives_friction"] = bool(
            h["buoyancy_head_pa"] > 0.0 and abs(h["head_defect_pa"]) < 0.01
        )
    if (
        name.startswith("load_rejection")
        and r["failure"] is None
        and r["completed_horizon_s"] == r["requested_horizon_s"]
    ):
        setting = r["load_rejection_settling"]
        span = setting["tail_relative_ranges"]
        checks.update(
            settled_final_rate=bool(setting["final_normalized_rate_per_s"] <= 1e-5),
            settled_power=bool(span["relative_fission"] <= 1e-3),
            settled_primary_flow=bool(span["primary_flow_kg_s"] <= 1e-3),
            settled_feed_flow=bool(span["feed_flow_kg_s"] <= 1e-2),
        )
    if r["checks"] != checks:
        raise ValueError("Controlled checks do not reproduce")
    accepted = bool(
        r["failure"] is None
        and r["completed_horizon_s"] == r["requested_horizon_s"]
        and all(checks.values())
    )
    if accepted != r["full_horizon_accepted"]:
        raise ValueError("Controlled acceptance differs")
    return r


def replay(folder, output):
    r = verify_run(folder)
    saved = envelope(Path(folder) / "checkpoint-envelope.json")
    session = Session.restore(saved)
    if session.ledger.report() != r["ledger"] or not np.array_equal(
        session.plant.pack(), r["final_state"]
    ):
        raise ValueError("Replayed controlled ledger/final stocks differ")
    receipt = dict(
        source=identity(),
        checkpoint_sha256=saved["sha256"],
        replay_equal=True,
        revision=session.revision,
        case=r["case"],
        full_horizon_accepted=r["full_horizon_accepted"],
    )
    publish(output, receipt)
    return receipt


def admission(folders, replays):
    matching = set()
    for p in replays:
        r = envelope(p)
        if r["source"] != identity() or r["replay_equal"] is not True:
            raise ValueError("Controlled replay receipt mismatch")
        matching.add(r["checkpoint_sha256"])
    admitted = []
    candidates = []
    rejected = []
    for folder in folders:
        r = verify_run(folder)
        if not r["full_horizon_accepted"]:
            rejected.append(dict(case=r["case"], checks=r["checks"], failure=r["failure"]))
            continue
        c = envelope(Path(folder) / "checkpoint-envelope.json")["payload"]
        row = dict(
            case=r["case"],
            configuration=c["config"],
            model_id=c["model_id"],
            duration_s=r["requested_horizon_s"],
            commands=CASES[r["case"]]["changes"],
            solver=r["solver"],
            source=r["source"],
            evidence_sha256=digest(r),
            checkpoint_sha256=r["checkpoint_sha256"],
            scope="Exact accepted command program/configuration; arbitrary perturbations are not promoted",
        )
        (admitted if r["checkpoint_sha256"] in matching else candidates).append(row)
    return dict(
        schema_version=1,
        admitted=admitted,
        candidates_awaiting_full_replay=candidates,
        rejected=rejected,
        validated=False,
        plant_referenced=False,
        old_model_failures_superseded=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("replay")
    a.add_argument("run", type=Path)
    a.add_argument("--output", required=True, type=Path)
    a = sub.add_parser("admit")
    a.add_argument("runs", nargs="+", type=Path)
    a.add_argument("--replays", nargs="*", type=Path, default=[])
    a.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.command == "replay":
        replay(args.run, args.output)
    else:
        publish(args.output, admission(args.runs, args.replays))


if __name__ == "__main__":
    main()
