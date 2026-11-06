"""Recompute every recorded step under current code before issuing new evidence.

This is an explicit recertification tool, NOT a permissive checkpoint loader.
The normal API still rejects changed code. Every old state/event/energy hash
must reproduce exactly; otherwise no new result or replay receipt is issued.
"""

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

import numpy as np

from atomic_model_controlled_plant.qualification import protocol, publish
from atomic_model_controlled_plant.session import Session, identity
from atomic_model_plant_extension.session import identity as base_identity
from atomic_model_workspace.common import digest
from scripts.controlled_plant_release import verify_run
from scripts.extended_plant_release import envelope


def recertify(folder, output, expected_old):
    folder = Path(folder)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    old = envelope(folder / "result.json")
    old_protocol = envelope(folder / "protocol.json")
    if (
        old["source"] != expected_old
        or old_protocol["source"] != expected_old
        or old["protocol_sha256"] != digest(old_protocol)
    ):
        raise ValueError("Unexpected parent source/protocol")
    saved = envelope(folder / "checkpoint-envelope.json")
    if digest(saved["payload"]) != saved["sha256"] or saved["sha256"] != old["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint mismatch")
    # Only the source bindings are explicitly upgraded. The strict
    # restore then recomputes all original calls and checks ALL old hashes.
    candidate = deepcopy(saved)
    candidate["payload"]["source"] = base_identity()
    candidate["payload"]["config"]["implementation_sha256"] = digest(identity())
    candidate["sha256"] = digest(candidate["payload"])
    started = time.monotonic()
    payload = candidate["payload"]
    if payload["revision"] != len(payload["history"]) or len(payload["history"]) > 10000:
        raise ValueError("Recertification replay budget/revision mismatch")
    print("START",folder.name, len(payload["history"]), flush=True)
    session = Session(**payload["config"])
    print("INITIALIZED",folder.name,flush=True)
    for index, entry in enumerate(payload["history"], 1):
        if type(entry.get("reset_solver_before_step")) is not bool:
            raise ValueError("Invalid reset journal")
        if entry["reset_solver_before_step"]:
            session.plant._hybrid = None
            session.plant._hybrid_signature = None
            session.plant._continuation = None
            session.plant._continuation_signature = None
        session.advance(entry["duration_s"], commands=entry["commands"], **entry["options"])
        actual = session.history[-1]
        if digest(actual) != digest(entry):
            changed = [key for key in set(actual) | set(entry) if actual.get(key) != entry.get(key)]
            raise ValueError(f"Recertification mismatch {folder.name} step{index}: {changed}")
        if index <= 5 or index % 100 == 0:
            print(folder.name, index, "/", len(payload["history"]), flush=True)
    if session.checkpoint() != candidate:
        raise ValueError("Final recertified checkpoint differs")
    if session.ledger.report() != old["ledger"] or not np.array_equal(
        session.plant.pack(), old["final_state"]
    ):
        raise ValueError("Current-source numerical replay differs from parent")
    output.mkdir(parents=True)
    declared = protocol()
    publish(output / "protocol.json", declared)
    shutil.copyfile(folder / "trace.jsonl", output / "trace.jsonl")
    current = session.checkpoint()
    publish(output / "checkpoint-envelope.json", current)
    result = deepcopy(old)
    result.update(
        source=identity(),
        protocol_sha256=digest(declared),
        checkpoint_sha256=current["sha256"],
        initialization=session.initialization,
        wall_seconds=time.monotonic() - started,
        recertification=dict(
            parent_result_sha256=digest(old),
            parent_checkpoint_sha256=saved["sha256"],
            parent_source=expected_old,
            full_history_recomputed=True,
            all_recorded_hashes_equal=True,
            reason="Corrected inherited passport notes and exact-boundary activation guard; every old numerical state/event/energy hash recomputed, acceptance limits unchanged",
        ),
    )
    publish(output / "result.json", result)
    verify_run(output)
    publish(
        output / "replay.json",
        dict(
            source=identity(),
            checkpoint_sha256=current["sha256"],
            replay_equal=True,
            revision=session.revision,
            case=result["case"],
            full_horizon_accepted=result["full_horizon_accepted"],
        ),
    )
    return result["full_horizon_accepted"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--old-source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    old = json.loads(a.old_source.read_text())
    a.output.mkdir(parents=True, exist_ok=a.resume)
    folders = sorted(path.parent for path in a.input.glob("*/result.json"))

    rows = []
    pending = []
    for folder in folders:
        target = a.output / folder.name
        if target.exists():
            if not a.resume:
                raise FileExistsError(target)
            r = verify_run(target)
            receipt = envelope(target / "replay.json")
            if (
                receipt["source"] != identity()
                or receipt["checkpoint_sha256"] != r["checkpoint_sha256"]
                or receipt["replay_equal"] is not True
            ):
                raise ValueError("Existing recertification is not complete/current")
            rows.append(dict(case=folder.name, accepted=r["full_horizon_accepted"]))
        else:
            pending.append(folder)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = {
            pool.submit(recertify, folder, a.output / folder.name, old): folder.name
            for folder in pending
        }
        for future in as_completed(futures):
            name = futures[future]
            accepted = future.result()
            print(name, accepted, flush=True)
            rows.append(dict(case=name, accepted=accepted))
    rows.sort(key=lambda row: row["case"])
    publish(
        a.output / "recertification.json",
        dict(source=identity(), runs=rows, full_current_code_replay=True),
    )
    return 0 if all(row["accepted"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
