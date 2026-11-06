"""Checkpoint after each accepted transaction; resume verifies the command prefix."""

import hashlib
from pathlib import Path

from atomic_model_controlled_plant.session import Session
from atomic_model_controlled_plant.session import identity as core_identity
from atomic_model_workspace.common import digest

from .schema import state_schema
from .spec import TIERS, admit


def identity():
    return dict(
        core=core_identity(),
        implementation={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
    )


def run_episode(case, *, template=None, saved=None, emit=lambda value: None):
    case = admit(case)
    source = digest(identity())
    if saved is None:
        session = Session() if template is None else template.new()
    else:
        if (
            saved.get("case_sha256") != digest(case)
            or saved.get("source_sha256") != source
            or digest(saved["checkpoint"]["payload"]) != saved["checkpoint"]["sha256"]
        ):
            raise ValueError("Resume source/case/checkpoint mismatch")
        session = Session.restore(saved["checkpoint"], max_replay_steps=256)
        if session.revision > len(case["steps"]):
            raise ValueError("Resume exceeds episode")
        for entry, step in zip(session.history, case["steps"]):
            if (
                entry["commands"] != step["commands"]
                or entry["duration_s"] != step["duration_s"]
                or any(entry["options"][k] != v for k, v in TIERS[case["tier"]].items())
            ):
                raise ValueError("Resume command/options prefix mismatch")

    def receipt(status):
        return dict(
            status=status,
            case_sha256=digest(case),
            source_sha256=source,
            checkpoint=session.checkpoint(),
            validated=False,
            plant_referenced=False,
        )

    emit(receipt("running"))
    for step in case["steps"][session.revision :]:
        session.advance(step["duration_s"], commands=step["commands"], **TIERS[case["tier"]])
        emit(receipt("running"))
    result = receipt("completed")
    result["state"] = list(map(float, session.plant.pack()))
    result["state_schema"] = state_schema(session.plant)
    result["balances"] = session.ledger.report()
    return result
