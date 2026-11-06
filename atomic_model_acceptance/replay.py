"""Local portable replay with predeclared numerical tolerances."""

import hashlib
import math
from pathlib import Path

import numpy as np

from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_research.project import Project
from atomic_model_research.release import identity
from atomic_model_training.spec import atomic_json, read_json

PROTOCOL = dict(
    schema_version=2,
    protocol_id="local-core-replay-v2",
    state_scaled_error_max=1e-8,
    event_time_absolute_error_s_max=1e-8,
    balance_relative_max=1e-5,
    execution_scope="local installation and source-bound export/import/replay",
    remote_execution_required=False,
    ci_required=False,
)


def source_identity():
    return dict(
        base=identity(),
        files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        resources={
            str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.joinpath("data").rglob("*"))
            if p.is_file()
        },
    )


def make_request(specification, output):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    project = Project.create(output / "project", specification)
    receipt = project.run(output / "reference-run")
    if not receipt["payload"]["report"]["completed"]:
        raise ValueError("Reference campaign incomplete")
    project.export(output / "project.zip")
    atomic_json(output / "source.json", identity())
    request = dict(
        schema_version=1,
        source=source_identity(),
        protocol=PROTOCOL,
        project_archive_sha256=hashlib.sha256((output / "project.zip").read_bytes()).hexdigest(),
        reference=receipt,
        validated=False,
        plant_referenced=False,
    )
    atomic_json(output / "request.json", seal(request))
    return request


def verify_packet(directory):
    directory = Path(directory)
    request = unseal(read_json(directory / "request.json"))
    if (
        request["source"] != source_identity()
        or request["protocol"] != PROTOCOL
        or read_json(directory / "source.json") != identity()
    ):
        raise ValueError("Packet source/protocol differs")
    from atomic_model_research.project import MAX_TOTAL, regular_bytes

    raw = regular_bytes(directory / "project.zip", limit=MAX_TOTAL)
    if hashlib.sha256(raw).hexdigest() != request["project_archive_sha256"]:
        raise ValueError("Packet project archive checksum differs")
    return request


def _report(receipt):
    p = unseal(receipt)
    report = p["report"]
    if (
        not report["completed"]
        or not report["results"]
        or any(r["status"] != "completed" for r in report["results"])
    ):
        raise ValueError("Incomplete campaign is not replay evidence")
    ids = [r["id"] for r in report["results"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate replay result ids")
    return p, {r["id"]: r["receipt"] for r in report["results"]}


def compare(request_envelope, replay_receipt):
    request = unseal(request_envelope)
    if request["source"] != source_identity() or request["protocol"] != PROTOCOL:
        raise ValueError("Replay source/protocol differs")
    a, reference = _report(request["reference"])
    b, remote = _report(replay_receipt)
    if (
        a["revision"] != b["revision"]
        or a["export_schema"] != b["export_schema"]
        or set(reference) != set(remote)
    ):
        raise ValueError("Replay project/schema/case inventory differs")
    comparisons = []
    for name in sorted(reference):
        x, y = reference[name], remote[name]
        for key in ["case_sha256", "source_sha256", "state_schema"]:
            if x[key] != y[key]:
                raise ValueError("Replay model/case/units differ")
        xv, yv = np.asarray(x["state"], float), np.asarray(y["state"], float)
        if xv.shape != yv.shape or not np.isfinite(xv).all() or not np.isfinite(yv).all():
            raise ValueError("Replay finite state dimensions differ")
        error = float(np.max(abs(xv - yv) / np.maximum(1.0, abs(xv))))
        hx = x["checkpoint"]["payload"]["history"]
        hy = y["checkpoint"]["payload"]["history"]
        if len(hx) != len(hy) or not hx:
            raise ValueError("Replay history dimensions differ")
        for checkpoint in (x["checkpoint"], y["checkpoint"]):
            unseal(checkpoint)
        events_equal = True
        for u, v in zip(hx, hy):
            if any(u[k] != v[k] for k in ["duration_s", "commands", "options", "events"]):
                events_equal = False
            t, s = u["protection_event_s"], v["protection_event_s"]
            if (t is None) != (s is None):
                events_equal = False
            elif t is not None and (
                not math.isfinite(t)
                or not math.isfinite(s)
                or abs(t - s) > PROTOCOL["event_time_absolute_error_s_max"]
            ):
                events_equal = False
        balances = []
        for receipt in (x, y):
            for key in ["energy_relative", "mass_relative", "boron_relative"]:
                value = receipt["balances"][key]
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid balance receipt")
                balances.append(value)
        accepted = (
            error <= PROTOCOL["state_scaled_error_max"]
            and events_equal
            and max(balances) <= PROTOCOL["balance_relative_max"]
        )
        comparisons.append(
            dict(
                case=name,
                state_scaled_error=error,
                events_equal=events_equal,
                maximum_balance_relative=max(balances),
                accepted=accepted,
            )
        )
    return dict(
        source=source_identity(),
        request_sha256=request_envelope["sha256"],
        cases=comparisons,
        numerical_replay_accepted=all(c["accepted"] for c in comparisons),
        verification_scope="local numerical replay",
        remote_execution_required=False,
        ci_required=False,
        validated=False,
        plant_referenced=False,
        physical_validation_claimed=False,
    )
