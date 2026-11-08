"""Evidence admission controls; fixtures are synthetic, not physical trajectories."""

import pytest
from scripts.verify_pwr_horizons import admit_result

from atomic_model_controlled_plant.qualification import protocol
from atomic_model_controlled_plant.session import identity
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_workspace.common import digest


def fixture(directory):
    declared = protocol()
    checkpoint = seal(dict(revision=12))
    trace = directory / "trace.jsonl"
    trace.write_text("synthetic gate fixture\n")
    result = dict(
        source=identity(),
        case="scram",
        protocol_sha256=digest(declared),
        full_horizon_accepted=True,
        requested_horizon_s=60.0,
        completed_horizon_s=60.0,
        failure=None,
        checks={
            key: True
            for key in (
                "energy_relative",
                "mass_relative",
                "boron_relative",
                "feed_nonnegative",
                "primary_scope",
                "shutdown_fission",
            )
        },
        trace_sha256=sha(trace),
        checkpoint_sha256=checkpoint["sha256"],
        steps=12,
        ledger={key: 0.0 for key in ("energy_relative", "mass_relative", "boron_relative")},
        validated=False,
        plant_referenced=False,
    )
    atomic_json(directory / "protocol.json", seal(declared))
    atomic_json(directory / "checkpoint-envelope.json", seal(checkpoint))
    atomic_json(directory / "result.json", seal(result))
    return result


def test_json_protocol_roundtrip_and_gate_admission(tmp_path):
    fixture(tmp_path)
    # Native protocol has tuples; canonical JSON identity survives lists on disk.
    assert admit_result(tmp_path, "scram", identity())["full_horizon_accepted"]


@pytest.mark.parametrize(
    "change",
    ["prefix", "claim", "missing-check", "budget", "trace", "source", "checkpoint", "protocol"],
)
def test_gate_refuses_rehashed_inconsistent_evidence(tmp_path, change):
    result = fixture(tmp_path)
    if change == "prefix":
        result["completed_horizon_s"] = 20.0
    elif change == "claim":
        result["full_horizon_accepted"] = 1
    elif change == "missing-check":
        result["checks"].pop("shutdown_fission")
    elif change == "budget":
        result["ledger"]["energy_relative"] = 1.0
    elif change == "trace":
        (tmp_path / "trace.jsonl").write_text("changed")
    elif change == "source":
        result["source"] = {}
    elif change == "checkpoint":
        result["checkpoint_sha256"] = "0" * 64
    elif change == "protocol":
        value = read_json(tmp_path / "protocol.json")["payload"]
        value["cases"]["scram"]["duration_s"] = 20
        atomic_json(tmp_path / "protocol.json", seal(value))
    atomic_json(tmp_path / "result.json", seal(result))
    with pytest.raises(ValueError):
        admit_result(tmp_path, "scram", identity())


def test_completed_case_reuse_rechecks_evidence_without_rerunning(tmp_path, monkeypatch):
    from scripts import verify_pwr_horizons as profile
    from scripts.verify_pwr_horizons import verify

    old = tmp_path / "old" / "scram"
    old.mkdir(parents=True)
    result = fixture(old)
    atomic_json(
        old / "replay.json",
        seal(
            dict(
                replay_equal=True,
                checkpoint_sha256=result["checkpoint_sha256"],
                revision=result["steps"],
            )
        ),
    )
    before = {p.name: p.read_bytes() for p in old.iterdir()}

    def forbidden(*args, **kwargs):
        raise AssertionError("Completed source-matching case must not rerun")

    monkeypatch.setattr(profile.subprocess, "run", forbidden)
    report = verify(tmp_path / "new", cases=["scram"], reuse_from=old.parent, jobs=2)
    assert report["passed"] and not report["complete_default_matrix"]
    assert report["cases"][0]["reused_from"] == str(old)
    assert {p.name: p.read_bytes() for p in old.iterdir()} == before


def test_corrupt_replay_is_not_reused_and_other_cases_continue(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from scripts import verify_pwr_horizons as profile

    old = tmp_path / "old" / "scram"
    old.mkdir(parents=True)
    result = fixture(old)
    atomic_json(
        old / "replay.json",
        seal(
            dict(
                replay_equal=False,
                checkpoint_sha256=result["checkpoint_sha256"],
                revision=result["steps"],
            )
        ),
    )
    calls = []

    def failed(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(profile.subprocess, "run", failed)
    report = profile.verify(
        tmp_path / "new", cases=["scram", "nominal"], reuse_from=old.parent, jobs=2
    )
    assert report["completed"] and not report["passed"]
    assert len(calls) == 2
    assert [r["case"] for r in report["cases"]] == ["scram", "nominal"]
    assert "reuse_rejected" in report["cases"][0]
    assert all(not r["passed"] for r in report["cases"])


@pytest.mark.parametrize("jobs", [0, 5, True])
def test_worker_budget_is_explicit(tmp_path, jobs):
    from scripts.verify_pwr_horizons import verify

    with pytest.raises(ValueError, match="worker budget"):
        verify(tmp_path / "output", cases=["scram"], jobs=jobs)
