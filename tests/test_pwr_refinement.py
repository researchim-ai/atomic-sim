"""Comparison admission controls; synthetic gate data are not physical evidence."""

from copy import deepcopy

import pytest

from scripts.verify_pwr_refinement import (
    BUDGETS, compare, event_roots, validate_samples, verify,
)


def rows():
    return [dict(time_s=t, scram_latched=False, **{k: 10 * v for k, v in BUDGETS.items()}) for t in (0.0, 5.0, 10.0)]


def result():
    return dict(events=[dict(events=[dict(kind="rod_stop", t=2.0)], protection_event_s=None)])


def test_identical_full_trajectories_and_event_roots():
    comparison = compare(rows(), rows(), result(), result(), 10)
    assert comparison["passed"] and comparison["common_samples"] == 3
    assert comparison["sampled_extrema_only"]
    assert all(r["maximum_absolute"] == 0 for r in comparison["differences"].values())


@pytest.mark.parametrize("key", list(BUDGETS))
def test_each_observable_budget_detects_interior_errors(key):
    changed = rows()
    changed[1][key] += 2 * BUDGETS[key]
    comparison = compare(changed, rows(), result(), result(), 10)
    assert not comparison["passed"]
    assert comparison["differences"][key]["time_s"] == 5
    assert comparison["differences"][key]["final_absolute"] == 0


@pytest.mark.parametrize("change", ["prefix", "interior", "duplicate", "nan", "missing", "boolean", "discrete"])
def test_refuse_incomplete_nonfinite_or_ambiguous_samples(change):
    changed = rows()
    if change == "prefix": changed.pop()
    elif change == "interior": changed.pop(1)
    elif change == "duplicate": changed[1]["time_s"] = 0
    elif change == "nan": changed[1]["fuel_temperature_k"] = float("nan")
    elif change == "missing": del changed[1]["mass_flow_kg_s"]
    elif change == "boolean": changed[1]["primary_pressure_pa"] = True
    else: changed[1]["scram_latched"] = 1
    with pytest.raises(ValueError): validate_samples(changed, 10)


@pytest.mark.parametrize("change", ["root", "kind", "missing", "extra", "discrete"])
def test_event_topology_time_and_discrete_state_are_separate_gates(change):
    actual, observations = result(), rows()
    if change == "root": actual["events"][0]["events"][0]["t"] += 0.03
    elif change == "kind": actual["events"][0]["events"][0]["kind"] = "other"
    elif change == "missing": actual["events"] = []
    elif change == "extra": actual["events"][0]["protection_event_s"] = 4.0
    else: observations[1]["scram_latched"] = True
    assert not compare(observations, rows(), actual, result(), 10)["passed"]


@pytest.mark.parametrize("value", [True, float("nan"), None])
def test_nonfinite_event_roots_refused(value):
    changed = result()
    changed["events"][0]["events"][0]["t"] = value
    with pytest.raises(ValueError): event_roots(changed)


def test_event_outside_full_horizon_refused():
    changed = result()
    changed["events"][0]["events"][0]["t"] = 11.0
    with pytest.raises(ValueError, match="horizon"):
        compare(rows(), rows(), changed, result(), 10)


@pytest.mark.parametrize("jobs", [0, 5, True])
def test_worker_budget(tmp_path, jobs):
    with pytest.raises(ValueError, match="Worker"):
        verify(tmp_path / "out", jobs=jobs)
    assert not (tmp_path / "out").exists()


def test_actual_initial_and_advanced_observations_have_si_fields():
    from atomic_model_controlled_plant.qualification import sampled_observation
    from atomic_model_controlled_plant.session import Session

    session = Session()
    initial = sampled_observation(session.plant)
    session.advance(0.1, commands={"scram": True})
    advanced = sampled_observation(session.plant)
    assert set(BUDGETS) <= initial.keys()
    assert initial["time_s"] == 0 and advanced["time_s"] == 0.1
    assert not initial["scram_latched"] and advanced["scram_latched"]
    assert advanced["thermal_power_w"] > advanced["fission_power_w"] > 0
    assert 300 < advanced["core_temperature_k"] < 650


def test_failed_campaigns_continue_and_cannot_promote_prefixes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import verify_pwr_refinement as driver

    calls = []
    def refuse(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(driver.subprocess, "run", refuse)
    report = verify(tmp_path / "out", cases=["scram"], jobs=2)
    assert report["completed"] and not report["passed"]
    assert not report["complete_default_matrix"]
    assert len(calls) == 4 and len(report["runs"]) == 4
    assert all(not run["passed"] for run in report["runs"])


def test_reuse_refuses_changed_protocol_before_creating_output(tmp_path):
    from atomic_model_training.spec import atomic_json
    from atomic_model_multiphysics.provenance import seal
    from scripts.verify_pwr_refinement import declaration

    previous = tmp_path / "previous"
    previous.mkdir()
    declared = deepcopy(declaration(["scram"]))
    declared["absolute_budgets_si"]["fuel_temperature_k"] *= 100
    atomic_json(previous / "protocol.json", seal(declared))
    with pytest.raises(ValueError, match="identical source"):
        verify(tmp_path / "out", cases=["scram"], reuse_from=previous)
    assert not (tmp_path / "out").exists()


def test_native_campaign_samples_and_checkpoint_admitted_then_corruption_refused(tmp_path, monkeypatch):
    import json
    from atomic_model_controlled_plant.qualification import CASES, run_case
    from atomic_model_controlled_plant.session import identity
    from atomic_model_multiphysics.provenance import seal, sha
    from atomic_model_training.spec import atomic_json, read_json
    from scripts.verify_pwr_refinement import PROFILES, load_evidence

    # Short integration verifies the file contract only, never a full 60s SCRAM claim.
    monkeypatch.setitem(CASES, "scram", dict(duration_s=5.0, changes=[(0.0, dict(scram=True))]))
    directory = tmp_path / "campaign"
    result = run_case("scram", output=directory, **PROFILES["cap-2"])
    assert result["full_horizon_accepted"]
    accepted, samples = load_evidence(directory, "scram", "cap-2", identity())
    assert len(samples) == 2 and samples[0]["time_s"] == 0 and samples[1]["time_s"] == 5
    assert accepted["observations_sha256"] == sha(directory / "observations.jsonl")
    # A consistently rehashed envelope still cannot misdeclare the step cap.
    checkpoint_path = directory / "checkpoint-envelope.json"
    checkpoint_file = read_json(checkpoint_path)
    checkpoint = deepcopy(checkpoint_file["payload"]["payload"])
    checkpoint["config"]["maximum_internal_step_s"] = 1.0
    changed = seal(checkpoint)
    atomic_json(checkpoint_path, seal(changed))
    result["checkpoint_sha256"] = changed["sha256"]
    atomic_json(directory / "result.json", seal(result))
    with pytest.raises(ValueError, match="internal step"):
        load_evidence(directory, "scram", "cap-2", identity())
    atomic_json(checkpoint_path, checkpoint_file)
    result["checkpoint_sha256"] = checkpoint_file["payload"]["sha256"]
    atomic_json(directory / "result.json", seal(result))
    original = read_json(directory / "result.json")["payload"]
    with (directory / "observations.jsonl").open("a") as stream:
        stream.write(json.dumps(samples[-1]) + "\n")
    with pytest.raises(ValueError, match="checksum"):
        load_evidence(directory, "scram", "cap-2", identity())
    original["observations_sha256"] = sha(directory / "observations.jsonl")
    atomic_json(directory / "result.json", seal(original))
    with pytest.raises(ValueError, match="samples"):
        load_evidence(directory, "scram", "cap-2", identity())


def test_completed_refinement_reuse_compares_again_without_subprocesses(tmp_path, monkeypatch):
    from scripts import verify_pwr_refinement as driver
    from atomic_model_training.spec import atomic_json
    from atomic_model_multiphysics.provenance import seal

    monkeypatch.setitem(driver.CASES, "scram", dict(duration_s=10., changes=[]))
    previous = tmp_path / "previous"
    previous.mkdir()
    declared = driver.declaration(["scram"])
    declared["driver_sha256"] = "a" * 64  # A different checker requires fresh admission.
    atomic_json(previous / "protocol.json", seal(declared))
    for name in driver.PROFILES:
        directory = previous / ("scram--" + name)
        directory.mkdir()
        (directory / "result.json").write_text("synthetic orchestration fixture")
    calls = []
    def admit(directory, case, profile, source):
        calls.append(profile)
        return result(), rows()
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed reusable evidence should not run subprocesses")
    monkeypatch.setattr(driver, "load_evidence", admit)
    monkeypatch.setattr(driver.subprocess, "run", forbidden)
    report = verify(tmp_path / "out", cases=["scram"], reuse_from=previous)
    assert report["passed"] and not report["complete_default_matrix"]
    assert all(r["reused"] for r in report["runs"])
    assert len(calls) == 8  # Admission on reuse and again before comparison.
