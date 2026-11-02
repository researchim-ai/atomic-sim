"""Independent IVPs, report reconstruction, tamper rejection and complete admission."""

import json
import shutil
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from physics.power.primary_steam import PrimarySteamCycle
from scripts.audit_primary_steam import audit
from scripts.primary_steam_references import _cached_reference, reference
from scripts.qualify_primary_steam import DRIVERS, compare_levels, matrix, reuse_packet, signature
from scripts.verify_primary_steam import BUDGET, assess
from scripts.verify_turbine import sha
from sdk.primary_steam_experiment import run_primary_steam_experiment


def test_refinement_failure_stops_before_next_family(tmp_path, monkeypatch):
    import sys

    from scripts import qualify_primary_steam as producer

    budget = json.loads(BUDGET.read_text())
    normalized = next(matrix(budget))[2]
    seen = []

    def candidates(_):
        for level in ("coarse", "medium", "fine", "never_start"):
            seen.append(level)
            yield "nominal", level, normalized

    def run(spec, folder):
        folder.mkdir()

    def reject(*args):
        raise ValueError("injected unrefined actual grid")

    monkeypatch.setattr(producer, "matrix", candidates)
    monkeypatch.setattr(producer, "run_primary_steam_experiment", run)
    monkeypatch.setattr(
        producer,
        "assess",
        lambda *args: dict(
            accepted=True, termination="completed", solver_statistics=dict(reported_mesh_steps=1)
        ),
    )
    monkeypatch.setattr(producer, "compare_levels", reject)
    output = tmp_path / "stopped"
    monkeypatch.setattr(sys, "argv", ["qualify", "--output", str(output)])
    with pytest.raises(ValueError, match="injected"):
        producer.main()
    assert seen == ["coarse", "medium", "fine"]
    assert not (output / "qualification.json").exists()
    assert json.loads((output / "rejection.json").read_text())["passed"] is False


@pytest.fixture
def budget():
    return json.loads(BUDGET.read_text())


@pytest.fixture
def spec(budget):
    raw = json.loads(Path("scenarios/ai05_primary_steam.json").read_text())
    raw.update(name="small_evidence", duration_s=0.2, step_s=0.1, timeline=[])
    raw["solver"] = dict(budget["fixed_solver"], max_step_s=0.01)
    return raw


@pytest.fixture(scope="module")
def packet(tmp_path_factory):
    raw = json.loads(Path("scenarios/ai05_primary_steam.json").read_text())
    budget = json.loads(BUDGET.read_text())
    raw.update(name="small_evidence", duration_s=0.2, step_s=0.1, timeline=[])
    raw["solver"] = dict(budget["fixed_solver"], max_step_s=0.01)
    folder = tmp_path_factory.mktemp("evidence") / "run"
    assert run_primary_steam_experiment(raw, folder)["status"] == "completed"
    return folder


def rewrite(folder, name, value):
    (folder / name).write_text(json.dumps(value))
    manifest = json.loads((folder / "manifest.json").read_text())
    manifest["artifacts"][name] = sha(folder / name)
    (folder / "manifest.json").write_text(json.dumps(manifest))


def test_independent_equations_never_call_production(spec, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Production model called by independent IVP")

    monkeypatch.setattr(PrimarySteamCycle, "derivatives", forbidden)
    monkeypatch.setattr(PrimarySteamCycle, "observation", forbidden)
    spec["name"] = "unique_independence_case"
    spec["duration_s"] = 0.0137
    at, boundary, _ = reference(spec)
    assert boundary is None and np.isfinite(at(0.0137)).all()


def test_reference_cache_keys_physics_and_detaches_mutable_outputs(spec):
    at, _, details = reference(spec)
    initial = at(0)
    before = _cached_reference.cache_info()
    other = deepcopy(spec)
    other.update(name="another_name", step_s=0.02)
    other["solver"]["rtol"] = 1e-8
    sample = at(0.1)
    expected = sample.copy()
    sample[:] = 123
    details[3]["capacity"][:] = 0
    details[-1][:] = 456
    reused, _, fresh = reference(other)
    assert _cached_reference.cache_info().hits == before.hits + 1
    assert np.array_equal(reused(0), initial)
    assert np.array_equal(reused(0.1), expected)
    assert np.all(fresh[3]["capacity"] > 0)
    before = _cached_reference.cache_info()
    other["coupling"]["ua_w_k"] += 1
    reference(other)
    assert _cached_reference.cache_info().misses == before.misses + 1
    altered = initial.copy()
    altered[-1] = 0.01
    continued, _, _ = reference(spec, initial_state=altered)
    assert continued(0)[-1] == 0.01


def test_independent_small_packet_and_checkpoint_continuation(packet):
    result = assess(packet)
    assert result["accepted"] and result["termination"] == "completed"
    assert result["metrics"]["power_absolute_w"] < 0.001


@pytest.mark.parametrize(
    "defect",
    [
        "energy_residual",
        "port_flux",
        "heat_integral",
        "flow",
        "primary_temperature",
        "thermodynamics",
        "report_maximum",
        "grid",
        "checksum",
        "boolean_state",
    ],
)
def test_corruption_cannot_pass_by_rehashing_a_manifest(packet, tmp_path, defect):
    folder = tmp_path / "forged"
    shutil.copytree(packet, folder)
    rows = json.loads((folder / "trajectory.json").read_text())
    report = json.loads((folder / "report.json").read_text())
    if defect == "energy_residual":
        rows[1]["energy_residual_j"] = 1
        report["energy_absolute_j"] = 1
        rewrite(folder, "report.json", report)
    elif defect == "port_flux":
        rows[1]["primary"]["loop"]["ports"]["s0.left"]["energy_flow_w"] += 10
    elif defect == "heat_integral":
        rows[1]["heat_transfer_energy_j"] += 1
    elif defect == "flow":
        rows[1]["primary"]["loop"]["mass_flow_kg_s"] += 1
    elif defect == "primary_temperature":
        rows[1]["primary"]["loop"]["fluid_temperature_k"]["s0"] += 1
    elif defect == "thermodynamics":
        rows[1]["steam_cycle"]["thermodynamics"]["outlet_quality"] += 0.01
    elif defect == "report_maximum":
        report["energy_absolute_j"] = 123
        rewrite(folder, "report.json", report)
    elif defect == "grid":
        rows[1]["phase"] = "after_control"
    elif defect == "boolean_state":
        rows[1]["primary"]["omega_rad_s"] = True
    if defect == "checksum":
        (folder / "trajectory.json").write_text("[]")
    else:
        rewrite(folder, "trajectory.json", rows)
    try:
        result = assess(folder)
    except (ValueError, KeyError):
        return
    assert not result["accepted"]


def test_shared_wrong_production_observation_does_not_self_certify(spec, tmp_path, monkeypatch):
    original = PrimarySteamCycle.observation

    def wrong(self, *args, **kwargs):
        row = original(self, *args, **kwargs)
        row["steam_cycle"]["thermodynamics"]["outlet_quality"] += 0.01
        return row

    monkeypatch.setattr(PrimarySteamCycle, "observation", wrong)
    folder = tmp_path / "wrong_derived_data"
    assert run_primary_steam_experiment(spec, folder)["status"] == "completed"
    result = assess(folder)
    assert not result["accepted"] and result["metrics"]["quality_absolute"] > 0.001


def test_complete_matrix_and_fixed_radau_tolerances(budget):
    rows = list(matrix(budget))
    assert len(rows) == 53 and len({name for name, _, _ in rows}) == 17
    for name in {name for name, _, _ in rows}:
        levels = {level: raw for n, level, raw in rows if n == name}
        for i, level in enumerate(budget["levels"]):
            assert levels[level]["solver"]["rtol"] == budget["fixed_solver"]["rtol"]
            assert levels[level]["solver"]["atol_scale"] == budget["fixed_solver"]["atol_scale"]
            assert (
                levels[level]["solver"]["max_step_s"]
                == levels["coarse"]["solver"]["max_step_s"] / 4**i
            )
    for name, horizon in budget["long_horizons_s"].items():
        assert all(raw["duration_s"] == horizon for n, _, raw in rows if n == name)


def test_actual_grids_must_refine_without_fake_order(spec, budget, tmp_path):
    records = []
    for index, level in enumerate(budget["levels"]):
        raw = deepcopy(spec)
        raw.update(duration_s=0.05, step_s=0.025)
        raw["solver"]["max_step_s"] = 0.001 / 4**index
        folder = tmp_path / level
        report = run_primary_steam_experiment(raw, folder)
        records.append(
            dict(
                name="nominal",
                policy=level,
                folder=str(folder),
                solver_statistics=report["solver_statistics"],
            )
        )
    result = compare_levels(records, budget)
    assert result[0]["observed_order"] is None
    assert all(p["measured_step_ratio"] >= 2 for p in result[0]["comparisons"])
    records[-1]["solver_statistics"]["max_reported_step_s"] = records[-2]["solver_statistics"][
        "max_reported_step_s"
    ]
    with pytest.raises(ValueError, match="actual meshes"):
        compare_levels(records, budget)


@pytest.mark.parametrize(
    "defect",
    ["missing", "duplicate", "signature", "policy", "early_stop", "outside", "reuse", "driver"],
)
def test_complete_matrix_required_before_expensive_audit(tmp_path, budget, defect):
    specs = list(matrix(budget))
    specs.sort(key=lambda x: 0 if x[0] == "long_nominal" else 1)
    records = [
        dict(name=name, policy=level, folder=str(tmp_path / f"{name}_{level}"))
        for name, level, _ in specs
    ]
    report = dict(
        signature=signature(),
        passed=True,
        integrations=len(records),
        case_families=17,
        cases=records,
    )
    drivers = tmp_path / "drivers"
    drivers.mkdir()
    for path in DRIVERS:
        shutil.copy2(path, drivers / Path(path).name)
    first = Path(records[0]["folder"])
    first.mkdir()
    raw = deepcopy(specs[0][2])
    (first / "input.json").write_text(json.dumps(raw))
    if defect == "missing":
        records.pop()
        report["integrations"] -= 1
    elif defect == "duplicate":
        records[-1] = deepcopy(records[0])
    elif defect == "signature":
        report["signature"]["source_sha256"] = "0" * 64
    elif defect == "policy":
        raw["solver"]["max_step_s"] *= 2
        (first / "input.json").write_text(json.dumps(raw))
    elif defect == "early_stop":
        (first / "report.json").write_text(
            json.dumps(dict(status="domain_exit", completed_time_s=0.01))
        )
    elif defect == "outside":
        records[0]["folder"] = str(tmp_path.parent / "escape")
    elif defect == "reuse":
        records[0]["reused_from"] = dict(manifest_sha256="0" * 64)
        (first / "manifest.json").write_text("{}")
    elif defect == "driver":
        (drivers / Path(DRIVERS[0]).name).write_text("changed")
    (tmp_path / "budget.json").write_text(json.dumps(budget))
    (tmp_path / "qualification.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit(tmp_path)


@pytest.mark.parametrize("defect", [None, "input", "source", "dependency", "bytes"])
def test_reuse_requires_exact_runtime_environment_and_bytes(packet, tmp_path, defect):
    source = tmp_path / "source"
    shutil.copytree(packet, source)
    raw = json.loads((source / "input.json").read_text())
    target = tmp_path / "target"
    if defect == "input":
        raw["solver"]["max_step_s"] /= 2
    elif defect in ("source", "dependency"):
        manifest = json.loads((source / "manifest.json").read_text())
        if defect == "source":
            manifest["source_sha256"] = "0" * 64
        else:
            manifest["versions"]["scipy"] = "0.0.0"
        (source / "manifest.json").write_text(json.dumps(manifest))
    elif defect == "bytes":
        (source / "trajectory.json").write_text("[]")
    if defect == "input":
        assert reuse_packet(source, target, raw) is None
    elif defect is None:
        assert reuse_packet(source, target, raw)["manifest_sha256"] == sha(source / "manifest.json")
        for path in source.iterdir():
            assert path.read_bytes() == (target / path.name).read_bytes()
    else:
        with pytest.raises(ValueError):
            reuse_packet(source, target, raw)
        assert not target.exists()
