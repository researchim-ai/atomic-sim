"""Qualification must actually refine the mesh and complete its declared long runs."""

import json
import shutil
from copy import deepcopy

import pytest

from scripts.audit_steam_refinement import audit
from scripts.qualify_steam_cycle import (
    BUDGET,
    DRIVERS,
    compare_levels,
    matrix,
    reuse_packet,
    signature,
    trace,
)
from sdk.steam_cycle_experiment import run_steam_cycle_experiment


@pytest.fixture
def budget():
    return json.loads(BUDGET.read_text())


def test_three_levels_use_fixed_tolerances_and_cover_all_workloads(budget):
    specs = list(matrix(budget))
    assert len(specs) == 44
    assert len({name for name, _, _ in specs}) == 14
    for name in {name for name, _, _ in specs}:
        levels = {level: spec for n, level, spec in specs if n == name}
        assert all(level in levels for level in ("coarse", "medium", "fine"))
        assert len({levels[level]["solver"]["rtol"] for level in budget["levels"]}) == 1
        assert len({levels[level]["solver"]["atol_scale"] for level in budget["levels"]}) == 1
        assert (
            levels["coarse"]["solver"]["max_step_s"] == 4 * levels["medium"]["solver"]["max_step_s"]
        )
        assert (
            levels["medium"]["solver"]["max_step_s"] == 4 * levels["fine"]["solver"]["max_step_s"]
        )
    for name, horizon in budget["long_horizons_s"].items():
        assert all(s["duration_s"] == horizon for n, _, s in specs if n == name)


def test_measured_mesh_refinement_and_zero_noise_order_claim(tmp_path, budget):
    records = []
    for name, level, spec in matrix(budget):
        if name != "nominal" or level == "bdf":
            continue
        spec = deepcopy(spec)
        spec.update(duration_s=0.2, step_s=0.1, timeline=[])
        folder = tmp_path / level
        run_steam_cycle_experiment(spec, folder)
        records.append(
            dict(name="nominal", policy=level, folder=str(folder), solver_statistics=trace(folder))
        )
    result = compare_levels(records, budget)
    assert result[0]["observed_order"] is None
    assert all(r["measured_step_ratio"] >= 2 for r in result[0]["comparisons"])
    fake = deepcopy(records)
    fake[-1]["solver_statistics"]["max_reported_step_s"] = fake[-2]["solver_statistics"][
        "max_reported_step_s"
    ]
    with pytest.raises(ValueError, match="did not refine"):
        compare_levels(fake, budget)


@pytest.mark.parametrize(
    "defect", ["missing_fine", "duplicate", "stale", "policy", "early_long_stop"]
)
def test_qualification_integrity_before_expensive_replay(tmp_path, budget, defect):
    specs = list(matrix(budget))
    if defect == "early_long_stop":
        specs.sort(key=lambda r: 0 if r[0] == "long_nominal" else 1)
    cases = [
        dict(name=name, policy=level, folder=str(tmp_path / f"{name}_{level}"))
        for name, level, _ in specs
    ]
    report = dict(
        signature=signature(), passed=True, integrations=len(cases), case_families=14, cases=cases
    )
    drivers = tmp_path / "drivers"
    drivers.mkdir()
    for path in DRIVERS:
        shutil.copyfile(path, drivers / str(path).split("/")[-1])
    if defect == "missing_fine":
        cases[:] = [c for c in cases if c["policy"] != "fine"]
        report["integrations"] = len(cases)
    elif defect == "duplicate":
        cases[-1] = deepcopy(cases[0])
    elif defect == "stale":
        report["signature"]["source_sha256"] = "0" * 64
    else:
        folder = tmp_path / f"{specs[0][0]}_{specs[0][1]}"
        folder.mkdir()
        spec = deepcopy(specs[0][2])
        if defect == "policy":
            spec["solver"]["max_step_s"] *= 2
        (folder / "input.json").write_text(json.dumps(spec))
        if defect == "early_long_stop":
            (folder / "report.json").write_text(
                json.dumps(dict(status="domain_exit", completed_time_s=2))
            )
    (tmp_path / "budget.json").write_text(json.dumps(budget))
    (tmp_path / "qualification.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        audit(tmp_path)


@pytest.mark.parametrize("defect", [None, "stale", "dependencies", "corrupt", "different_input"])
def test_raw_packet_reuse_checks_runtime_bytes_and_exact_input(tmp_path, budget, defect):
    spec = deepcopy(next(matrix(budget))[2])
    spec.update(duration_s=0.02, step_s=0.01, timeline=[])
    source, target = tmp_path / "source", tmp_path / "target"
    run_steam_cycle_experiment(spec, source)
    spec = json.loads((source / "input.json").read_text())
    if defect in ("stale", "dependencies"):
        manifest = json.loads((source / "manifest.json").read_text())
        if defect == "stale":
            manifest["source_sha256"] = "0" * 64
        else:
            manifest["versions"]["scipy"] = "0.0.0"
        (source / "manifest.json").write_text(json.dumps(manifest))
    elif defect == "corrupt":
        (source / "trajectory.json").write_text("[]")
    elif defect == "different_input":
        spec["solver"]["max_step_s"] /= 2
    if defect in ("stale", "dependencies", "corrupt"):
        with pytest.raises(ValueError, match="Cannot reuse"):
            reuse_packet(source, target, spec)
        assert not target.exists()
    elif defect == "different_input":
        assert reuse_packet(source, target, spec) is None
        assert not target.exists()
    else:
        result = reuse_packet(source, target, spec)
        assert result["source_folder"] == str(source)
        for p in source.iterdir():
            assert p.read_bytes() == (target / p.name).read_bytes()
        assert trace(source) == trace(target)
