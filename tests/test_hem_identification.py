"""Feasible-set certificates, sensitivities, primary-series isolation and refusal gates."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from atomic_model_hem_assessment.audit import dataset
from atomic_model_hem_identification import (
    cohorts,
    precision,
    prediction,
    qualification,
    reference,
    region,
    workflow,
)
from atomic_model_hem_identification.__main__ import main
from atomic_model_hem_qualified import eos
from atomic_model_hem_throughflow import closures
from atomic_model_hem_throughflow.benchmark import evaluate as frozen_benchmark
from atomic_model_hem_throughflow.model import signature as frozen_signature
from atomic_model_workspace.common import digest


@pytest.mark.parametrize("name,observations,status,expected", qualification.cases())
def test_closed_form_regions_and_independent_oracle(name, observations, status, expected):
    solved = region.solve(observations)
    A, b = region.constraints(observations)
    oracle = reference.enumerate_region(A, b)
    workflow.agree(solved, oracle)
    assert solved["status"] == status
    if expected is not None:
        for bound, (lo, hi) in zip(solved["bounds"], expected):
            assert bound["lower"] == pytest.approx(lo, abs=1e-8)
            assert bound["upper"] == pytest.approx(hi) if hi is not None else bound["upper"] is None
    assert not solved["physical_acceptance"]


@pytest.mark.parametrize("p", [15e6, 15.5e6, 16e6])
@pytest.mark.parametrize("x", [0.01, 0.05, 0.1])
def test_shadow_unit_gains_match_frozen_production_and_derivative(p, x):
    G, D, q, L = 520.0, 0.01581, 2e5, 3.0
    out = prediction.predict(p, x, G, D, q, L)
    actual = closures.wall_temperature(p, x, G, D, q, L, scope={k: True for k in closures.FLAGS})
    assert out["wall_temperature_k"] == pytest.approx(actual["wall_temperature_k"], abs=2e-8)
    assert out["root_bracket_width_k"] <= 1e-9
    assert abs(out["heat_flux_residual_w_m2"]) <= 0.001
    for j in (0, 1):
        a = [1.0, 1.0]
        b = [1.0, 1.0]
        a[j] += 0.001
        b[j] -= 0.001
        fd = (
            prediction.predict(p, x, G, D, q, L, a)["wall_temperature_k"]
            - prediction.predict(p, x, G, D, q, L, b)["wall_temperature_k"]
        ) / 0.002
        assert out["sensitivity_k_per_gain"][j] == pytest.approx(fd, rel=2e-4, abs=2e-6)
    assert out["diagnostic_only"] and not out["physical_acceptance"]


@pytest.mark.parametrize(
    "gains", [(0.0, 0.0), (-1.0, 1.0), (True, 1.0), (float("nan"), 1.0), (1.0, float("inf"))]
)
def test_invalid_diagnostic_gains_refused(gains):
    with pytest.raises(ValueError):
        prediction.predict(15.5e6, 0.05, 520.0, 0.01581, 2e5, 3.0, gains)


def test_underresolved_large_gain_is_not_reported_as_a_trusted_prediction():
    with pytest.raises(ValueError):
        prediction.predict(15.5e6, 0.05, 520.0, 0.01581, 2e5, 3.0, (1e12, 0.0))


def test_terms_require_declared_pressure_and_wall_domain():
    with pytest.raises(ValueError):
        prediction.terms(14e6, 0.05, 520.0, 0.01581, 620.0)
    with pytest.raises(ValueError):
        prediction.terms(15.5e6, 0.05, 520.0, 0.01581, 500.0)


def rectangle():
    return qualification.cases()[0][1]


@pytest.mark.parametrize("kind", ["empty", "unknown", "bool", "nan", "reversed", "duplicate"])
def test_bad_feasible_set_inputs_refused(kind):
    rows = rectangle()
    if kind == "empty":
        rows = []
    elif kind == "unknown":
        rows[0]["validated"] = True
    elif kind == "bool":
        rows[0]["lower_basis"][0] = True
    elif kind == "nan":
        rows[0]["observed_q_bounds"][0] = float("nan")
    elif kind == "reversed":
        rows[0]["observed_q_bounds"] = [3.0, 1.0]
    else:
        rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError):
        region.solve(rows)


def test_optimizer_unbounded_status_alone_is_not_a_certificate(monkeypatch):
    original = region.linprog

    def false_upper(c, **kwargs):
        if np.min(c) < 0:
            return SimpleNamespace(status=3)
        return original(c, **kwargs)

    monkeypatch.setattr(region, "linprog", false_upper)
    with pytest.raises(ValueError, match="recession"):
        region.solve(rectangle())


def test_corrupt_dual_certificate_is_rejected(monkeypatch):
    original = region.linprog

    def corrupt(c, **kwargs):
        out = original(c, **kwargs)
        if np.max(c) > 0:
            out.ineqlin.marginals = np.ones_like(out.ineqlin.marginals)
        return out

    monkeypatch.setattr(region, "linprog", corrupt)
    with pytest.raises(ValueError, match="certificate"):
        region.solve(rectangle())


def precision_point():
    p = 15.5e6
    t = eos.px(p, 0.0).temperature_k + 2
    return dict(
        pressure_pa=p,
        pressure_pa_lower=p,
        pressure_pa_upper=p,
        wall_temperature_k=t,
        wall_temperature_k_lower=t - 1,
        wall_temperature_k_upper=t + 1,
        heat_flux_w_m2=100.0,
        heat_flux_w_m2_lower=99.0,
        heat_flux_w_m2_upper=101.0,
    )


def test_precision_requirement_matches_independent_closed_form():
    row = precision_point()
    out = precision.requirement(row)
    ratio = 1.2 * 99 / 101
    exact = 2 * (ratio - 1) / (ratio + 1)
    assert out["maximum_joint_scale"] <= exact <= out["rejected_scale_upper"]
    assert out["rejected_scale_upper"] - out["maximum_joint_scale"] <= 1e-9
    assert not out["physical_acceptance"]


def test_heat_flux_noise_cannot_be_hidden_by_perfect_p_temperature():
    row = precision_point()
    row["heat_flux_w_m2_lower"] = 80.0
    row["heat_flux_w_m2_upper"] = 120.0
    out = precision.requirement(row)
    assert out["status"] == "heat_flux_uncertainty_floor_exceeds_target"
    assert out["maximum_joint_scale"] is None


@pytest.mark.parametrize("value", [float("nan"), True, "100"])
def test_nonphysical_measurement_bounds_fail_closed(value):
    row = precision_point()
    row["heat_flux_w_m2_lower"] = value
    with pytest.raises(ValueError):
        precision.requirement(row)


def test_precision_current_box_may_have_unbounded_upper():
    row = precision_point()
    row["wall_temperature_k_lower"] -= 3
    assert precision.interval(row, 1.0)["htc_upper_w_m2_k"] is None
    assert precision.interval(row, 0.0)["htc_upper_w_m2_k"] is not None


@pytest.fixture(scope="module")
def real_rows():
    manifest, rows = dataset()
    ids = {r["row_id"] for r in frozen_benchmark()["rows"] if r["candidate"]}
    return manifest, [r for r in rows if r["row_id"] in ids]


def test_source_curated_series_are_disjoint_but_not_claimed_prospective(real_rows):
    manifest, rows = real_rows
    groups, info = cohorts.split(rows, manifest)
    assert len(groups["development"]) == 57 and len(groups["evaluation"]) == 93
    assert len({r["run"] for r in groups["development"]}) == 9
    assert len({r["run"] for r in groups["evaluation"]}) == 12
    assert info["runs_disjoint"] and info["operating_series_disjoint"]
    assert not info["prospective_holdout"] and not info["independent_facility"]


@pytest.mark.parametrize(
    "kind", ["duplicate", "unknown_run", "wrong_temperature", "wrong_manifest", "empty"]
)
def test_group_admission_never_silently_drops_physical_rows(real_rows, kind):
    manifest, original = real_rows
    rows = deepcopy(original)
    manifest = deepcopy(manifest)
    if kind == "duplicate":
        rows.append(rows[0])
    elif kind == "unknown_run":
        rows[0]["run"] = 999
    elif kind == "wrong_temperature":
        rows[0]["inlet_temperature_k"] = 500.0
    elif kind == "wrong_manifest":
        manifest["readme_pdf_sha256"] = "0" * 64
    else:
        rows = []
    with pytest.raises(ValueError):
        cohorts.split(rows, manifest)


def test_real_data_proves_practical_not_structural_nonidentifiability():
    before = frozen_signature()
    r = workflow.evaluate()
    assert r["calibration_status"] == "rejected" and r["parameter_update"] is None
    for group in ("development", "evaluation"):
        v = r[group]
        lp = v["parameter_region"]
        assert lp["status"] == "unbounded"
        assert all(b["upper"] is None for b in lp["bounds"])
        rays = [c["recession_ray"] for c in lp["certificates"] if "recession_ray" in c]
        assert rays == [[1.0, 0.0], [0.0, 1.0]]
        assert v["sensitivity"]["rank"] == 2
        assert not v["conditional_identifiable_at_declared_resolution"]
        assert lp["baseline_feasible"]
    assert r["frozen_production_benchmark"]["screening_status"] == "failed"
    assert before == frozen_signature()


def test_cli_preserves_denied_calibration_and_rejects_rehashed_status(tmp_path):
    import json

    path = tmp_path / "assessment.json"
    assert (
        main(["analyse", "--output", str(path), "--require-identifiable", "--require-calibratable"])
        == 3
    )
    assert main(["audit", str(path)]) == 0
    before = path.read_bytes()
    with pytest.raises(SystemExit):
        main(["analyse", "--output", str(path)])
    assert path.read_bytes() == before
    saved = json.loads(path.read_text())
    saved["payload"]["calibration_status"] = "admitted"
    saved["sha256"] = digest(saved["payload"])
    path.write_text(json.dumps(saved))
    with pytest.raises(SystemExit):
        main(["audit", str(path)])


def test_direct_cone_certificate_does_not_call_an_optimizer(real_rows, monkeypatch):
    _, rows = real_rows
    observations = [workflow.lean(prediction.envelope(r)) for r in rows]

    def fail(*args, **kwargs):
        raise AssertionError("No optimizer is needed for this certificate")

    monkeypatch.setattr(region, "linprog", fail)
    cert = reference.monotone_cone_certificate(observations)
    assert cert["minimum_lower_constraint_margin_w_m2"] > 0
    for r in observations:
        point = np.array(cert["feasible_witness"])
        assert np.dot(r["upper_basis"], point) >= r["observed_q_bounds"][0]
        assert np.dot(r["lower_basis"], point) <= r["observed_q_bounds"][1]
