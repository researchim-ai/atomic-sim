"""Independent covariance propagation, genuine correlation and held-out semantics."""

import copy

import numpy as np
import pytest

from atomic_model_uncertainty.coverage import COMPONENTS, evaluate


def specification():
    joint = np.zeros((5, 5))
    joint[0, 0] = 1
    joint[1, 1] = 4
    joint[0, 1] = joint[1, 0] = -1
    case = dict(
        id="heldout",
        family="experiment-b",
        split="validation",
        observed=[1.0],
        predicted=[0.0],
        units=["K"],
        component_covariances={name: [[float(joint[i, i])]] for i, name in enumerate(COMPONENTS)},
        joint_covariance=joint.tolist(),
        provenance="Manufactured covariance test; not experimental data",
    )
    return dict(
        schema_version=1,
        evidence_kind="numerical",
        distribution="gaussian",
        confidence=0.95,
        target_coverage=0.9,
        cases=[case],
    )


def test_cross_correlation_preserved_not_summed_as_independent():
    spec = specification()
    result = evaluate(spec)
    assert result["cases"][0]["total_covariance"] == [[3.0]]
    assert result["heldout_coverage"] == 1
    assert not result["validated"] and not result["experimental_validation_promoted"]


def test_full_matrix_propagation_against_direct_linear_map():
    rng = np.random.default_rng(51)
    a = rng.normal(size=(10, 10))
    joint = a @ a.T
    spec = specification()
    case = spec["cases"][0]
    case.update(
        observed=[1.0, 2.0],
        predicted=[0.9, 1.8],
        units=["K", "Pa"],
        joint_covariance=joint.tolist(),
        component_covariances={
            name: joint[i * 2 : (i + 1) * 2, i * 2 : (i + 1) * 2].tolist()
            for i, name in enumerate(COMPONENTS)
        },
    )
    result = evaluate(spec)
    expected = sum(
        joint[i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2] for i in range(5) for j in range(5)
    )
    np.testing.assert_allclose(result["cases"][0]["total_covariance"], expected)


@pytest.mark.parametrize(
    "bad",
    [
        lambda s: s["cases"][0].pop("joint_covariance"),
        lambda s: s["cases"][0]["joint_covariance"][0].__setitem__(0, -1),
        lambda s: s["cases"][0]["joint_covariance"][0].__setitem__(1, 9),
        lambda s: s["cases"][0]["component_covariances"]["data"][0].__setitem__(0, 2),
        lambda s: s.update(confidence=float("nan")),
        lambda s: s.update(distribution="unknown"),
        lambda s: s["cases"][0].update(units=[]),
    ],
)
def test_missing_or_invalid_uncertainty_never_inferred(bad):
    spec = specification()
    bad(spec)
    with pytest.raises(ValueError):
        evaluate(spec)


def test_physical_case_leakage_refused():
    spec = specification()
    calibration = copy.deepcopy(spec["cases"][0])
    calibration.update(id="calibration", split="calibration")
    spec["cases"].append(calibration)
    with pytest.raises(ValueError, match="leakage"):
        evaluate(spec)


def test_experimental_label_alone_does_not_promote_physics():
    spec = specification()
    spec["evidence_kind"] = "experimental"
    result = evaluate(spec)
    assert not result["validated"] and not result["experimental_validation_promoted"]


def test_zero_variance_and_noncovered_case_retained():
    spec = specification()
    case = spec["cases"][0]
    case["joint_covariance"] = np.zeros((5, 5)).tolist()
    case["component_covariances"] = {key: [[0.0]] for key in COMPONENTS}
    result = evaluate(spec)
    assert result["heldout_coverage"] == 0 and not result["target_met"]


@pytest.mark.parametrize("scale", [1e-200, 1e-16, 1.0, 1e150])
def test_non_psd_covariance_rejected_at_any_unit_scale(scale):
    from atomic_model_uncertainty.coverage import covariance

    with pytest.raises(ValueError):
        covariance(np.array([[1.0, 2.0], [2.0, 1.0]]) * scale, 2)


def test_mixed_units_cannot_hide_negative_covariance_eigenvalue():
    from atomic_model_uncertainty.coverage import covariance

    # Pa and another variable with much smaller units; correlation remains invalid.
    matrix = np.array([[1e20, 2.0], [2.0, 1e-20]])
    with pytest.raises(ValueError):
        covariance(matrix, 2)


def test_unit_conversion_preserves_valid_psd_covariance():
    from atomic_model_uncertainty.coverage import covariance

    original = np.array([[4.0, 1.0], [1.0, 9.0]])
    units = np.array([1e-10, 1e10])
    converted = original * units[:, None] * units[None, :]
    np.testing.assert_array_equal(covariance(converted, 2), converted)


def test_zero_variance_does_not_allow_nonzero_cross_covariance():
    from atomic_model_uncertainty.coverage import covariance

    with pytest.raises(ValueError):
        covariance([[0.0, 1e-100], [1e-100, 1.0]], 2)


def test_tiny_asymmetry_relative_to_variance_is_rejected():
    from atomic_model_uncertainty.coverage import covariance

    with pytest.raises(ValueError):
        covariance([[1e-20, 0.0], [0.5e-20, 1e-20]], 2)


def test_complex_covariance_is_not_silently_projected_to_real():
    from atomic_model_uncertainty.coverage import covariance

    with pytest.raises(ValueError):
        covariance(np.array([[1 + 1j]]), 1)


def test_extreme_representable_confidence_has_finite_intervals():
    spec = specification()
    spec["confidence"] = float(np.nextafter(1.0, 0.0))
    result = evaluate(spec)
    assert np.isfinite(result["cases"][0]["interval_half_width"]).all()


def test_tiny_diagonal_block_mismatch_not_hidden_by_absolute_tolerance():
    spec = specification()
    case = spec["cases"][0]
    case["joint_covariance"] = (np.eye(5) * 1e-20).tolist()
    case["component_covariances"] = {name: [[1e-20]] for name in COMPONENTS}
    case["component_covariances"]["data"] = [[1e-16]]
    with pytest.raises(ValueError, match="diagonal block"):
        evaluate(spec)


def test_family_diagnostics_expose_imbalance_without_changing_acceptance():
    spec = specification()
    large = spec["cases"][0]
    large.update(
        observed=[0.0] * 9,
        predicted=[0.0] * 9,
        units=["K"] * 9,
        joint_covariance=np.zeros((45, 45)).tolist(),
        component_covariances={name: np.zeros((9, 9)).tolist() for name in COMPONENTS},
    )
    small = specification()["cases"][0]
    small.update(id="miss", family="other-experiment", observed=[100.0])
    calibration = copy.deepcopy(small)
    calibration.update(id="fit", family="calibration-family", split="calibration")
    spec["cases"].extend([small, calibration])
    before = copy.deepcopy(spec)
    result = evaluate(spec)
    assert spec == before
    assert result["heldout_coverage"] == 0.9
    assert result["target_met"]
    assert result["target_metric"] == "heldout_component_coverage"
    assert result["heldout_families"] == 2
    assert result["family_balanced_coverage"] == 0.5
    assert not result["family_balanced_target_met"]
    assert len(result["missed_components"]) == 2
    miss = next(r for r in result["missed_components"] if r["case_id"] == "miss")
    assert miss["component_index"] == 0 and miss["units"] == "K"
    assert miss["residual"] == 100.0
    assert miss["interval_half_width"] > 0
    assert not result["validated"]


def test_family_weight_is_invariant_under_duplicate_sampling():
    spec = specification()
    bad = copy.deepcopy(spec["cases"][0])
    bad.update(id="bad", family="second", observed=[100.0])
    spec["cases"].append(bad)
    baseline = evaluate(spec)
    duplicate = copy.deepcopy(spec["cases"][0])
    duplicate["id"] = "repeat"
    spec["cases"].append(duplicate)
    repeated = evaluate(spec)
    assert repeated["heldout_coverage"] != baseline["heldout_coverage"]
    assert repeated["family_balanced_coverage"] == baseline["family_balanced_coverage"] == 0.5
    family = next(r for r in repeated["family_diagnostics"] if r["family"] == "experiment-b")
    assert family["case_ids"] == ["heldout", "repeat"]
    assert family["components"] == 2


@pytest.mark.parametrize("metric,expected", [(None, 0), ("pooled", 0), ("family-balanced", 2)])
def test_cli_records_selected_criterion_and_preserves_failed_report(tmp_path, metric, expected):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from atomic_model_multiphysics.provenance import unseal

    spec = specification()
    # Nine independent case records from one physical family, one from another.
    # Repeated family membership must not increase its family-balanced weight.
    template = spec["cases"][0]
    spec["cases"] = []
    for i in range(9):
        case = copy.deepcopy(template)
        case["id"] = f"pass-{i}"
        spec["cases"].append(case)
    miss = copy.deepcopy(template)
    miss.update(id="miss", family="second-family", observed=[100.0])
    spec["cases"].append(miss)
    source = tmp_path / "input.json"
    output = tmp_path / "report.json"
    source.write_text(json.dumps(spec), encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root)])
    command = [
        sys.executable,
        "-m",
        "atomic_model_uncertainty",
        str(source),
        "--output",
        str(output),
    ]
    if metric:
        command.extend(["--acceptance-metric", metric])
    process = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
    )
    assert process.returncode == expected, process.stderr
    summary = json.loads(process.stdout)
    report = unseal(json.loads(output.read_text(encoding="utf-8")))
    assert summary["acceptance"] == report["acceptance"]
    assert report["acceptance"]["target_met"] == (expected == 0)
    assert report["acceptance"]["metric"] == (
        "family_balanced_coverage" if metric == "family-balanced" else "heldout_coverage"
    )
    assert report["heldout_coverage"] == 0.9
    assert report["family_balanced_coverage"] == 0.5
    assert report["target_met"]  # Legacy pooled result remains explicitly available.
    assert len(report["missed_components"]) == 1
    assert not report["validated"]
    assert json.loads(source.read_text(encoding="utf-8")) == spec
