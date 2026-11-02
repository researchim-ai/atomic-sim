"""Branched steady hydraulics: analytic solutions, power, orientation and failures."""

import json
import sys
from copy import deepcopy

import numpy as np
import pytest

from core.network import NetworkError
from scripts.solve_hydraulic_graph import HydraulicSolveError, main, solve_hydraulic_graph


def graph():
    return dict(
        format="atomic-hydraulic-steady",
        schema_version=1,
        density_kg_m3=1000,
        gravity_m_s2=9.81,
        nodes=[
            dict(id="supply", elevation_m=0, pressure_pa=200000),
            dict(id="junction", elevation_m=0, injection_kg_s=0),
            dict(id="return", elevation_m=0, pressure_pa=100000),
        ],
        edges=[
            dict(
                id="first",
                **{"from": "supply", "to": "junction"},
                linear_pa_s_kg=1000,
                quadratic_pa_s2_kg2=100,
                pressure_gain_pa=0,
            ),
            dict(
                id="second",
                **{"from": "junction", "to": "return"},
                linear_pa_s_kg=1000,
                quadratic_pa_s2_kg2=100,
                pressure_gain_pa=0,
            ),
        ],
    )


def test_two_series_resistors_analytic_and_purity():
    spec = graph()
    before = deepcopy(spec)
    result = solve_hydraulic_graph(spec)
    assert spec == before
    assert result["pressures_pa"]["junction"] == pytest.approx(150000, abs=1e-7)
    expected = (-2000 + np.sqrt(2000**2 + 4 * 200 * 100000)) / 400
    assert list(result["flows_kg_s"].values()) == pytest.approx([expected, expected], abs=1e-10)
    assert result["powers_w"]["dissipation"] == pytest.approx(expected * 100, abs=1e-7)
    assert result["warmup_steps"] == 0 and not result["commands_changed"]


@pytest.mark.parametrize("injection", [-15.0, 0.0, 15.0])
def test_branch_junction_with_withdrawal_or_supply(injection):
    spec = graph()
    spec["nodes"][1]["injection_kg_s"] = injection
    result = solve_hydraulic_graph(spec)
    q = result["flows_kg_s"]
    assert q["second"] - q["first"] == pytest.approx(injection, abs=1e-9)
    assert abs(sum(result["external_injections_kg_s"].values())) < 1e-9
    assert result["residuals"]["maximum_power_residual_w"] < 1e-5


@pytest.mark.parametrize("guess", [1.0, 1e5, 1e10])
def test_arbitrary_initial_guesses_do_not_change_solution(guess):
    spec = graph()
    spec["nodes"][1]["initial_pressure_pa"] = guess
    result = solve_hydraulic_graph(spec)
    assert result["pressures_pa"]["junction"] == pytest.approx(150000, abs=1e-5)


def test_parallel_branches_linear_exact():
    spec = graph()
    spec["edges"].append(dict(spec["edges"][0], id="parallel"))
    for edge in spec["edges"]:
        edge["quadratic_pa_s2_kg2"] = 0
    result = solve_hydraulic_graph(spec)
    assert result["pressures_pa"]["junction"] == pytest.approx(500000 / 3, abs=1e-7)
    assert result["flows_kg_s"]["first"] == pytest.approx(100 / 3)
    assert result["flows_kg_s"]["second"] == pytest.approx(200 / 3)


def test_reversing_all_edge_orientations_preserves_node_solution():
    spec = graph()
    spec["edges"][0]["pressure_gain_pa"] = 4000
    original = solve_hydraulic_graph(spec)
    for edge in spec["edges"]:
        edge["from"], edge["to"] = edge["to"], edge["from"]
        edge["pressure_gain_pa"] *= -1
    result = solve_hydraulic_graph(spec)
    assert result["pressures_pa"] == pytest.approx(original["pressures_pa"])
    assert result["flows_kg_s"] == pytest.approx({k: -v for k, v in original["flows_kg_s"].items()})


def test_hydrostatic_column_zero_flow():
    spec = graph()
    spec["nodes"][1]["elevation_m"] = 5
    spec["nodes"][2]["elevation_m"] = 10
    spec["nodes"][2]["pressure_pa"] = 200000 - 98100
    result = solve_hydraulic_graph(spec)
    assert result["pressures_pa"]["junction"] == pytest.approx(200000 - 49050, abs=1e-7)
    assert list(result["flows_kg_s"].values()) == pytest.approx([0.0, 0.0], abs=1e-10)


def test_prescribed_head_drives_cycle_without_boundary_flow():
    spec = graph()
    spec["nodes"][2].pop("pressure_pa")
    spec["nodes"][2]["injection_kg_s"] = 0
    spec["edges"].append(dict(spec["edges"][0], id="closing", **{"from": "return", "to": "supply"}))
    spec["edges"][0]["pressure_gain_pa"] = 60000
    result = solve_hydraulic_graph(spec)
    assert result["external_injections_kg_s"]["supply"] == pytest.approx(0, abs=1e-9)
    assert result["flows_kg_s"]["first"] > 0
    assert result["powers_w"]["prescribed_gain"] == pytest.approx(
        result["powers_w"]["dissipation"], abs=1e-5
    )


def test_all_nodes_fixed_requires_no_nonlinear_iteration():
    spec = graph()
    spec["nodes"][1].pop("injection_kg_s")
    spec["nodes"][1]["pressure_pa"] = 150000
    result = solve_hydraulic_graph(spec)
    assert result["iterations"] == 0
    assert result["external_injections_kg_s"]["junction"] == pytest.approx(0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("linear_pa_s_kg", 0),
        ("linear_pa_s_kg", -1),
        ("quadratic_pa_s2_kg2", -1),
        ("pressure_gain_pa", float("nan")),
        ("linear_pa_s_kg", True),
    ],
)
def test_unsupported_or_nonfinite_edge_laws_fail(field, value):
    spec = graph()
    spec["edges"][0][field] = value
    with pytest.raises(NetworkError):
        solve_hydraulic_graph(spec)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_node",
        "duplicate_edge",
        "self_loop",
        "unknown_endpoint",
        "missing_condition",
        "both_conditions",
        "unknown_field",
        "bool_schema",
        "negative_density",
        "negative_gravity",
        "unknown_format",
        "empty_edges",
        "pressure_guess_on_boundary",
        "empty_domain",
    ],
)
def test_invalid_graph_contracts(change):
    spec = graph()
    if change == "duplicate_node":
        spec["nodes"][1]["id"] = "supply"
    elif change == "duplicate_edge":
        spec["edges"][1]["id"] = "first"
    elif change == "self_loop":
        spec["edges"][0]["to"] = "supply"
    elif change == "unknown_endpoint":
        spec["edges"][0]["to"] = "missing"
    elif change == "missing_condition":
        spec["nodes"][1].pop("injection_kg_s")
    elif change == "both_conditions":
        spec["nodes"][1]["pressure_pa"] = 150000
    elif change == "unknown_field":
        spec["edges"][0]["pump_speed"] = 5
    elif change == "bool_schema":
        spec["schema_version"] = True
    elif change == "negative_density":
        spec["density_kg_m3"] = -1
    elif change == "negative_gravity":
        spec["gravity_m_s2"] = -1
    elif change == "unknown_format":
        spec["format"] = "general-dae"
    elif change == "empty_edges":
        spec["edges"] = []
    elif change == "pressure_guess_on_boundary":
        spec["nodes"][0]["initial_pressure_pa"] = 1e5
    elif change == "empty_domain":
        spec["nodes"][1].update(min_pressure_pa=2e5, max_pressure_pa=1e5)
    with pytest.raises(NetworkError):
        solve_hydraulic_graph(spec)


def test_disconnected_pressure_islands_rejected():
    spec = graph()
    spec["nodes"].append(dict(id="isolated", elevation_m=0, pressure_pa=100000))
    with pytest.raises(HydraulicSolveError, match="connected"):
        solve_hydraulic_graph(spec)


def test_no_hidden_pressure_gauge():
    spec = graph()
    for node in spec["nodes"]:
        node.pop("pressure_pa", None)
        node["injection_kg_s"] = 0
    with pytest.raises(HydraulicSolveError) as error:
        solve_hydraulic_graph(spec)
    assert error.value.report["criterion"] == "pressure_gauge"


def test_numerically_unresolved_pressure_anchor_is_rejected():
    spec = graph()
    spec["nodes"][2].pop("pressure_pa")
    spec["nodes"][2]["injection_kg_s"] = 0
    spec["edges"][0]["linear_pa_s_kg"] = 1e20
    spec["edges"][0]["quadratic_pa_s2_kg2"] = 0
    with pytest.raises(HydraulicSolveError) as error:
        solve_hydraulic_graph(spec)
    assert error.value.report["criterion"] == "hydraulic_rank"


@pytest.mark.parametrize("endpoint", [[], {}, 1, None])
def test_malformed_endpoint_has_typed_validation_error(endpoint):
    spec = graph()
    spec["edges"][0]["from"] = endpoint
    with pytest.raises(NetworkError):
        solve_hydraulic_graph(spec)


def test_gravity_power_is_accounted_for_with_open_boundary_flow():
    spec = graph()
    spec["nodes"][2]["elevation_m"] = 2
    result = solve_hydraulic_graph(spec)
    q = result["flows_kg_s"]["second"]
    assert result["powers_w"]["gravity"] == pytest.approx(-9.81 * 2 * q)
    assert result["residuals"]["maximum_power_residual_w"] < 1e-5


@pytest.mark.parametrize("bound", ["min_pressure_pa", "max_pressure_pa"])
def test_solved_pressure_is_not_clipped_to_domain(bound):
    spec = graph()
    spec["nodes"][1][bound] = 160000 if bound.startswith("min") else 140000
    with pytest.raises(HydraulicSolveError) as error:
        solve_hydraulic_graph(spec)
    assert error.value.report["criterion"] == "pressure_domain"


def test_cli_manifest_success_and_refuse_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "request.json"
    source.write_text(json.dumps(graph()))
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["solve", str(source), "--output", str(output)])
    assert main() == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["files"]) == {"request.json", "solution.json"}
    with pytest.raises(FileExistsError):
        main()


def test_cli_failure_has_no_solution(tmp_path, monkeypatch):
    spec = graph()
    spec["nodes"][1]["max_pressure_pa"] = 140000
    source = tmp_path / "request.json"
    source.write_text(json.dumps(spec))
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["solve", str(source), "--output", str(output)])
    assert main() == 2
    assert not (output / "solution.json").exists()
    assert json.loads((output / "diagnostics.json").read_text())["criterion"] == "pressure_domain"
