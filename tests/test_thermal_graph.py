"""Analytic enthalpy mixing, friction heat, conservative ports and typed failures."""

import json
import sys
from copy import deepcopy

import pytest

from core.network import NetworkError
from scripts.solve_thermal_graph import ThermalSolveError, main, solve_thermal_graph


def example():
    return dict(
        format="atomic-thermal-hydraulic-steady",
        schema_version=1,
        hydraulic=dict(
            format="atomic-hydraulic-steady",
            schema_version=1,
            density_kg_m3=1000,
            gravity_m_s2=9.81,
            nodes=[
                dict(id="supply", elevation_m=0, pressure_pa=200000),
                dict(id="outlet", elevation_m=0, pressure_pa=199000),
            ],
            edges=[
                dict(
                    id="pipe",
                    **{"from": "supply", "to": "outlet"},
                    linear_pa_s_kg=1000,
                    quadratic_pa_s2_kg2=0,
                    pressure_gain_pa=0,
                )
            ],
        ),
        thermal=dict(
            medium="test-liquid",
            heat_capacity_j_kg_k=1000,
            enthalpy_reference_temperature_k=300,
            min_temperature_k=250,
            max_temperature_k=500,
            nodes={
                name: dict(
                    heat_power_w=0,
                    ambient_ua_w_k=0,
                    ambient_temperature_k=300,
                    supply_temperature_k=300,
                )
                for name in ("supply", "outlet")
            },
        ),
    )


def cycle():
    spec = example()
    spec["hydraulic"]["nodes"] = [
        dict(id=name, elevation_m=0, pressure_pa=200000) for name in ("a", "b", "c")
    ]
    spec["hydraulic"]["edges"] = [
        dict(
            id="e" + str(i),
            **{"from": left, "to": right},
            linear_pa_s_kg=1000,
            quadratic_pa_s2_kg2=0,
            pressure_gain_pa=3000,
        )
        for i, (left, right) in enumerate([("a", "b"), ("b", "c"), ("c", "a")])
    ]
    spec["thermal"]["nodes"] = {
        name: dict(heat_power_w=0, ambient_ua_w_k=10, ambient_temperature_k=300)
        for name in ("a", "b", "c")
    }
    return spec


def test_friction_pressure_work_becomes_sensible_heat_without_double_counting():
    spec = example()
    before = deepcopy(spec)
    result = solve_thermal_graph(spec)
    assert spec == before
    assert result["temperatures_k"] == pytest.approx(dict(supply=300, outlet=300.001), abs=1e-10)
    assert result["powers_w"]["dissipation_to_fluid"] == pytest.approx(1.0)
    assert result["powers_w"]["external_enthalpy"] == pytest.approx(0.0, abs=1e-8)
    assert result["residuals"]["maximum_global_energy_residual_w"] < 1e-5
    assert result["warmup_steps"] == 0 and not result["commands_changed"]


def test_reversing_orientation_preserves_temperatures_and_inverts_edge_flows():
    spec = example()
    before = solve_thermal_graph(spec)
    edge = spec["hydraulic"]["edges"][0]
    edge["from"], edge["to"] = edge["to"], edge["from"]
    after = solve_thermal_graph(spec)
    assert before["temperatures_k"] == pytest.approx(after["temperatures_k"])
    assert after["hydraulic"]["flows_kg_s"]["pipe"] == pytest.approx(-1.0)


@pytest.mark.parametrize("power", [-1000, 0, 1000])
def test_nodal_heat_or_cooling_transported_to_outlet(power):
    spec = example()
    spec["thermal"]["nodes"]["outlet"]["heat_power_w"] = power
    result = solve_thermal_graph(spec)
    assert result["temperatures_k"]["outlet"] == pytest.approx(300.001 + power / 1000, abs=1e-10)


def test_open_column_gravity_and_total_enthalpy():
    spec = example()
    spec["hydraulic"]["nodes"][1]["elevation_m"] = 0.05
    result = solve_thermal_graph(spec)
    assert result["powers_w"]["gravity"] < 0
    assert abs(result["powers_w"]["external_enthalpy"] + result["powers_w"]["gravity"]) < 1e-5


def test_closed_driven_cycle_rejects_missing_thermal_gauge():
    spec = cycle()
    for node in spec["thermal"]["nodes"].values():
        node["ambient_ua_w_k"] = 0
    with pytest.raises(ThermalSolveError) as error:
        solve_thermal_graph(spec)
    assert error.value.report["criterion"] == "thermal_rank"


def test_closed_driven_cycle_balances_motor_work_and_ambient_removal():
    result = solve_thermal_graph(cycle())
    assert result["temperatures_k"] == pytest.approx({n: 300.9 for n in ("a", "b", "c")}, abs=1e-9)
    assert result["powers_w"]["ambient_removal"] == pytest.approx(27.0)
    assert result["powers_w"]["hydraulic_gain"] == pytest.approx(27.0)


def test_zero_flow_nodes_anchored_only_by_ambient():
    spec = example()
    spec["hydraulic"]["nodes"][1]["pressure_pa"] = 200000
    for i, node in enumerate(spec["thermal"]["nodes"].values()):
        node.update(ambient_ua_w_k=10, heat_power_w=100 * i)
    result = solve_thermal_graph(spec)
    assert result["temperatures_k"] == pytest.approx(dict(supply=300, outlet=310))
    assert result["powers_w"]["dissipation_to_fluid"] == 0
    assert all(port["stream_temperature_k"] is None for port in result["ports"])


def test_zero_flow_unanchored_node_is_not_assigned_a_temperature():
    spec = example()
    spec["hydraulic"]["nodes"][1]["pressure_pa"] = 200000
    with pytest.raises(ThermalSolveError) as error:
        solve_thermal_graph(spec)
    assert error.value.report["criterion"] == "thermal_rank"


def test_ports_are_pairwise_conservative_and_follow_enthalpy_convention():
    result = solve_thermal_graph(example())
    for port in result["ports"]:
        assert port["edge_inward_mass_kg_s"] + port["node_inward_mass_kg_s"] == 0
        assert port["edge_inward_energy_w"] + port["node_inward_energy_w"] == 0
        assert port["energy_convention"] == "specific_enthalpy_transport"
        assert port["medium"] == "test-liquid"
    assert max(abs(v) for v in result["node_energy_residuals_w"].values()) < 1e-5


def test_two_external_supplies_mix_by_enthalpy_including_pressure_loss_heat():
    from scripts.thermal_graph_references import limiting_cases

    name, spec, exact = list(limiting_cases())[-1]
    assert name == "two_supplies"
    result = solve_thermal_graph(spec)
    assert result["temperatures_k"] == pytest.approx(exact["temperatures_k"], abs=1e-9)
    assert result["powers_w"]["dissipation_to_fluid"] == pytest.approx(7.2)


def test_enthalpy_reference_is_an_offset_not_an_extra_heat_source():
    spec = example()
    original = solve_thermal_graph(spec)
    spec["thermal"]["enthalpy_reference_temperature_k"] = 273.15
    result = solve_thermal_graph(spec)
    assert result["temperatures_k"] == pytest.approx(original["temperatures_k"], abs=1e-9)
    assert result["residuals"]["maximum_global_energy_residual_w"] < 1e-5


def test_reverse_flow_can_absorb_prescribed_pressure_gain_power():
    spec = example()
    spec["hydraulic"]["nodes"][1]["pressure_pa"] = 203000
    spec["hydraulic"]["edges"][0]["pressure_gain_pa"] = 1000
    spec["thermal"]["nodes"]["outlet"]["supply_temperature_k"] = 320
    result = solve_thermal_graph(spec)
    assert result["hydraulic"]["flows_kg_s"]["pipe"] == pytest.approx(-2.0)
    assert result["powers_w"]["hydraulic_gain"] == pytest.approx(-2.0)
    assert result["temperatures_k"]["supply"] == pytest.approx(320.002)
    assert result["residuals"]["maximum_global_energy_residual_w"] < 1e-5


def test_unresolved_ambient_anchor_is_rejected_instead_of_fixing_mean_temperature():
    spec = cycle()
    for node in spec["thermal"]["nodes"].values():
        node["ambient_ua_w_k"] = 1e-18
    with pytest.raises(ThermalSolveError) as error:
        solve_thermal_graph(spec)
    assert error.value.report["criterion"] == "thermal_rank"


def test_missing_supply_temperature_never_defaults_to_initial_guess():
    spec = example()
    spec["thermal"]["nodes"]["supply"].pop("supply_temperature_k")
    with pytest.raises(ThermalSolveError) as error:
        solve_thermal_graph(spec)
    assert error.value.report["criterion"] == "missing_supply_temperature"


def test_outflow_does_not_use_reservoir_supply_temperature():
    spec = example()
    before = solve_thermal_graph(spec)
    spec["thermal"]["nodes"]["outlet"]["supply_temperature_k"] = 450
    assert solve_thermal_graph(spec)["temperatures_k"] == before["temperatures_k"]


@pytest.mark.parametrize(
    "change",
    [
        "format",
        "schema",
        "extra",
        "missing_node",
        "extra_node",
        "negative_ua",
        "zero_cp",
        "empty_medium",
        "domain",
        "nan",
    ],
)
def test_invalid_thermal_contract(change):
    spec = example()
    if change == "format":
        spec["format"] = "general-eos"
    elif change == "schema":
        spec["schema_version"] = True
    elif change == "extra":
        spec["thermal"]["nodes"]["outlet"]["initial_guess_k"] = 300
    elif change == "missing_node":
        spec["thermal"]["nodes"].pop("outlet")
    elif change == "extra_node":
        spec["thermal"]["nodes"]["unexpected"] = {}
    elif change == "negative_ua":
        spec["thermal"]["nodes"]["outlet"]["ambient_ua_w_k"] = -1
    elif change == "zero_cp":
        spec["thermal"]["heat_capacity_j_kg_k"] = 0
    elif change == "empty_medium":
        spec["thermal"]["medium"] = ""
    elif change == "domain":
        spec["thermal"]["min_temperature_k"] = 500
    elif change == "nan":
        spec["thermal"]["nodes"]["outlet"]["heat_power_w"] = float("nan")
    with pytest.raises(NetworkError):
        solve_thermal_graph(spec)


def test_overheated_node_is_not_clipped():
    spec = example()
    spec["thermal"]["nodes"]["outlet"]["heat_power_w"] = 1e9
    with pytest.raises(ThermalSolveError) as error:
        solve_thermal_graph(spec)
    assert error.value.report["criterion"] == "temperature_domain"


def test_hot_stream_cannot_be_hidden_by_cold_node_ambient():
    spec = example()
    spec["thermal"]["max_temperature_k"] = 300.0005
    spec["thermal"]["nodes"]["outlet"].update(ambient_ua_w_k=1e6, ambient_temperature_k=300)
    with pytest.raises(ThermalSolveError) as error:
        solve_thermal_graph(spec)
    assert error.value.report["criterion"] == "edge_temperature_domain"


def test_unresolved_external_exchange_is_not_used_as_thermal_anchor():
    spec = example()
    spec["hydraulic"]["nodes"][1]["pressure_pa"] = 200000 - 1e-6
    with pytest.raises(ThermalSolveError) as error:
        solve_thermal_graph(spec)
    assert error.value.report["criterion"] == "unresolved_external_flow"


def test_cli_success_manifest_and_refusal_to_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(example()))
    out = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["thermal", str(source), "--output", str(out)])
    assert main() == 0
    assert set(json.loads((out / "manifest.json").read_text())["files"]) == {
        "request.json",
        "solution.json",
    }
    with pytest.raises(FileExistsError):
        main()


def test_cli_rejection_has_no_solution(tmp_path, monkeypatch):
    spec = example()
    spec["thermal"]["nodes"]["supply"].pop("supply_temperature_k")
    source = tmp_path / "input.json"
    source.write_text(json.dumps(spec))
    out = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["thermal", str(source), "--output", str(out)])
    assert main() == 2 and not (out / "solution.json").exists()
    assert (out / "diagnostics.json").exists()
