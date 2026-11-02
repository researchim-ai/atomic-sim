"""Fixed commands, explicit inventory constraints, diagnostics and no warmup."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from core.network import NetworkError
from physics.power.primary_steam import PrimarySteamCycle
from scripts.design_primary_steam import design_steady_primary_steam
from scripts.primary_steam_references import power_balance, system
from scripts.solve_fixed_primary_steam import (
    SteadySolveError,
    inventory_anchors,
    main,
    solve_fixed_primary_steam,
)
from scripts.verify_turbine import sha
from sdk.primary_steam_experiment import prepare


@pytest.fixture
def spec():
    value = json.loads(Path("scenarios/ai05_primary_steam.json").read_text())
    value["timeline"] = []
    return value


def operating_point(spec, flow=100, opening=1):
    return design_steady_primary_steam(spec, primary_mass_flow_kg_s=flow, valve_opening=opening)[0]


def bad_guesses(value):
    value = deepcopy(value)
    init = value["primary"]["loop"]["initialization"]
    init["mass_flow_kg_s"] = -33
    for key in ("fluid_temperature_k", "wall_temperature_k"):
        init[key] = {name: 1000 for name in init[key]}
    value["primary"]["initial_omega_rad_s"] = 19
    for name, p, x in [("steam_generator", 7e6, 0.2), ("condenser", 12000, 0.005)]:
        value["steam_model"][name].update(initial_pressure_pa=p, initial_quality=x)
    value["steam_model"]["hot_wall"]["initial_temperature_k"] = 600
    value["steam_model"]["cold_wall"]["initial_temperature_k"] = 330
    value["steam_model"]["turbine"]["initial_omega_rad_s"] = 350
    return value


@pytest.mark.parametrize("flow", [100, -100, 10, -10])
@pytest.mark.parametrize("opening", [1, 0.6])
def test_fixed_commands_recover_point_from_different_initial_guesses(
    spec, flow, opening, monkeypatch
):
    point = operating_point(spec, flow, opening)
    anchors = inventory_anchors(point)
    guesses = bad_guesses(point)
    before = deepcopy(guesses)

    def forbidden(*args, **kwargs):
        raise AssertionError("Steady initialization must not advance time or redesign commands")

    monkeypatch.setattr(PrimarySteamCycle, "advance_until_limit", forbidden)
    monkeypatch.setattr("scripts.design_primary_steam.design_steady_primary_steam", forbidden)
    solved, report = solve_fixed_primary_steam(guesses, **anchors)
    assert guesses == before
    assert solved["controls"] == point["controls"] == report["fixed_commands"]
    assert report["accepted"] and report["warmup_steps"] == 0 and not report["commands_changed"]
    assert solved["primary"]["loop"]["initialization"]["mass_flow_kg_s"] == pytest.approx(
        flow, abs=1e-10
    )
    for name in ("steam_generator", "condenser"):
        assert solved["steam_model"][name]["initial_pressure_pa"] == pytest.approx(
            point["steam_model"][name]["initial_pressure_pa"], abs=0.001
        )
    assert report["dimensional_residuals"]["maximum_stock_power_residual_w"] < 1e-5
    y, rhs, _, _, data, k = system(solved)
    assert abs(power_balance(y, rhs(0, y, solved["controls"]), data, k, solved)) < 1e-5


def test_existing_ambient_losses_remain_in_fixed_boundary_balance(spec):
    for node in spec["primary"]["loop"]["model"]["topology"]["nodes"]:
        node["parameters"].update(ambient_ua_w_k=10, wall_power_w=100)
    point = operating_point(spec)
    solved, report = solve_fixed_primary_steam(bad_guesses(point), **inventory_anchors(point))
    assert solved["controls"] == point["controls"] and report["accepted"]
    assert report["thermal_rank"] == report["thermal_unknowns"] == 6


def test_split_primary_and_steam_heating_is_solved_without_rewriting_either(spec):
    point = operating_point(spec)
    point["controls"]["primary_wall_heater_power_w"] -= 20000
    point["controls"]["steam_cycle"]["heater_power_w"] = 20000
    solved, report = solve_fixed_primary_steam(point, **inventory_anchors(point))
    assert solved["controls"] == point["controls"]
    assert report["accepted"]
    assert (
        solved["primary"]["loop"]["initialization"]["fluid_temperature_k"]["s1"]
        < spec["primary"]["loop"]["initialization"]["fluid_temperature_k"]["s1"]
    )


def test_mass_and_quality_anchors_remove_neutral_inventory_partition(spec):
    point = operating_point(spec)
    anchors = inventory_anchors(point)
    anchors["steam_generator_quality"] = 0.12
    solved, report = solve_fixed_primary_steam(point, **anchors)
    assert solved["controls"] == point["controls"]
    assert solved["steam_model"]["steam_generator"]["initial_quality"] == 0.12
    assert (
        solved["steam_model"]["condenser"]["initial_quality"]
        != point["steam_model"]["condenser"]["initial_quality"]
    )
    assert inventory_anchors(solved)["total_steam_mass_kg"] == pytest.approx(
        anchors["total_steam_mass_kg"], abs=1e-8
    )
    assert not report["global_uniqueness_proven"]


def test_incompatible_heater_is_rejected_instead_of_optimized(spec):
    point = operating_point(spec)
    anchors = inventory_anchors(point)
    point["controls"]["primary_wall_heater_power_w"] += 1000
    before = deepcopy(point)
    with pytest.raises(SteadySolveError) as caught:
        solve_fixed_primary_steam(point, **anchors)
    assert caught.value.report["criterion"] == "hot_energy_compatibility"
    assert abs(caught.value.report["details"]["hot_inventory_residual_w"]) > 999
    assert point == before and not caught.value.report["commands_changed"]


@pytest.mark.parametrize(
    "failure", ["feed", "valve", "cooling", "rotor_unloaded", "rotor_overspeed", "thermal_rank"]
)
def test_fixed_command_and_rank_failures_are_distinguishable(spec, failure):
    point = operating_point(spec)
    anchors = inventory_anchors(point)
    c = point["controls"]["steam_cycle"]
    if failure == "feed":
        c["feed_mass_flow_kg_s"] *= 100
    elif failure == "valve":
        c["valve_opening"] = 0
    elif failure == "cooling":
        c["sink_temperature_k"] = 500
    elif failure == "rotor_unloaded":
        c["generator_connected"] = False
        point["steam_model"]["turbine"]["bearing_drag_nm_s_rad"] = 0
    elif failure == "rotor_overspeed":
        c["generator_connected"] = False
    else:
        for node in point["primary"]["loop"]["model"]["topology"]["nodes"]:
            node["parameters"]["fluid_wall_ua_w_k"] = 0
    expected = dict(
        feed="pressure_compatibility",
        valve="open_flow_required",
        cooling="condenser_compatibility",
        rotor_unloaded="rotor_rank",
        rotor_overspeed="rotor_band",
        thermal_rank="thermal_rank",
    )
    with pytest.raises(SteadySolveError) as caught:
        solve_fixed_primary_steam(point, **anchors)
    assert caught.value.report["criterion"] == expected[failure]


@pytest.mark.parametrize("total,quality", [(1, 0.1), (1e6, 0.1), (749, 0.5)])
def test_infeasible_inventory_anchors_do_not_project_mass_or_quality(spec, total, quality):
    point = operating_point(spec)
    before = deepcopy(point)
    with pytest.raises(SteadySolveError) as caught:
        solve_fixed_primary_steam(point, total_steam_mass_kg=total, steam_generator_quality=quality)
    assert caught.value.report["criterion"] == "inventory_anchor"
    assert point == before


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf")])
def test_invalid_numeric_anchors_fail_before_solving(spec, value):
    point = operating_point(spec)
    with pytest.raises((NetworkError, ValueError)):
        solve_fixed_primary_steam(point, total_steam_mass_kg=value, steam_generator_quality=0.1)


def test_timeline_cannot_be_silently_reinterpreted(spec):
    spec["timeline"] = [dict(t=0.1, controls=deepcopy(spec["controls"]))]
    with pytest.raises(NetworkError, match="empty timeline"):
        solve_fixed_primary_steam(spec, **inventory_anchors(spec))


@pytest.mark.parametrize("method", ["Radau", "BDF"])
@pytest.mark.parametrize("flow", [100, -100])
def test_solved_fixed_point_remains_stationary_with_linear_energy_counters(spec, method, flow):
    point = operating_point(spec, flow)
    solved, _ = solve_fixed_primary_steam(bad_guesses(point), **inventory_anchors(point))
    model = prepare(solved)[1]
    initial = model.observation()
    final = model.advance(2, method=method, rtol=1e-10, atol_scale=0.01, max_step_s=0.02)
    assert abs(final["energy_residual_j"]) < 0.01
    assert final["primary"]["loop"]["mass_flow_kg_s"] == pytest.approx(flow, abs=1e-8)
    for name in ("steam_generator", "condenser"):
        assert final["steam_cycle"]["vessels"][name]["pressure_pa"] == pytest.approx(
            initial["steam_cycle"]["vessels"][name]["pressure_pa"], abs=0.001
        )
    assert final["thermal_ports"]["primary_into_j"] < 0


def test_exact_boundary_is_rejected_without_projection_or_bogus_root_statistics(spec):
    spec["steam_model"]["condenser"]["initial_pressure_pa"] = 4000
    point = operating_point(spec)
    before = deepcopy(point)
    with pytest.raises(SteadySolveError) as caught:
        solve_fixed_primary_steam(point, **inventory_anchors(point))
    report = caught.value.report
    assert report["criterion"] == "active_pressure_boundary"
    assert report["details"]["pressure_root_mode"] == "endpoint_within_residual_budget"
    assert report["details"]["pressure_root_iterations"] == 0
    assert point == before


def test_cli_success_failure_manifest_and_no_replacement(spec, tmp_path):
    point = operating_point(spec)
    anchors = inventory_anchors(point)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(point))
    common = [
        str(path),
        "--total-steam-mass-kg",
        str(anchors["total_steam_mass_kg"]),
        "--steam-generator-quality",
        str(anchors["steam_generator_quality"]),
    ]
    output = tmp_path / "success"
    assert main([*common, "--output", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        assert sha(output / name) == digest
    with pytest.raises(FileExistsError):
        main([*common, "--output", str(output)])
    point["controls"]["primary_wall_heater_power_w"] += 1000
    path.write_text(json.dumps(point))
    failed = tmp_path / "rejected"
    assert main([*common, "--output", str(failed)]) == 2
    assert not (failed / "input.json").exists()
    assert (
        json.loads((failed / "diagnostics.json").read_text())["criterion"]
        == "hot_energy_compatibility"
    )


def test_invalid_primary_structure_and_extra_schema_fields_are_input_errors(spec):
    for bad in (dict(spec, unexpected=True), dict(spec, primary=None)):
        with pytest.raises(NetworkError):
            solve_fixed_primary_steam(bad, **inventory_anchors(spec))


def test_parameter_probe_respects_a_valid_higher_primary_temperature_floor(spec):
    spec["coupling"]["min_primary_temperature_k"] = 550
    point = operating_point(spec)
    solved, report = solve_fixed_primary_steam(bad_guesses(point), **inventory_anchors(point))
    assert report["accepted"] and report["parameter_probe_temperature_k"] > 550
    assert solved["coupling"]["min_primary_temperature_k"] == 550
    assert solved["controls"] == point["controls"]


@pytest.mark.parametrize("boundary", ["quality", "rotor", "subcooling", "primary_pressure"])
def test_all_active_passport_boundaries_are_rejected_without_changing_commands(spec, boundary):
    point = operating_point(spec)
    anchors = inventory_anchors(point)
    if boundary == "quality":
        anchors["steam_generator_quality"] = point["steam_model"]["steam_generator"]["min_quality"]
    else:
        solved, _ = solve_fixed_primary_steam(point, **anchors)
        if boundary == "rotor":
            omega = solved["steam_model"]["turbine"]["initial_omega_rad_s"]
            point["steam_model"]["turbine"].update(initial_omega_rad_s=omega, min_omega_rad_s=omega)
        else:
            margins = prepare(solved)[1].observation()["domain_margins"]
            if boundary == "subcooling":
                point["coupling"]["minimum_subcooling_k"] += min(
                    v["value"]
                    for k, v in margins.items()
                    if k.startswith("primary.") and k.endswith(".subcooling")
                )
            else:
                point["coupling"]["min_primary_pressure_pa"] += min(
                    v["value"]
                    for k, v in margins.items()
                    if k.startswith("primary.") and k.endswith(".pressure_low")
                )
    before = deepcopy(point)
    with pytest.raises(SteadySolveError) as caught:
        solve_fixed_primary_steam(point, **anchors)
    assert caught.value.report["criterion"] == "active_state_boundary"
    assert point == before


def test_empty_subcooled_temperature_interval_is_reported_before_parameter_probe(spec):
    point = operating_point(spec)
    anchors = inventory_anchors(point)
    point["coupling"]["min_primary_temperature_k"] = 700
    with pytest.raises(SteadySolveError) as caught:
        solve_fixed_primary_steam(point, **anchors)
    assert caught.value.report["criterion"] == "primary_temperature_domain"
