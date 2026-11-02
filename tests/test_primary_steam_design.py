"""Joint steady design, independent dimensional residuals and physical rejection."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from core.network import NetworkError
from core.status import DomainError
from physics.power.primary_steam import PrimarySteamCycle
from scripts.design_primary_steam import design_steady_primary_steam, main
from scripts.primary_steam_references import energy_balances, power_balance, reference, system
from scripts.verify_turbine import sha
from sdk.primary_steam_experiment import prepare


@pytest.fixture
def spec():
    raw = json.loads(Path("scenarios/ai05_primary_steam.json").read_text())
    raw["timeline"] = []
    return raw


@pytest.mark.parametrize("flow", [100.0, -100.0, 10.0, -10.0])
@pytest.mark.parametrize("opening", [1.0, 0.6])
def test_joint_design_matches_independent_balance_without_warmup(spec, flow, opening, monkeypatch):
    original = deepcopy(spec)

    def forbidden(*args, **kwargs):
        raise AssertionError("A steady design must not warm up a transient")

    monkeypatch.setattr(PrimarySteamCycle, "advance_until_limit", forbidden)
    normalized, report = design_steady_primary_steam(
        spec, primary_mass_flow_kg_s=flow, valve_opening=opening
    )
    assert spec == original
    assert report["warmup_steps"] == 0 and report["maximum_stock_power_residual_w"] < 1e-5
    assert report["chosen_commands"] == normalized["controls"]
    assert normalized["controls"]["steam_cycle"]["heater_power_w"] == 0
    assert np.sign(normalized["primary"]["initial_omega_rad_s"]) == np.sign(flow)
    y, rhs, _, _, data, k = system(normalized)
    rates = rhs(0, y, normalized["controls"])
    assert max(abs(rates[: 2 + 2 * data["n"]])) < 1e-8
    assert max(abs(rates[k : k + 7])) < 1e-5
    assert abs(power_balance(y, rates, data, k, normalized)) < 1e-5
    assert energy_balances(y, y, data, k, normalized)["total_residual_j"] == 0
    prepare(normalized)


def test_designer_replaces_temperature_guesses_and_handles_existing_losses(spec):
    original = deepcopy(spec)
    for node in spec["primary"]["loop"]["model"]["topology"]["nodes"]:
        node["parameters"].update(ambient_ua_w_k=10, wall_power_w=100)
    init = spec["primary"]["loop"]["initialization"]
    for key in ("fluid_temperature_k", "wall_temperature_k"):
        init[key] = {name: 1000 for name in init[key]}
    normalized, report = design_steady_primary_steam(spec, primary_mass_flow_kg_s=100)
    assert (
        max(normalized["primary"]["loop"]["initialization"]["fluid_temperature_k"].values()) < 610
    )
    assert report["maximum_stock_power_residual_w"] < 1e-5
    baseline = design_steady_primary_steam(original, primary_mass_flow_kg_s=100)[1]
    assert (
        report["chosen_commands"]["primary_wall_heater_power_w"]
        > baseline["chosen_commands"]["primary_wall_heater_power_w"]
    )


@pytest.mark.parametrize(
    "flow,message", [(0.0, "unanchored"), (1000.0, "cooler"), (1.0, "subcooling")]
)
def test_physically_unavailable_steady_points_are_rejected(spec, flow, message):
    before = deepcopy(spec)
    with pytest.raises((NetworkError, DomainError), match=message):
        design_steady_primary_steam(spec, primary_mass_flow_kg_s=flow)
    assert spec == before


@pytest.mark.parametrize("flow", [True, float("nan"), float("inf")])
def test_invalid_design_flow_is_rejected(spec, flow):
    with pytest.raises(NetworkError):
        design_steady_primary_steam(spec, primary_mass_flow_kg_s=flow)


def test_commands_are_not_silently_rewritten(spec):
    spec["timeline"] = [dict(t=0.1, controls=deepcopy(spec["controls"]))]
    before = deepcopy(spec)
    with pytest.raises(NetworkError, match="empty command timeline"):
        design_steady_primary_steam(spec, primary_mass_flow_kg_s=100)
    assert spec == before


@pytest.mark.parametrize("method", ["Radau", "BDF"])
def test_steady_point_remains_stationary_in_two_integrators(spec, method):
    normalized, _ = design_steady_primary_steam(spec, primary_mass_flow_kg_s=100)
    normalized.update(duration_s=2, step_s=0.25)
    model = prepare(normalized)[1]
    initial = model.observation()
    at, boundary, details = reference(normalized)
    assert boundary is None
    after = model.advance(2, method=method, rtol=1e-10, max_step_s=0.02, atol_scale=0.01)
    assert after["primary"]["loop"]["mass_flow_kg_s"] == pytest.approx(100, abs=1e-8)
    for name in ("steam_generator", "condenser"):
        assert after["steam_cycle"]["vessels"][name]["pressure_pa"] == pytest.approx(
            initial["steam_cycle"]["vessels"][name]["pressure_pa"], abs=0.001
        )
    assert abs(after["energy_residual_j"]) < 1e-5
    assert max(abs(at(2)[: 2 + 2 * details[3]["n"]] - at(0)[: 2 + 2 * details[3]["n"]])) < 1e-8


def test_cli_design_manifest_and_exact_failure_outputs(spec, tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps(spec))
    output = tmp_path / "designed"
    main([str(request), "--primary-flow-kg-s", "-100", "--output", str(output)])
    manifest = json.loads((output / "manifest.json").read_text())
    for name, checksum in manifest["artifacts"].items():
        assert sha(output / name) == checksum
    assert json.loads((output / "design.json").read_text())["warmup_steps"] == 0
    prepare(json.loads((output / "input.json").read_text()))
    rejected = tmp_path / "rejected"
    with pytest.raises(NetworkError):
        main([str(request), "--primary-flow-kg-s", "0", "--output", str(rejected)])
    assert not rejected.exists()
