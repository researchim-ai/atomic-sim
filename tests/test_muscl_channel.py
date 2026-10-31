import json
from dataclasses import replace

import pytest

from core.status import DomainError
from physics.hydraulics.dynamic_channel import DynamicWallChannel, _check_face_liquid
from physics.hydraulics.initialization import equilibrated_config
from scripts.generate_pipe_pulse import make_spec
from scripts.verify_pipe_mesh import configuration
from sdk.dynamic_experiment import run_dynamic_experiment, validate_dynamic_experiment


@pytest.mark.parametrize("flow", [-1.0, 1.0])
def test_muscl_global_energy_closes_for_both_flow_directions(flow):
    c = replace(
        configuration(4, 0.0),
        enthalpy_transport="muscl_mc",
        initial_flow_kg_s=flow,
        initial_temperature_profile_k=(499.0, 500.0, 501.0, 502.0),
        initial_wall_temperature_profile_k=(505.0,) * 4,
    )
    m = DynamicWallChannel(c)
    rates = m.rates(0.0, m._state)
    d = rates["derivative"]
    assert sum(d[:8:2]) == pytest.approx(rates["boundary_mass_kg_s"], abs=1e-12)
    assert sum(d[1:8:2]) + rates["wall_energy_rate_w"] == pytest.approx(
        rates["boundary_energy_w"], abs=1e-7
    )
    # MUSCL must actually alter the nonuniform donor energy operator.
    donor = DynamicWallChannel(replace(c, enthalpy_transport="donor"))
    assert max(abs(d[1:8:2] - donor.derivatives(0.0, donor._state)[1:8:2])) > 1.0


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_muscl_profile_input_snapshot_and_restart(method, tmp_path):
    s = make_spec(4, scheme="muscl_mc")
    s.update(duration_s=0.01, step_s=0.005)
    s["solver"]["method"] = method
    p = tmp_path / "run"
    r = run_dynamic_experiment(s, p)
    assert r["numerical_acceptance"]
    assert r["solver_diagnostics"]["accepted_output_intervals"] == 2
    payload = json.loads((p / "checkpoint.json").read_text())
    m = DynamicWallChannel.from_snapshot(payload)
    assert m.snapshot() == payload and m.config.enthalpy_transport == "muscl_mc"
    normalized = json.loads((p / "input.json").read_text())
    assert validate_dynamic_experiment(normalized) == normalized
    m.advance(0.001, method=method, absolute_tolerances=s["solver"]["absolute_tolerances"])
    assert abs(m.observation()["energy_residual_j"]) < 0.001
    assert m.solver_diagnostics()["accepted_output_intervals"] == 1  # process-local reset


def test_unknown_scheme_and_nonpipe_muscl_are_rejected():
    with pytest.raises(ValueError):
        replace(configuration(4, 0.0), enthalpy_transport="invented")
    with pytest.raises(ValueError):
        replace(configuration(4, 0.0), enthalpy_transport="muscl_mc", junction_lengths_m=None)


def test_reconstructed_steam_is_not_silently_projected_to_liquid():
    from physics.materials.iapws import saturation_properties

    liquid, vapor = saturation_properties(7e6)
    with pytest.raises(DomainError, match="Reconstructed"):
        _check_face_liquid(7e6, (liquid.h_j_kg + vapor.h_j_kg) / 2)


def test_heated_donor_initializer_cannot_claim_a_muscl_equilibrium():
    with pytest.raises(DomainError, match="Heated MUSCL"):
        equilibrated_config(replace(configuration(4, 3000.0), enthalpy_transport="muscl_mc"))


def test_tolerance_policy_reset_and_rejected_interval_counters(monkeypatch):
    m = DynamicWallChannel(equilibrated_config(configuration(3, 0.0), wall_offset_k=1.0))
    m.enable_continuation(0.03)
    m.advance(0.01, absolute_tolerances={"flow_kg_s": 1e-9})
    first = m._continuation
    m.advance(0.01, absolute_tolerances={"flow_kg_s": 1e-10})
    assert m._continuation is not first
    counts = m.solver_diagnostics()
    assert counts["accepted_rhs_evaluations"] > 0 and counts["accepted_solver_steps"] > 0
    saved = m.snapshot()
    old = m.derivatives

    def fail(t, y):
        old(t, y)
        raise DomainError("injected failure")

    monkeypatch.setattr(m, "derivatives", fail)
    with pytest.raises(DomainError):
        m.advance(0.01)
    assert m.snapshot() == saved
    assert m.solver_diagnostics()["accepted_output_intervals"] == 2
    assert m.solver_diagnostics()["failed_output_intervals"] == 1
    assert m.solver_diagnostics()["accepted_rhs_evaluations"] == counts["accepted_rhs_evaluations"]


@pytest.mark.parametrize(
    "tolerances",
    [{"invented": 1e-8}, {"mass_kg": True}, {"flow_kg_s": 0.0}, {"energy_j": float("nan")}],
)
def test_invalid_tolerance_budget_creates_no_output(tmp_path, tolerances):
    s = make_spec(4)
    s["solver"]["absolute_tolerances"] = tolerances
    with pytest.raises(ValueError):
        run_dynamic_experiment(s, tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    "pressure,temperature",
    [(1e5, 300.0), (1e6, 400.0), (7e6, 500.0), (15.5e6, 580.0), (30e6, 620.0)],
)
def test_liquid_face_admission_matches_if97_reference_states(pressure, temperature):
    from physics.materials.iapws import properties_ph, properties_pt

    state = properties_pt(pressure, temperature)
    assert state.region == 1 and properties_ph(pressure, state.h_j_kg).region == 1
    _check_face_liquid(pressure, state.h_j_kg)


def test_face_admission_rejects_both_below_temperature_range_and_above_saturation():
    from physics.materials.iapws import properties_pt, saturation_properties

    cold = properties_pt(7e6, 273.16)
    liquid, _ = saturation_properties(7e6)
    with pytest.raises(DomainError):
        _check_face_liquid(7e6, cold.h_j_kg - 1000.0)
    with pytest.raises(DomainError):
        _check_face_liquid(7e6, liquid.h_j_kg + 1.0)
