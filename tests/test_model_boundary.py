"""Допустимый префикс, отдельный статус границы и транзакционный баланс."""

import copy
import csv
import hashlib
import json

import pytest

from core.status import DomainError, ModelLimitError, SolverError
from plants.ledger import HorizonLedger
from sdk.api import create_session
from sdk.boundary import advance_until_limit
from sdk.experiment import run_experiment, validate_experiment


class LinearPlant:
    def __init__(self, limit=3.7, interruption=None):
        self.time_s = 0.0
        self.limit = limit
        self.interruption = interruption
        self.accepted = 0
        self.synchronized_at = 0.0
        self.last_energy_ledger = None
        self.last_motion_events = ()
        self.last_protection_event_s = None
        self.horizon_ledger = None

    def snapshot(self):
        from types import SimpleNamespace

        return SimpleNamespace(time_s=self.time_s)

    def restore(self, value):
        self.time_s = value.time_s

    def observation(self):
        return {"time_s": self.time_s}

    def pack(self):
        return self.time_s

    def refresh_algebraic(self, state):
        assert state == self.time_s
        self.synchronized_at = state

    def model_domain_diagnostics(self):
        return {"linear": {"observed": self.time_s, "limit": self.limit}}

    def closed_inventories(self):
        return {"stored_j": 100.0 + self.time_s, "closed_mass_kg": 1.0, "boron_kg": 0.1}

    def advance(self, dt, **options):
        if self.accepted and self.interruption:
            self.synchronized_at = -1.0
            raise self.interruption("injected interruption")
        end = self.time_s + dt
        if end > self.limit:
            error = ModelLimitError(
                "analytic boundary", criterion="test", node="linear", observed=end, limit=self.limit
            )
            error.evaluation_time_s = end
            error.evaluation_phase = "rhs_probe"
            raise error
        self.last_motion_events = (
            ({"kind": "rod_stop", "t": 0.2, "position": 1.0},) if self.time_s < 0.2 <= end else ()
        )
        self.time_s = end
        self.synchronized_at = end
        self.last_energy_ledger = {
            "delta_j": {"linear": dt},
            "supplied_j": {"linear": dt},
            "residual_j": 0.0,
            "interval_s": dt,
        }
        self.accepted += 1
        if self.horizon_ledger:
            self.horizon_ledger.observe(self)
        return self.observation()


@pytest.mark.parametrize("tolerance", [1e-2, 1e-4, 1e-6])
def test_analytic_boundary_preserves_valid_prefix_and_exact_balance(tolerance):
    plant = LinearPlant()
    ledger = HorizonLedger()
    ledger.attach(plant)
    result = advance_until_limit(plant, 5.0, time_tolerance_s=tolerance)
    boundary = result.boundary
    assert 0 <= 3.7 - plant.time_s <= tolerance
    assert boundary["interval_width_s"] <= tolerance
    assert boundary["failed_endpoint_s"] > 3.7
    assert boundary["scope"] == "integration_admissibility"
    assert not boundary["physical_crossing_localized"]
    assert boundary["last_accepted_state"]["linear"]["observed"] <= 3.7
    assert result.events == [{"kind": "rod_stop", "t": 0.2, "position": 1.0}]
    assert ledger.report()["energy_global_j"] == pytest.approx(0.0, abs=1e-12)
    assert ledger.report()["steps"] == result.accepted_prefix_steps
    assert plant.last_energy_ledger["interval_s"] == plant.time_s
    assert plant.last_energy_ledger["supplied_j"]["linear"] == pytest.approx(plant.time_s)


def test_normal_step_has_no_boundary_and_no_extra_advances():
    plant = LinearPlant()
    result = advance_until_limit(plant, 1.0)
    assert result.boundary is None
    assert result.accepted_prefix_steps == 1
    assert plant.time_s == 1.0


@pytest.mark.parametrize("error_type", [SolverError, DomainError, RuntimeError, KeyboardInterrupt])
def test_other_failures_roll_back_state_metadata_and_partial_ledger(error_type):
    plant = LinearPlant(interruption=error_type)
    ledger = HorizonLedger()
    ledger.attach(plant)
    before = copy.deepcopy(ledger.__dict__)
    with pytest.raises(error_type, match="injected interruption"):
        advance_until_limit(plant, 5.0)
    assert plant.time_s == 0.0
    assert plant.last_energy_ledger is None
    assert plant.last_motion_events == ()
    assert ledger.__dict__ == before
    assert plant.synchronized_at == 0.0


def test_boundary_at_initial_time_has_no_accepted_prefix():
    plant = LinearPlant(limit=0.0)
    result = advance_until_limit(plant, 1.0)
    assert result.accepted_prefix_steps == 0
    assert plant.time_s == 0.0
    assert result.events == []


@pytest.mark.parametrize("value", [True, 0.0, float("nan"), 0.2])
def test_experiment_rejects_invalid_boundary_tolerance(value):
    with pytest.raises(ValueError):
        validate_experiment(
            dict(
                name="bad",
                schema_version=1,
                engine="vver",
                duration_s=1.0,
                domain_boundary_tolerance_s=value,
            )
        )


def test_experiment_stores_last_valid_prefix_without_claiming_completion(monkeypatch, tmp_path):
    plant = create_session()
    original = plant.derivatives

    def rhs(t, y):
        if t > 0.04:
            exc = ModelLimitError(
                "test boundary", criterion="time", node="test", observed=t, limit=0.04
            )
            exc.evaluation_time_s, exc.evaluation_phase = t, "rhs_probe"
            raise exc
        return original(t, y)

    plant.derivatives = rhs
    monkeypatch.setattr("sdk.experiment.create_session", lambda kind: plant)
    spec = dict(
        name="bounded",
        schema_version=1,
        engine="vver",
        duration_s=0.1,
        step_s=0.1,
        domain_boundary_tolerance_s=1e-5,
        timeline=[{"t": 0.0, "throttle": 0.98}],
        solver={"method": "Radau", "jacobian_relative_step": 1e-5},
        budgets={"mass_relative": 1e-6},
    )
    folder = tmp_path / "run"
    report = run_experiment(spec, folder)
    assert report["status"] == "model_limit"
    assert not report["numerical_acceptance"]
    assert 0.04 - 1e-5 <= report["completed_time_s"] <= 0.04
    assert report["model_boundary"]["interval_width_s"] <= 1e-5
    checkpoint = json.loads((folder / "checkpoint.json").read_text())
    assert checkpoint["time_s"] == report["completed_time_s"]
    assert checkpoint["throttle"] == 0.98
    assert (
        float(list(csv.DictReader((folder / "trajectory.csv").open()))[-1]["time_s"])
        == checkpoint["time_s"]
    )
    with (folder / "trajectory.csv").open() as stream:
        residuals = [
            float(row["energy_residual_j"])
            for row in csv.DictReader(stream)
            if row["energy_residual_j"]
        ]
    assert sum(residuals) == pytest.approx(report["balance"]["energy_integral_j"], abs=1e-6)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest


def test_zero_prefix_rolls_back_unaccepted_commands_in_experiment(monkeypatch, tmp_path):
    plant = create_session()

    def fail(t, y):
        raise ModelLimitError(
            "no admissible advance", criterion="test", node="test", observed=1.0, limit=0.0
        )

    plant.derivatives = fail
    monkeypatch.setattr("sdk.experiment.create_session", lambda kind: plant)
    report = run_experiment(
        dict(
            name="zero",
            schema_version=1,
            engine="vver",
            duration_s=0.1,
            step_s=0.1,
            domain_boundary_tolerance_s=1e-4,
            timeline=[{"t": 0.0, "throttle": 0.98}],
        ),
        tmp_path / "run",
    )
    assert report["status"] == "model_limit"
    assert report["accepted_steps"] == 0
    assert report["balance"] is None
    assert plant.throttle == 1.0
    assert all(event["kind"] != "command" for event in report["events"])
    checkpoint = json.loads((tmp_path / "run/checkpoint.json").read_text())
    assert checkpoint["throttle"] == 1.0
