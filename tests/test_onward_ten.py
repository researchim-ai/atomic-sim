"""Отражение, пар без бора, критический бор, нормировка мощности и манифест."""

import math

import numpy as np
import pytest

from control.regulation import HysteresisSwitch
from core.manifest import input_manifest
from core.mesh import Mesh1D
from core.status import DomainError
from instrumentation.channels import Measurement, Sensor
from instrumentation.replay import _measurement_from_payload, _measurement_payload
from physics.depletion.chains import DecayHeat, XenonIodine
from physics.hydraulics.network import BoundaryState, FlowNetwork, Junction, Volume
from physics.materials.iapws import saturation_properties
from physics.neutronics.diffusion import (
    diffusion_operator,
    fission_operator,
    k_eigenvalue,
    macroscopic_at_density,
)
from physics.neutronics.transport import (
    analytical_reflective_k,
    diffusion_k_sparse,
    recover_manufactured_flux,
)
from physics.neutronics.xs import fission_power_w, normalize_flux_to_power
from validation.uq import critical_boron_ppm


def test_reflective_buckling_improves_and_manufactured_flux_returns():
    diffusion = 0.01
    absorption = 0.2
    fission = 0.3
    lengths = (1.0, 1.2)
    reflected = (True, False)
    expected = analytical_reflective_k(diffusion, absorption, fission, lengths, reflected)
    coarse, _phi, ok = diffusion_k_sparse(
        (8, 8), lengths, diffusion, absorption, fission, lower_reflective=reflected
    )
    fine, _phi2, ok2 = diffusion_k_sparse(
        (16, 16), lengths, diffusion, absorption, fission, lower_reflective=reflected
    )
    assert ok and ok2
    assert abs(fine - expected) < abs(coarse - expected)

    shape = (3, 3, 2)
    rng = np.random.default_rng(3)
    conductivity = 0.01 + 0.02 * rng.random(shape)
    flux = rng.random(shape)
    recovered = recover_manufactured_flux(shape, (1.0, 0.8, 0.5), conductivity, 0.25, flux)
    assert recovered == pytest.approx(flux, abs=1.0e-10)


def test_vapor_leaves_boron_behind_and_density_increases_leakage():
    drum = Volume("drum", 8.0, 0.0)
    drum.set_quality(6.0e6, 0.2)
    drum.boron_kg = 0.4
    _liquid, vapor = saturation_properties(drum.pressure_pa)
    sink = BoundaryState(1.0e5, vapor.h_j_kg, 1.0, 0.0, boron_fraction=0.0)
    steam = Junction(
        "steam",
        "drum",
        "sink",
        0.2,
        4.0,
        0.0,
        mass_flow_kg_s=3.0,
        enthalpy_mode="saturated_vapor",
        boron_carry=0.0,
        flow_law="target",
        flow_target_kg_s=3.0,
    )
    network = FlowNetwork({"drum": drum}, {"steam": steam}, {"sink": sink})
    derivative = network.derivatives(network.pack())
    assert derivative[0] == pytest.approx(-3.0)
    assert derivative[1] == pytest.approx(-3.0 * vapor.h_j_kg, rel=1.0e-8)
    assert derivative[2] == pytest.approx(0.0, abs=1.0e-12)
    concentration_rate = -drum.boron_kg * derivative[0] / drum.mass_kg**2
    assert concentration_rate > 0.0

    mesh = Mesh1D.uniform(1.0, 24)
    reference = macroscopic_at_density(0.02, 0.3, 0.36, 1.0)
    dilute = macroscopic_at_density(0.02, 0.3, 0.36, 0.7)

    def eigenvalue(macros):
        diffusion, absorption, nu = macros
        removal = diffusion_operator(mesh, np.full(24, diffusion), np.full(24, absorption))
        source = fission_operator(mesh, np.full(24, nu))
        value, _flux, ok = k_eigenvalue(removal, source)
        assert ok
        return value

    assert dilute[2] / dilute[1] == pytest.approx(reference[2] / reference[1])
    assert eigenvalue(dilute) < eigenvalue(reference)


def test_critical_boron_power_norm_decay_and_manifest():
    def reactivity(ppm: float) -> float:
        return 8.0e-5 * (900.0 - ppm)

    assert critical_boron_ppm(reactivity, 0.0, 2000.0) == pytest.approx(900.0)
    with pytest.raises(DomainError):
        critical_boron_ppm(reactivity, 0.0, 100.0)

    kappa = np.array([1.0e-11, 2.0e-11])
    flux = np.array([1.0e18, 2.0e18])
    volume = np.array([0.2, 0.2])
    scaled, factor = normalize_flux_to_power(kappa, flux, volume, 2.0e6)
    assert fission_power_w(kappa, scaled, volume) == pytest.approx(2.0e6)
    assert scaled / flux == pytest.approx(factor)

    heat = DecayHeat.illustrative()
    stored = np.array([1.0e7, 2.0e7, 3.0e7])
    exact = heat.advance_constant_fission(stored, 1.0e6, 40.0)
    numerical = heat.integrate(stored, 1.0e6, 40.0)
    assert numerical == pytest.approx(exact, rel=1.0e-5)
    stopped = heat.advance_constant_fission(stored, 0.0, 25.0)
    _released, remaining = heat.shutdown_release_j(stored, 25.0)
    assert stopped == pytest.approx(remaining)

    left = input_manifest({"boron_ppm": 1100.0, "rod": 0.2, "flags": [True, False]})
    right = input_manifest({"flags": [True, False], "rod": 0.2, "boron_ppm": 1100.0})
    assert left["sha256"] == right["sha256"]
    changed = input_manifest({"boron_ppm": 1101.0, "rod": 0.2, "flags": [True, False]})
    assert changed["sha256"] != left["sha256"]
    with pytest.raises(DomainError):
        input_manifest({"power": math.nan})


def test_hysteresis_xenon_field_and_open_sensor_snapshot():
    switch = HysteresisSwitch(engage_at=1.0, release_at=0.2)
    assert switch.step(0.5) is False
    assert switch.step(1.0) is True
    assert switch.step(0.5) is True
    assert switch.step(0.2) is False
    with pytest.raises(ValueError):
        HysteresisSwitch(engage_at=0.2, release_at=0.2)

    xenon = XenonIodine()
    fission = np.array([1.0e18, 2.0e18])
    flux = np.array([1.0e17, 4.0e17])
    iodine, concentration = xenon.equilibrium_field(fission, flux)
    for index in range(2):
        rate = xenon.derivatives(iodine[index], concentration[index], fission[index], flux[index])
        production = xenon.gamma_xe * fission[index] + xenon.lambda_i * iodine[index]
        assert rate[0] == pytest.approx(0.0, abs=1.0e-9 * fission[index])
        assert rate[1] == pytest.approx(0.0, abs=1.0e-9 * production)
    assert concentration[1] / fission[1] < concentration[0] / fission[0]

    sensor = Sensor(tau_s=1.0, value=3.0, fault="open")
    measurement = Measurement("pressure", sensor, time_s=1.5)
    restored = _measurement_from_payload(_measurement_payload(measurement))
    reading = restored.read(9.0, 0.1, np.random.default_rng(0))
    assert restored.sensor.fault == "open"
    assert math.isnan(reading["value"])
    assert reading["quality"] == "bad"
    legacy = _measurement_payload(measurement)
    del legacy["fault"]
    assert _measurement_from_payload(legacy).sensor.fault == ""
