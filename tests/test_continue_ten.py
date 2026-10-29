"""Сопряжённый beta_eff, бор, датчик, косимуляция, критический поток и наблюдаемость."""

import math

import numpy as np
import pytest
from scipy.linalg import expm

from core.assembly import check_topology
from core.diagnostics import failure_report
from core.mesh import Mesh1D
from core.status import DomainError, SolverError
from coupling.cosim import zero_order_hold_pair
from instrumentation.channels import Measurement, Sensor
from physics.depletion.predictor import constant_power_step
from physics.hydraulics.network import FlowNetwork, Junction, Volume, diluted_boron_kg
from physics.hydraulics.two_phase_ext import (
    choked_mass_flow,
    critical_mass_flux_ideal_gas,
    equilibrium_mixture_critical_flux,
    frozen_mixture_critical_flux,
)
from physics.materials.iapws import saturation_properties
from physics.neutronics.diffusion import (
    adjoint_delayed_fraction,
    adjoint_k_eigenvalue,
    advance_fixed_flux_precursors,
    diffusion_operator,
    fission_operator,
    k_eigenvalue,
)
from physics.neutronics.xs import CrossSectionProvider
from validation.uq import require_observable


def test_adjoint_beta_and_fixed_flux_precursors():
    flux = np.array([1.0, 2.0])
    weight = np.array([4.0, 1.0])
    fission = np.array([1.0, 1.0])
    widths = np.array([1.0, 1.0])
    delayed = np.array([0.01, 0.02])
    expected = (4.0 * 0.01 * 1.0 + 1.0 * 0.02 * 2.0) / (4.0 * 1.0 + 1.0 * 2.0)
    assert adjoint_delayed_fraction(flux, weight, delayed, fission, widths) == pytest.approx(
        expected
    )
    assert expected != pytest.approx(float(np.mean(delayed)))

    mesh = Mesh1D.uniform(1.0, 8)
    removal = diffusion_operator(mesh, np.full(8, 0.01), np.full(8, 0.8))
    fission_matrix = fission_operator(mesh, np.full(8, 1.1))
    _k, phi, ok = k_eigenvalue(removal, fission_matrix)
    _k_star, phi_star, ok_star = adjoint_k_eigenvalue(removal, fission_matrix)
    assert ok and ok_star
    beta = 0.0065
    assert adjoint_delayed_fraction(phi, phi_star, beta, np.full(8, 1.1), mesh.dx) == pytest.approx(
        beta
    )

    cells = np.array([1.0, 3.0, 2.0])
    nu = np.array([0.4, 0.5, 0.2])
    decay = 0.08
    fraction = 0.007
    equilibrium = fraction * nu * cells / decay
    assert advance_fixed_flux_precursors(
        equilibrium, cells, fraction, nu, decay, 1.2
    ) == pytest.approx(equilibrium)
    start = equilibrium * 2.0
    advanced = advance_fixed_flux_precursors(start, cells, fraction, nu, decay, 4.0)
    assert advanced == pytest.approx(equilibrium + (start - equilibrium) * math.exp(-decay * 4.0))


def test_boron_dilution_matches_the_network_and_a_closed_pair_keeps_the_sum():
    mass = 80.0
    boron = 0.16
    exchange = 4.0
    assert diluted_boron_kg(mass, boron, exchange, 0.0, 0.0) == pytest.approx(boron)
    later = diluted_boron_kg(mass, boron, exchange, 0.0, 20.0)
    assert later == pytest.approx(boron * math.exp(-exchange * 20.0 / mass))
    rate = exchange * (0.0 - boron / mass)
    short = diluted_boron_kg(mass, boron, exchange, 0.0, 1.0e-4)
    assert (short - boron) / 1.0e-4 == pytest.approx(rate, rel=1.0e-4)

    hot = Volume("hot", 2.0, 0.0)
    cold = Volume("cold", 2.0, 0.0)
    hot.set_liquid(15.5e6, 560.0, boron_ppm=2000.0)
    cold.set_liquid(15.5e6, 560.0, boron_ppm=100.0)
    link = Junction(
        "link",
        "hot",
        "cold",
        area_m2=0.05,
        length_m=1.0,
        loss_coefficient=0.0,
        mass_flow_kg_s=12.0,
        flow_law="target",
        flow_target_kg_s=12.0,
    )
    network = FlowNetwork({"hot": hot, "cold": cold}, {"link": link})
    derivative = network.derivatives(network.pack())
    assert derivative[2] == pytest.approx(-12.0 * hot.boron_fraction)
    assert derivative[5] == pytest.approx(12.0 * hot.boron_fraction)
    assert derivative[2] + derivative[5] == pytest.approx(0.0, abs=1.0e-12)
    link.mass_flow_kg_s = -12.0
    link.flow_target_kg_s = -12.0
    reverse = network.derivatives(network.pack())
    assert reverse[2] == pytest.approx(12.0 * cold.boron_fraction)
    assert reverse[5] == pytest.approx(-12.0 * cold.boron_fraction)
    assert reverse[2] + reverse[5] == pytest.approx(0.0, abs=1.0e-12)


def test_open_and_stuck_sensors_are_different_faults():
    stuck = Sensor(tau_s=1.0, value=10.0, fault="stuck")
    assert stuck.step(40.0, 1.0, np.random.default_rng(0)) == pytest.approx(10.0)
    assert stuck.quality == "bad"
    opened = Sensor(tau_s=1.0, value=10.0, fault="open")
    reading = Measurement("pressure", opened).read(12.0, 0.2, np.random.default_rng(1))
    assert math.isnan(reading["value"])
    assert reading["quality"] == "bad"
    assert reading["fault"] == "open"
    assert reading["unit"] == "Pa"
    with pytest.raises(ValueError):
        Sensor(tau_s=1.0, failed=True, fault="open")


def test_cosim_error_shrinks_and_failure_names_the_constraint():
    matrix = np.array([[-2.0, 1.0], [1.0, -3.0]])
    exact = expm(matrix * 1.0) @ np.array([1.0, 0.0])
    coarse = np.array(zero_order_hold_pair(1.0, 0.0, 1.0, 0.2))
    fine = np.array(zero_order_hold_pair(1.0, 0.0, 1.0, 0.05))
    assert np.linalg.norm(fine - exact) < 0.5 * np.linalg.norm(coarse - exact)
    with pytest.raises(DomainError):
        zero_order_hold_pair(1.0, 0.0, 1.0, 0.3)

    report = failure_report(
        "pressurizer", "p <= p_max", 1.5e5, "advance(0.25)", DomainError("dome")
    )
    assert report.error_class == "domain"
    assert report.residual == pytest.approx(1.5e5)
    solver = failure_report("integrator", "BDF step", 0.0, "same seed", SolverError("step"))
    assert solver.error_class == "solver"
    with pytest.raises(ValueError):
        failure_report("volume", "mass", math.nan, "retry", DomainError("nan"))


def test_constant_power_corrector_equilibrium_choke_and_boron_table():
    predicted, corrected, linear = constant_power_step(1.0e3, 2.0e-28, 1.0e18, 1.0e9)
    assert abs(corrected - linear) < abs(predicted - linear)
    with pytest.raises(DomainError):
        constant_power_step(1.0, 1.0, 1.0, 2.0)

    pressure = 1.0e6
    quality = 0.2
    equilibrium = equilibrium_mixture_critical_flux(pressure, quality)
    frozen = frozen_mixture_critical_flux(pressure, quality)
    liquid, _vapor = saturation_properties(pressure)
    ideal = critical_mass_flux_ideal_gas(pressure, liquid.T_k, 1.33, 461.526)
    assert equilibrium > 0.0
    assert equilibrium < frozen
    assert equilibrium != pytest.approx(ideal, rel=0.2)
    choked = choked_mass_flow(pressure, 0.2e6, 0.5, equilibrium, 1.0e-3)
    lower = choked_mass_flow(pressure, 0.05e6, 0.5, equilibrium, 1.0e-3)
    assert lower == pytest.approx(choked)
    with pytest.raises(DomainError):
        equilibrium_mixture_critical_flux(pressure, 0.0)

    temperatures = np.array([600.0, 900.0])
    groups = np.array([[1.0, 2.0], [1.0, 2.0]])
    provider = CrossSectionProvider(
        temperatures,
        {
            "D": np.ones((2, 2)),
            "sigma_a": groups,
            "sigma_f": groups,
            "nu_sigma_f": groups,
            "kappa_sigma_f": groups,
            "scatter": np.zeros((2, 2, 2)),
        },
        boron_ppm=np.array([0.0, 2000.0]),
        boron_delta_sigma_a=np.array([[0.0, 0.0], [0.2, 0.4]]),
    )
    middle = provider.at_state(600.0, boron_ppm=1000.0)
    assert middle.sigma_a_per_m == pytest.approx(np.array([1.1, 2.2]))
    bare = CrossSectionProvider(
        temperatures,
        {
            "D": np.ones((2, 2)),
            "sigma_a": groups,
            "sigma_f": groups,
            "nu_sigma_f": groups,
            "kappa_sigma_f": groups,
            "scatter": np.zeros((2, 2, 2)),
        },
    )
    with pytest.raises(DomainError):
        provider.at_state(600.0, boron_ppm=5000.0)
    with pytest.raises(DomainError):
        bare.at_state(600.0, boron_ppm=100.0)


def test_amplitude_is_observable_and_a_loose_volume_is_rejected():
    generation = 1.0e-4
    beta = 0.0065
    decay = 0.08
    reactivity = -0.002
    state = np.array(
        [
            [(reactivity - beta) / generation, decay],
            [beta / generation, -decay],
        ]
    )
    assert require_observable(state, np.array([[1.0, 0.0]])) == 2
    with pytest.raises(DomainError):
        require_observable(state, np.array([[0.0, 0.0]]))

    loose = check_topology(
        ["core", "pressurizer"], [("surge", "core", "boundary:cold")], {"boundary:cold": 1.0e5}
    )
    assert any("pressurizer" in error for error in loose.errors)
    paired = check_topology(["left", "right"], [], {})
    assert not paired.ok
    closed = check_topology(["tank"], [], {})
    assert closed.ok
