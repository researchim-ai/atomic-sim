"""Эталоны ядра: аналитика, сохранения и границы области."""

import math

import numpy as np
import pytest
from scipy.linalg import expm

from components.equipment import coastdown_omega, orifice_mass_flow
from control.regulation import PID, ProtectionLogic
from core.assembly import EquationSpec, check_topology, differential_mask
from core.events import EventSpec, collect_sign_changes
from core.mesh import Mesh1D
from core.solvers import integrate, substep_count
from core.status import Applicability, DomainError
from core.units import Quantity, barns_to_m2, per_cm_to_per_m
from coupling.maps import ExternalSolverAdapter, conserve_integral, picard_power
from data.open_library import open_two_group_provider
from instrumentation.channels import Actuator, Sensor
from physics.depletion.chains import (
    DecayHeat,
    XenonIodine,
    advance_nuclides,
    bateman_daughter,
    decay_chain_matrix,
)
from physics.hydraulics.closures import heat_transfer_coefficient, nusselt_dittus_boelter
from physics.hydraulics.network import BoundaryState, FlowNetwork, Junction, Volume
from physics.materials.coolants import helium_density_kg_m3, sodium_density_kg_m3
from physics.materials.iapws import (
    properties_ph,
    properties_pt,
    properties_rho_u,
    saturation_pressure_mpa,
    saturation_properties,
    viscosity_pa_s,
    void_fraction,
)
from physics.materials.solids import uo2_conductivity_w_m_k
from physics.neutronics.diffusion import (
    analytical_slab_k,
    diffusion_operator,
    fission_operator,
    k_eigenvalue,
    removal_with_scatter,
    rod_absorption,
    two_group_infinite_k,
)
from physics.neutronics.point_kinetics import PointKinetics, batch_derivatives, inhour_root
from physics.neutronics.xs import micro_to_macro
from physics.thermal.fuel_rod import (
    FuelRod,
    analytical_cylinder_temperature,
    cell_average_analytical,
    gap_conductance,
)
from plants.rbmk import RBMKChannel
from plants.registry import build_plant


def test_units_do_not_mix():
    with pytest.raises(TypeError):
        Quantity(1.0, "m") + Quantity(1.0, "s")
    assert barns_to_m2(1.0) == pytest.approx(1e-28)
    assert per_cm_to_per_m(0.015) == pytest.approx(1.5)


def test_iapws_reference_points():
    liquid = properties_pt(3.0e6, 300.0)
    assert liquid.v_m3_kg == pytest.approx(0.00100215168, rel=1e-8)
    assert liquid.h_j_kg / 1000.0 == pytest.approx(115.331273, rel=1e-8)
    assert liquid.u_j_kg / 1000.0 == pytest.approx(112.324818, rel=1e-8)
    dense = properties_pt(80.0e6, 300.0)
    assert dense.s_j_kg_k / 1000.0 == pytest.approx(0.368563852, rel=1e-8)
    hot = properties_pt(3.0e6, 500.0)
    assert hot.sound_speed_m_s == pytest.approx(1240.71337, rel=1e-7)
    vapor = properties_pt(30.0e6, 700.0)
    assert vapor.v_m3_kg == pytest.approx(0.00542946619, rel=1e-7)
    assert vapor.h_j_kg / 1000.0 == pytest.approx(2631.49474, rel=1e-7)
    assert saturation_pressure_mpa(300.0) == pytest.approx(0.00353658941, rel=1e-8)
    assert saturation_pressure_mpa(500.0) == pytest.approx(2.63889776, rel=1e-8)
    state = properties_ph(3.0e6, 500.0e3)
    assert state.h_j_kg == pytest.approx(500.0e3, rel=1e-8)
    back = properties_rho_u(state.rho_kg_m3, state.u_j_kg, state.p_pa, state.T_k)
    assert back.T_k == pytest.approx(state.T_k, abs=1e-4)
    finite = properties_pt(15.7e6, 580.0)
    shifted = properties_pt(15.7e6, 580.1)
    assert (shifted.h_j_kg - finite.h_j_kg) / 0.1 == pytest.approx(finite.cp_j_kg_k, rel=2e-3)
    assert finite.h_j_kg - finite.u_j_kg - finite.p_pa * finite.v_m3_kg == pytest.approx(
        0.0, abs=1e-6
    )
    assert viscosity_pa_s(998.0, 298.15) == pytest.approx(8.897351001498108e-4, rel=1e-6)
    region3 = properties_pt(40.0e6, 700.0)
    assert region3.region == 3
    assert region3.h_j_kg > 0.0
    with pytest.raises(DomainError):
        properties_pt(60.0e6, 1500.0)


def test_two_phase_void_and_energy_identity():
    liquid, vapor = saturation_properties(6.4e6)
    mix = properties_ph(6.4e6, 0.5 * (liquid.h_j_kg + vapor.h_j_kg))
    alpha = void_fraction(mix.quality, liquid.rho_kg_m3, vapor.rho_kg_m3)
    assert mix.region == 4
    assert 0.0 < alpha < 1.0
    assert abs(alpha - mix.quality) > 0.2


def test_other_coolants_are_not_water():
    assert sodium_density_kg_m3(700.0) == pytest.approx(1016.0 - 0.235 * 700.0)
    with pytest.raises(DomainError):
        sodium_density_kg_m3(300.0)
    pressure = 5.0e6
    temperature = 800.0
    expected = pressure * 0.004002602 / (8.314462618 * temperature)
    assert helium_density_kg_m3(pressure, temperature) == pytest.approx(expected, rel=1e-12)
    conductivity = uo2_conductivity_w_m_k(1000.0)
    assert 3.0 < conductivity < 4.0


def test_point_kinetics_matches_matrix_exponential_and_period():
    kinetics = PointKinetics.thermal_u235()
    state = kinetics.initial_state(1.0)
    critical = kinetics.reference_step(state, 0.0, 2.0)
    assert critical[0] == pytest.approx(1.0, rel=1e-8)
    assert np.all(critical[1:] > 0.0)
    reactivity = 0.001
    exact = kinetics.reference_step(state, reactivity, 1.0)
    numerical = kinetics.integrate(state, reactivity, 1.0)
    assert numerical[0] == pytest.approx(exact[0], rel=1e-4)
    omega = inhour_root(kinetics, reactivity)
    late = kinetics.reference_step(state, reactivity, 200.0)
    later = kinetics.reference_step(state, reactivity, 200.5)
    period = 0.5 / math.log(later[0] / late[0])
    assert period == pytest.approx(1.0 / omega, rel=2e-2)
    source = 1.0
    subcritical = -0.01
    equilibrium_n = -source * kinetics.generation_time_s / subcritical
    started = np.concatenate([[equilibrium_n], kinetics.equilibrium_precursors(equilibrium_n)])
    held = kinetics.integrate(started, subcritical, 1.0, source=source)
    assert held[0] == pytest.approx(equilibrium_n, rel=1e-3)
    amplitudes = np.array([1.0, 1.2])
    precursors = np.vstack(
        [kinetics.equilibrium_precursors(1.0), kinetics.equilibrium_precursors(1.2)]
    )
    rho = np.array([0.0, 0.001])
    dn, dc_dt = batch_derivatives(amplitudes, precursors, rho, kinetics)
    scalar = kinetics.derivatives(1.2, precursors[1], 0.001)
    assert dn[1] == pytest.approx(scalar[0], rel=1e-12)
    assert dc_dt[1, 0] == pytest.approx(scalar[1], rel=1e-12)


def test_diffusion_converges_and_rod_is_continuous():
    height = 1.0
    diffusion = 0.01
    sigma_a = 1.0
    buckling = (math.pi / height) ** 2
    nu_sigma_f = sigma_a + diffusion * buckling
    expected = analytical_slab_k(diffusion, sigma_a, nu_sigma_f, height)
    assert expected == pytest.approx(1.0, rel=1e-12)
    errors = []
    for cells in (20, 40, 80):
        mesh = Mesh1D.uniform(height, cells)
        removal = diffusion_operator(mesh, np.full(cells, diffusion), np.full(cells, sigma_a))
        fission = fission_operator(mesh, np.full(cells, nu_sigma_f))
        k_value, _phi, converged = k_eigenvalue(removal, fission)
        assert converged
        errors.append(abs(k_value - expected))
    assert errors[1] < errors[0]
    assert errors[2] < errors[1]
    assert errors[2] < 5e-3
    base = np.ones(10)
    coarse = rod_absorption(base, 0.50, 2.0)
    finer = rod_absorption(base, 0.51, 2.0)
    assert finer[5] - coarse[5] == pytest.approx(0.1 * 2.0)
    sigma_a_g = np.array([0.01, 0.1])
    scatter = 0.05
    nu_f = np.array([0.0, 0.2])
    chi = np.array([1.0, 0.0])
    analytical = two_group_infinite_k(sigma_a_g, scatter, nu_f, chi)
    removal_matrix = np.array([[sigma_a_g[0] + scatter, 0.0], [-scatter, sigma_a_g[1]]])
    fission_matrix = np.array(
        [[chi[0] * nu_f[0], chi[0] * nu_f[1]], [chi[1] * nu_f[0], chi[1] * nu_f[1]]]
    )
    k_num, _flux, ok = k_eigenvalue(removal_matrix, fission_matrix)
    assert ok
    assert k_num == pytest.approx(analytical, rel=1e-8)
    scatter_tensor = np.zeros((2, 2, 1))
    scatter_tensor[1, 0, 0] = scatter
    removal_groups = removal_with_scatter(sigma_a_g[:, None], scatter_tensor)
    assert removal_groups[0, 0] == pytest.approx(sigma_a_g[0] + scatter)
    assert removal_groups[1, 0] == pytest.approx(sigma_a_g[1])


def test_cross_section_macroscopic_and_domain():
    provider = open_two_group_provider()
    cold = provider.at_temperature(600.0)
    mid = provider.at_temperature(900.0)
    assert np.all(mid.sigma_a_per_m <= cold.sigma_a_per_m + 1e-12)
    with pytest.raises(DomainError):
        provider.at_temperature(500.0)
    macro = micro_to_macro({"U235": 1.0e28}, {"U235": {"sigma_a": np.array([10.0, 100.0])}})
    assert macro["sigma_a"][1] == pytest.approx(1.0e28 * barns_to_m2(100.0))


def test_fuel_rod_matches_cylinder():
    rod = FuelRod(
        r_fuel_m=0.004,
        r_clad_inner_m=0.0041,
        r_clad_outer_m=0.00475,
        n_fuel=32,
        n_clad=4,
        length_m=1.0,
        n_rods=1,
        rho_fuel=10000.0,
        cp_fuel=300.0,
        rho_clad=6500.0,
        cp_clad=300.0,
        k_fuel=3.0,
        k_clad=18.0,
        h_gap=1.0e5,
        h_coolant=2.0e4,
    )
    volumetric = 5.0e8
    surface = 600.0
    numerical = rod.steady_fuel(volumetric, surface)
    r_in, r_out = rod.fuel_cell_bounds()
    for index in (0, 15, 31):
        expected = cell_average_analytical(
            r_in[index], r_out[index], rod.r_fuel_m, volumetric, rod.k_fuel, surface
        )
        assert numerical[index] == pytest.approx(expected, rel=0.03, abs=2.0)
    center = analytical_cylinder_temperature(0.0, rod.r_fuel_m, volumetric, rod.k_fuel, surface)
    assert numerical[0] == pytest.approx(center, rel=0.05)
    assert gap_conductance(0.3, 1.0e-4, 0.6, 900.0, 620.0) > 0.3 / 1.0e-4
    with pytest.raises(DomainError):
        gap_conductance(0.3, 0.0, 0.6, 900.0, 620.0)


def test_closed_volume_energy_and_flow_reversal():
    volume = Volume("tank", 1.0, 0.0)
    volume.set_liquid(15.7e6, 560.0)
    volume.heat_w = 2.0e6
    network = FlowNetwork({"tank": volume}, {})
    mass0 = network.total_mass()
    energy0 = network.total_energy()
    result = integrate(
        lambda _t, y: network.derivatives(y), network.pack(), (0.0, 1.0), rtol=1e-6, atol=1e-4
    )
    network.unpack(result.final)
    assert network.total_mass() == pytest.approx(mass0, rel=1e-8)
    assert network.total_energy() - energy0 == pytest.approx(2.0e6, rel=1e-4)

    lower = Volume("lower", 2.0, 0.0)
    upper = Volume("upper", 2.0, 10.0)
    lower.set_liquid(10.0e6, 500.0)
    density = lower.thermo.rho_kg_m3
    upper.set_liquid(10.0e6 - density * 9.80665 * 10.0, 500.0)
    link = Junction("riser", "lower", "upper", 0.05, 10.0, 0.0, mass_flow_kg_s=0.0)
    column = FlowNetwork({"lower": lower, "upper": upper}, {"riser": link})
    derivative = column.derivatives(column.pack())
    assert derivative[-1] == pytest.approx(0.0, abs=5.0)

    left = BoundaryState(1.2e6, 5.0e5, 900.0, 0.0)
    right = BoundaryState(1.0e6, 5.0e5, 900.0, 0.0)
    pipe = Junction("pipe", "boundary:left", "boundary:right", 0.01, 5.0, 2.0, mass_flow_kg_s=1.0)
    forced = FlowNetwork({}, {"pipe": pipe}, {"boundary:left": left, "boundary:right": right})
    steady = integrate(
        lambda _t, y: forced.derivatives(y), forced.pack(), (0.0, 2.0), rtol=1e-6, atol=1e-6
    ).final
    analytical = 0.01 * math.sqrt(2.0 * 900.0 * 0.2e6 / 2.0)
    assert steady[-1] == pytest.approx(analytical, rel=1e-3)
    reversed_pipe = Junction(
        "pipe", "boundary:left", "boundary:right", 0.01, 5.0, 2.0, mass_flow_kg_s=0.0
    )
    swapped = FlowNetwork(
        {},
        {"pipe": reversed_pipe},
        {
            "boundary:left": BoundaryState(1.0e6, 5.0e5, 900.0, 0.0),
            "boundary:right": BoundaryState(1.2e6, 5.0e5, 900.0, 0.0),
        },
    )
    reversed_flow = integrate(
        lambda _t, y: swapped.derivatives(y), swapped.pack(), (0.0, 2.0), rtol=1e-6, atol=1e-6
    ).final
    assert reversed_flow[-1] < 0.0


def test_boron_and_boiling_conserve_mass():
    donor = Volume("a", 1.0, 0.0)
    receiver = Volume("b", 1.0, 0.0)
    donor.set_liquid(5.0e6, 500.0, boron_ppm=1000.0)
    receiver.set_liquid(5.0e6, 500.0, boron_ppm=0.0)
    junction = Junction("link", "a", "b", 0.05, 2.0, 0.0, mass_flow_kg_s=20.0)
    loop = FlowNetwork({"a": donor, "b": receiver}, {"link": junction})
    boron0 = loop.total_boron()
    mass0 = loop.total_mass()
    derivative = loop.derivatives(loop.pack())
    assert derivative[2] + derivative[5] == pytest.approx(0.0, abs=1e-8)
    assert derivative[0] + derivative[3] == pytest.approx(0.0, abs=1e-6)
    loop.unpack(loop.pack() + 0.01 * derivative)
    assert loop.total_boron() == pytest.approx(boron0, rel=1e-6)
    assert loop.total_mass() == pytest.approx(mass0, rel=1e-8)

    boiler = Volume("boiler", 0.2, 0.0)
    boiler.set_quality(6.0e6, 0.05)
    quality_0 = boiler.thermo.quality
    boiler.heat_w = 1.0e6
    vessel = FlowNetwork({"boiler": boiler}, {})
    mass_b = vessel.total_mass()
    energy_b = vessel.total_energy()
    result = integrate(
        lambda _t, y: vessel.derivatives(y), vessel.pack(), (0.0, 1.0), rtol=1e-6, atol=1e-3
    )
    vessel.unpack(result.final)
    vessel.refresh_properties()
    assert vessel.total_mass() == pytest.approx(mass_b, rel=1e-6)
    assert vessel.total_energy() - energy_b == pytest.approx(1.0e6, rel=2e-3)
    assert vessel.volumes["boiler"].thermo.region == 4
    assert vessel.volumes["boiler"].thermo.quality > quality_0


def test_xenon_decay_and_chain_conservation():
    xenon = XenonIodine()
    fission_rate = 1.0e18
    flux = 1.0e17
    equilibrium = xenon.equilibrium(fission_rate, flux)
    assert np.allclose(
        xenon.derivatives(equilibrium[0], equilibrium[1], fission_rate, flux), 0.0, atol=1.0e3
    )
    exact = xenon.reference(np.zeros(2), fission_rate, flux, 1.0e4)
    numerical = xenon.integrate(np.zeros(2), fission_rate, flux, 1.0e4)
    assert numerical == pytest.approx(exact, rel=1e-4)
    daughter = bateman_daughter(1.0, 0.1, 0.02, 10.0)
    matrix = decay_chain_matrix(np.array([0.1, 0.02, 0.0]))
    evolved = advance_nuclides(np.array([1.0, 0.0, 0.0]), matrix, 10.0)
    assert evolved[1] == pytest.approx(daughter, rel=1e-8)
    assert evolved.sum() == pytest.approx(1.0, rel=1e-10)
    heat = DecayHeat.illustrative()
    stored = heat.equilibrium(1.0e9)
    assert heat.total_power_w(1.0e9, stored) == pytest.approx(1.0e9, rel=1e-10)
    after = heat.integrate(stored, 0.0, 50.0)
    assert heat.decay_power_w(after) < heat.decay_power_w(stored)
    again = heat.integrate(stored, 0.0, 50.0)
    assert stored.sum() - after.sum() == pytest.approx(stored.sum() - again.sum(), rel=1e-10)
    matrix_2 = np.array([[-0.2, 0.0], [0.2, -0.05]])
    state = expm(matrix_2 * 3.0) @ np.array([2.0, 0.0])
    assert advance_nuclides(np.array([2.0, 0.0]), matrix_2, 3.0) == pytest.approx(state)


def test_nusselt_orifice_protection_and_mapping():
    nu = nusselt_dittus_boelter(1.0e5, 1.0)
    assert nu.value == pytest.approx(0.023 * 1.0e5**0.8, rel=1e-12)
    with pytest.raises(DomainError):
        nusselt_dittus_boelter(1000.0, 1.0)
    coefficient = heat_transfer_coefficient(10.0, 0.01, 1.0e-3, 4000.0, 0.6)
    assert coefficient.value > 0.0
    flow = orifice_mass_flow(0.62, 0.01, 1000.0, 2.0e5)
    assert flow == pytest.approx(0.62 * 0.01 * math.sqrt(2.0 * 1000.0 * 2.0e5))
    assert coastdown_omega(100.0, 20.0, 10.0, 5.0) == pytest.approx(90.0)
    logic = ProtectionLogic(setpoint=10.0, delay_s=0.2)
    assert logic.step([11.0, 9.0, 8.0], 1.0) is False
    assert logic.step([11.0, 12.0, 8.0], 0.1) is False
    assert logic.step([11.0, 12.0, 8.0], 0.1) is True
    pid = PID(0.5, 0.1, 0.0, output_limit=0.2)
    first = pid.step(10.0, 0.1)
    frozen = pid.integral
    pid.step(10.0, 0.1)
    assert pid.integral == pytest.approx(frozen)
    assert abs(first) <= 0.2
    sensor = Sensor(tau_s=2.0, value=0.0)
    assert sensor.step(1.0, 2.0, np.random.default_rng(0)) == pytest.approx(
        1.0 - math.exp(-1.0), rel=1e-6
    )
    actuator = Actuator(0.0, speed_limit=0.1)
    assert actuator.step(1.0, 0.5) == pytest.approx(0.05)
    mapped = conserve_integral(
        np.array([1.0, 3.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5, 1.0])
    )
    assert np.sum(mapped * np.array([0.5, 0.5, 1.0])) == pytest.approx(4.0)
    report = check_topology(["a"], [("j", "a", "boundary:out")], {})
    assert not report.ok
    assert differential_mask([EquationSpec("m", True), EquationSpec("p", False)]) == [True, False]

    def event(_time, y):
        return y[0] - 0.3

    found = collect_sign_changes(
        [EventSpec("trip", event, direction=1.0)], 0.0, np.array([0.0]), 1.0, np.array([1.0])
    )
    assert found[0].time_s == pytest.approx(0.3, abs=1e-6)

    def fail(_payload):
        raise RuntimeError("down")

    result = ExternalSolverAdapter(fail, "test-1").exchange({"x": 1})
    assert result["applicability"] == Applicability.UNSUPPORTED.value
    assert substep_count(3.2, 1.0) == 4
    solved = integrate(lambda _t, y: -y, np.array([1.0]), (0.0, 1.0), method="Radau")
    assert solved.final[0] == pytest.approx(math.exp(-1.0), rel=1e-5)
    assert picard_power(lambda value: 1.0e9 - value, 1.0e9) == pytest.approx(1.0e9, rel=1e-4)


def test_rbmk_is_separate_and_unsupported_types_stay_unsupported():
    channel = RBMKChannel()
    channel.steady_temperatures()
    d_fuel, d_graphite, alpha, reactivity = channel.derivatives()
    assert d_fuel == pytest.approx(0.0, abs=1e-6)
    assert d_graphite == pytest.approx(0.0, abs=1e-6)
    assert reactivity == pytest.approx(channel.void_coefficient * (alpha - channel.alpha_ref))
    assert channel.void_coefficient > 0.0
    with pytest.raises(DomainError) as caught:
        build_plant("unknown-reactor")
    assert caught.value.applicability == Applicability.UNSUPPORTED
    plant = build_plant("vver")
    assert plant.passport.plant_referenced is False
    assert plant.passport.status.value == "verified"
