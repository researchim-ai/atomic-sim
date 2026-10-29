"""Хвост сценария, равные теплоёмкости, конденсация, гелий и распад предшественников."""

import numpy as np
import pytest

from components.heat_exchanger import equal_capacity_counterflow, lmtd_heat_w
from core.status import DomainError
from physics.hydraulics.closures import poiseuille_drop_pa, series_poiseuille_drop_pa
from physics.hydraulics.two_phase_ext import condensation_onset_length_m
from physics.materials.coolants import (
    helium_density_kg_m3,
    helium_specific_gas_constant_j_kg_k,
    lead_internal_energy_j_kg,
    lead_specific_heat_j_kg_k,
)
from physics.neutronics.point_kinetics import PointKinetics
from plants.rbmk import RBMKChannel
from scenarios.session import step_lengths


def test_scenario_tail_counterflow_identity_and_desuperheat():
    exact = step_lengths(0.4, 0.2)
    assert exact == pytest.approx([0.2, 0.2])
    tail = step_lengths(1.0, 0.4)
    assert tail == pytest.approx([0.4, 0.4, 0.2])
    assert sum(tail) == pytest.approx(1.0)
    with pytest.raises(DomainError):
        step_lengths(1.0, 6.0)

    capacity = 1000.0
    difference = 80.0
    ntu = 2.0
    heat, lmtd = equal_capacity_counterflow(capacity, difference, ntu)
    assert heat == pytest.approx(capacity * difference * ntu / (1.0 + ntu))
    assert lmtd == pytest.approx(difference / (1.0 + ntu))
    assert lmtd_heat_w(ntu * capacity, lmtd, lmtd) == pytest.approx(heat)

    onset = condensation_onset_length_m(1.5, 2080.0, 600.0, 550.0, 1.0e5)
    assert onset == pytest.approx(1.5 * 2080.0 * 50.0 / 1.0e5)
    assert condensation_onset_length_m(1.5, 2080.0, 550.0, 550.0, 1.0e5) == pytest.approx(0.0)
    with pytest.raises(DomainError):
        condensation_onset_length_m(1.5, 2080.0, 500.0, 550.0, 1.0e5)


def test_precursor_decay_rbmk_split_helium_lead_and_series_pipe():
    kinetics = PointKinetics.thermal_u235()
    start = np.array([1.0, 2.0, 0.5, 0.2, 0.1, 0.05])
    decayed = kinetics.decayed_precursors(start, 3.0)
    assert decayed == pytest.approx(start * np.exp(-kinetics.decay * 3.0))
    step = 1.0e-8
    rate = (kinetics.decayed_precursors(start, step) - start) / step
    assert rate == pytest.approx(kinetics.derivatives(0.0, start, -0.01)[1:], rel=1.0e-5)

    channel = RBMKChannel()
    fuel, graphite = channel.heat_split_w()
    assert fuel + graphite == pytest.approx(channel.fission_power_w)
    with pytest.raises(DomainError):
        RBMKChannel(graphite_fraction=1.2)

    pressure, temperature = 6.0e6, 800.0
    density = helium_density_kg_m3(pressure, temperature)
    assert density * helium_specific_gas_constant_j_kg_k() * temperature == pytest.approx(pressure)

    center = 900.0
    delta = 1.0e-3
    slope = (
        lead_internal_energy_j_kg(center + delta) - lead_internal_energy_j_kg(center - delta)
    ) / (2.0 * delta)
    assert slope == pytest.approx(lead_specific_heat_j_kg_k(center), rel=1.0e-6)

    one = poiseuille_drop_pa(0.004, 0.01, 1.0e-3, 3.0, 800.0)
    parts = series_poiseuille_drop_pa(0.004, 0.01, 1.0e-3, [1.0, 2.0], 800.0)
    assert parts == pytest.approx(one)
