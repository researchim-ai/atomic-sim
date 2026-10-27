"""Подогреваемый канал из контрольных объёмов воды.

Импульс соединений — тот же закон сети. Этот модуль задаёт цепочку и проверяет,
что подогрев, вход и выход согласованы с dU/dt. Критическое истечение кипящей
воды здесь не добавляется.
"""

from __future__ import annotations

import numpy as np

from core.status import DomainError, ModelPassport, ModelStatus
from physics.hydraulics.network import BoundaryState, FlowNetwork, Junction, Volume
from physics.materials.iapws import properties_pt

PASSPORT = ModelPassport(
    model_id="heated-water-channel",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Цепочка объёмов воды с заданным расходом и подогревом",
    approximations=(
        "расход по соединениям задан и одинаков",
        "трение не пересчитывает расход в этой проверке баланса",
        "кипение допускается, пока обращение (p, h) внутри IF97",
    ),
    applicability="Положительный расход, ненулевые объёмы, свойства воды по IF97",
    data_sources=("Баланс энергии сети контрольных объёмов",),
)


def heated_water_channel(
    n_cells: int,
    volume_m3: float,
    pressure_pa: float,
    inlet_temperature_k: float,
    mass_flow_kg_s: float,
    heat_w: float,
) -> FlowNetwork:
    if n_cells < 1 or volume_m3 <= 0.0 or mass_flow_kg_s == 0.0:
        raise DomainError("Канал требует ячейки, объём и ненулевой расход")
    inlet = properties_pt(pressure_pa, inlet_temperature_k)
    volumes = {}
    for index in range(n_cells):
        volume = Volume("c%d" % index, volume_m3, 0.0)
        volume.set_liquid(pressure_pa, inlet_temperature_k)
        volume.heat_w = heat_w
        volumes[volume.name] = volume
    junctions = {
        "inlet": Junction(
            "inlet", "boundary:inlet", "c0", 0.05, 0.2, 0.0, mass_flow_kg_s=mass_flow_kg_s
        )
    }
    for index in range(n_cells - 1):
        name = "j%d" % index
        junctions[name] = Junction(
            name, "c%d" % index, "c%d" % (index + 1), 0.05, 0.5, 0.2, mass_flow_kg_s=mass_flow_kg_s
        )
    junctions["outlet"] = Junction(
        "outlet",
        "c%d" % (n_cells - 1),
        "boundary:outlet",
        0.05,
        0.2,
        0.0,
        mass_flow_kg_s=mass_flow_kg_s,
    )
    network = FlowNetwork(volumes, junctions)
    network.boundaries["boundary:inlet"] = BoundaryState(
        pressure_pa, inlet.h_j_kg, inlet.rho_kg_m3, 0.0
    )
    network.boundaries["boundary:outlet"] = BoundaryState(
        pressure_pa, inlet.h_j_kg, inlet.rho_kg_m3, 0.0
    )
    return network


def channel_energy_closure(network: FlowNetwork) -> tuple[float, float]:
    """Сумма dU и внешний баланс m h_in - m h_out + Q."""
    derivative = network.derivatives(network.pack())
    stored = 0.0
    for index, _name in enumerate(network.volume_names()):
        stored += float(derivative[3 * index + 1])
    inlet = network.junctions["inlet"]
    outlet = network.junctions["outlet"]
    last = network.volumes[outlet.upstream]
    supplied = (
        inlet.mass_flow_kg_s * network.boundaries["boundary:inlet"].enthalpy_j_kg
        - outlet.mass_flow_kg_s * last.thermo.h_j_kg
        + float(sum(volume.heat_w for volume in network.volumes.values()))
    )
    return stored, supplied
