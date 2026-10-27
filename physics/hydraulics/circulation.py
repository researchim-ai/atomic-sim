"""Замкнутая петля естественной циркуляции. В паспорт ВВЭР она не входит.

Напор (rho_cold - rho_hot) g H уравновешивает трение нижнего участка.
Подогрев и охлаждение держат энтальпии ног; это не модель кипения.
"""

from __future__ import annotations

from core.status import DomainError, ModelPassport, ModelStatus
from physics.hydraulics.network import G, FlowNetwork, Junction, Volume
from physics.materials.iapws import properties_pt

PASSPORT = ModelPassport(
    model_id="natural-circulation-loop",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Замкнутый контур с горячей подъёмной и холодной опускной ветвью",
    approximations=(
        "четыре сосредоточенных объёма",
        "трение собрано на нижнем горизонтальном участке",
        "температуры ног поддерживаются внешним подогревом и охлаждением",
    ),
    applicability="Обе ноги — сжатая жидкость IF97; выравнивание плотностей гасит напор",
    data_sources=("Гидростатика rho g H и квадратичное трение отверстия",),
)


def _riser_bottom_pressure(top_pressure_pa: float, temperature_k: float, height_m: float) -> float:
    """Донор подъёмной ноги — нижний объём, поэтому плотность берётся при нижнем давлении."""
    pressure = top_pressure_pa
    for _ in range(12):
        density = properties_pt(pressure, temperature_k).rho_kg_m3
        updated = top_pressure_pa + density * G * height_m
        if abs(updated - pressure) < 0.01:
            return updated
        pressure = updated
    raise DomainError("Гидростатика подъёмной ноги не сошлась")


def natural_circulation_loop(
    top_pressure_pa: float = 10.0e6,
    cold_temperature_k: float = 450.0,
    hot_temperature_k: float = 560.0,
    height_m: float = 8.0,
    mass_flow_kg_s: float = 25.0,
    area_m2: float = 0.05,
) -> FlowNetwork:
    if hot_temperature_k <= cold_temperature_k:
        raise DomainError("Естественная циркуляция этой схемы требует горячую ногу легче холодной")
    if mass_flow_kg_s <= 0.0 or height_m <= 0.0 or area_m2 <= 0.0:
        raise DomainError("Расход, высота и сечение петли должны быть положительными")
    cold_top_density = properties_pt(top_pressure_pa, cold_temperature_k).rho_kg_m3
    cold_bottom_pressure = top_pressure_pa + cold_top_density * G * height_m
    hot_bottom_pressure = _riser_bottom_pressure(top_pressure_pa, hot_temperature_k, height_m)
    cold_bottom = properties_pt(cold_bottom_pressure, cold_temperature_k)
    hot_bottom = properties_pt(hot_bottom_pressure, hot_temperature_k)
    cold_top = properties_pt(top_pressure_pa, cold_temperature_k)
    hot_top = properties_pt(top_pressure_pa, hot_temperature_k)
    driving_pa = cold_bottom_pressure - hot_bottom_pressure
    if driving_pa <= 0.0:
        raise DomainError("Напор естественной циркуляции не положительный")
    loss = driving_pa * 2.0 * cold_bottom.rho_kg_m3 * area_m2**2 / mass_flow_kg_s**2

    def volume(name: str, elevation: float, pressure: float, temperature: float, heat_w: float) -> Volume:
        node = Volume(name, 1.0, elevation)
        node.set_liquid(pressure, temperature)
        node.heat_w = heat_w
        return node

    heat_hot_bottom = mass_flow_kg_s * (hot_bottom.h_j_kg - cold_bottom.h_j_kg)
    heat_hot_top = mass_flow_kg_s * (hot_top.h_j_kg - hot_bottom.h_j_kg)
    heat_cold_top = mass_flow_kg_s * (cold_top.h_j_kg - hot_top.h_j_kg)
    heat_cold_bottom = mass_flow_kg_s * (cold_bottom.h_j_kg - cold_top.h_j_kg)
    volumes = {
        "cold_bot": volume("cold_bot", 0.0, cold_bottom_pressure, cold_temperature_k, heat_cold_bottom),
        "hot_bot": volume("hot_bot", 0.0, hot_bottom_pressure, hot_temperature_k, heat_hot_bottom),
        "hot_top": volume("hot_top", height_m, top_pressure_pa, hot_temperature_k, heat_hot_top),
        "cold_top": volume("cold_top", height_m, top_pressure_pa, cold_temperature_k, heat_cold_top),
    }
    junctions = {
        "bottom": Junction("bottom", "cold_bot", "hot_bot", area_m2, 2.0, loss, mass_flow_kg_s),
        "riser": Junction("riser", "hot_bot", "hot_top", area_m2, height_m, 0.0, mass_flow_kg_s),
        "top": Junction("top", "hot_top", "cold_top", area_m2, 2.0, 0.0, mass_flow_kg_s),
        "downcomer": Junction("downcomer", "cold_top", "cold_bot", area_m2, height_m, 0.0, mass_flow_kg_s),
    }
    return FlowNetwork(volumes, junctions)


def loop_residuals(network: FlowNetwork) -> dict:
    derivative = network.derivatives(network.pack())
    n_volumes = len(network.volumes)
    mass = [float(derivative[3 * index]) for index in range(n_volumes)]
    energy = [float(derivative[3 * index + 1]) for index in range(n_volumes)]
    flows = [float(value) for value in derivative[3 * n_volumes :]]
    return {"mass": mass, "energy": energy, "flow_acceleration": flows}
