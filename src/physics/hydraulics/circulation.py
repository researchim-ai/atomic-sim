"""Замкнутая петля естественной циркуляции. В паспорт ВВЭР она не входит.

Напор (rho_cold - rho_hot) g H уравновешивает трение нижнего участка.
Отдельная сборка thermal-circulation-loop хранит энергию в объёмах:
нагреватель постоянен, холодильник — UA на сток, насос можно остановить.
"""

from __future__ import annotations

import numpy as np

from core.status import DomainError, ModelPassport, ModelStatus
from physics.hydraulics.network import FlowNetwork, G, Junction, Volume
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


def buoyancy_head_pa(rho_cold_kg_m3: float, rho_hot_kg_m3: float, height_m: float) -> float:
    """Напор двух несжимаемых столбов одной высоты: (ρ_хол − ρ_гор) g H.

    Это не перепад петли ВВЭР: там отметки объёмов совпадают.
    """
    if rho_cold_kg_m3 <= 0.0 or rho_hot_kg_m3 <= 0.0 or height_m < 0.0:
        raise DomainError("Плотности столбов положительны, высота неотрицательна")
    return (rho_cold_kg_m3 - rho_hot_kg_m3) * G * height_m


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

    def volume(
        name: str, elevation: float, pressure: float, temperature: float, heat_w: float
    ) -> Volume:
        node = Volume(name, 1.0, elevation)
        node.set_liquid(pressure, temperature)
        node.heat_w = heat_w
        return node

    heat_hot_bottom = mass_flow_kg_s * (hot_bottom.h_j_kg - cold_bottom.h_j_kg)
    heat_hot_top = mass_flow_kg_s * (hot_top.h_j_kg - hot_bottom.h_j_kg)
    heat_cold_top = mass_flow_kg_s * (cold_top.h_j_kg - hot_top.h_j_kg)
    heat_cold_bottom = mass_flow_kg_s * (cold_bottom.h_j_kg - cold_top.h_j_kg)
    volumes = {
        "cold_bot": volume(
            "cold_bot", 0.0, cold_bottom_pressure, cold_temperature_k, heat_cold_bottom
        ),
        "hot_bot": volume("hot_bot", 0.0, hot_bottom_pressure, hot_temperature_k, heat_hot_bottom),
        "hot_top": volume("hot_top", height_m, top_pressure_pa, hot_temperature_k, heat_hot_top),
        "cold_top": volume(
            "cold_top", height_m, top_pressure_pa, cold_temperature_k, heat_cold_top
        ),
    }
    junctions = {
        "bottom": Junction("bottom", "cold_bot", "hot_bot", area_m2, 2.0, loss, mass_flow_kg_s),
        "riser": Junction("riser", "hot_bot", "hot_top", area_m2, height_m, 0.0, mass_flow_kg_s),
        "top": Junction("top", "hot_top", "cold_top", area_m2, 2.0, 0.0, mass_flow_kg_s),
        "downcomer": Junction(
            "downcomer", "cold_top", "cold_bot", area_m2, height_m, 0.0, mass_flow_kg_s
        ),
    }
    return FlowNetwork(volumes, junctions)


def loop_residuals(network: FlowNetwork) -> dict:
    derivative = network.derivatives(network.pack())
    n_volumes = len(network.volumes)
    mass = [float(derivative[3 * index]) for index in range(n_volumes)]
    energy = [float(derivative[3 * index + 1]) for index in range(n_volumes)]
    flows = [float(value) for value in derivative[3 * n_volumes :]]
    return {"mass": mass, "energy": energy, "flow_acceleration": flows}


THERMAL_PASSPORT = ModelPassport(
    model_id="thermal-circulation-loop",
    version="1.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Совместные запас энергии, насос и теплоотвод замкнутой петли",
    approximations=(
        "четыре объёма воды, их внутренняя энергия и есть тепловой запас",
        "нагреватель постоянный, холодильник UA на фиксированный сток",
        "мелкие гидростатические поправки энтальпии зафиксированы в номинале",
        "напор насоса масштабируется квадратом скорости и не входит в петлю ВВЭР",
    ),
    applicability="Сжатая жидкость IF97, прямой расход через насос, скорость вала выше нуля до останова",
    data_sources=("IF97, гидростатика и квадратичное трение этой петли",),
    notes=(
        "verified — баланс энергии и гидравлики модели. Это не выбег блока и не plant_referenced.",
    ),
)


class ThermalCirculation:
    """Насосный номинал на той же геометрии. После останова расход держит гидростатика."""

    passport = THERMAL_PASSPORT

    def __init__(self) -> None:
        self.network = natural_circulation_loop(
            top_pressure_pa=12.0e6,
            cold_temperature_k=430.0,
            hot_temperature_k=500.0,
        )
        hot_bottom = self.network.volumes["hot_bot"]
        cold_top = self.network.volumes["cold_top"]
        self.passive_heat_w = {
            "hot_top": self.network.volumes["hot_top"].heat_w,
            "cold_bot": self.network.volumes["cold_bot"].heat_w,
        }
        self.sink_temperature_k = cold_top.temperature_k - 15.0
        design_cool_w = cold_top.heat_w
        self.ua_cooler_w_k = design_cool_w / (self.sink_temperature_k - cold_top.temperature_k)
        bottom = self.network.junctions["bottom"]
        density = self.network.volumes["cold_bot"].thermo.rho_kg_m3
        friction = bottom.friction_dp(density)
        self.dp_nom_pa = 0.5 * friction
        bottom.loss_coefficient *= (friction + self.dp_nom_pa) / friction
        self.omega_nom = 150.0
        self.omega = self.omega_nom
        self.motor_on = True
        self.inertia = 0.02
        self.time_s = 0.0
        pump_work = bottom.mass_flow_kg_s * self.dp_nom_pa / density
        self.heater_w = hot_bottom.heat_w - pump_work
        torque = pump_work / self.omega_nom
        self.loss_coef = 0.05 * torque / self.omega_nom
        self.drive_torque = torque + self.loss_coef * self.omega_nom
        self.last_energy_residual_j = None
        steady = self.derivatives(0.0, self.pack())
        energies = [float(steady[3 * index + 1]) for index in range(4)]
        flows = [float(value) for value in steady[12:16]]
        if abs(sum(energies)) > 5.0 or max(abs(value) for value in flows) > 1.0e-3:
            raise DomainError("Номинал тепловой петли не уравновешен")
        if abs(float(steady[-1])) > 1.0e-6:
            raise DomainError("Вал тепловой петли не удерживается в номинале")

    def pack(self) -> np.ndarray:
        return np.append(self.network.pack(), self.omega)

    def unpack(self, state: np.ndarray) -> None:
        self.network.unpack(state[:-1])
        self.omega = float(state[-1])

    def _pump_dp(self) -> float:
        if self.omega <= 0.0:
            return 0.0
        ratio = self.omega / self.omega_nom
        return self.dp_nom_pa * ratio * ratio

    def _apply_boundary(self) -> float:
        self.network.refresh_properties()
        self.network.volumes["hot_bot"].heat_w = self.heater_w
        self.network.volumes["hot_top"].heat_w = self.passive_heat_w["hot_top"]
        self.network.volumes["cold_bot"].heat_w = self.passive_heat_w["cold_bot"]
        cold = self.network.volumes["cold_top"]
        cold.heat_w = self.ua_cooler_w_k * (self.sink_temperature_k - cold.temperature_k)
        bottom = self.network.junctions["bottom"]
        bottom.pump_pressure_pa = self._pump_dp()
        density = self.network.volumes["cold_bot"].thermo.rho_kg_m3
        if bottom.mass_flow_kg_s <= 0.0 or bottom.pump_pressure_pa <= 0.0 or density <= 0.0:
            return 0.0
        return bottom.mass_flow_kg_s * bottom.pump_pressure_pa / density

    def _shaft_rate(self, pump_work_w: float) -> float:
        if self.omega <= 1.0:
            torque_hyd = 0.0
        else:
            torque_hyd = pump_work_w / self.omega
        drive = self.drive_torque if self.motor_on else 0.0
        return (drive - torque_hyd - self.loss_coef * self.omega) / self.inertia

    def derivatives(self, _time: float, state: np.ndarray) -> np.ndarray:
        self.unpack(state)
        pump_work = self._apply_boundary()
        net_dot = self.network.derivatives(self.network.pack())
        return np.append(net_dot, self._shaft_rate(pump_work))

    def stored_energy_j(self) -> float:
        return float(sum(volume.internal_energy_j for volume in self.network.volumes.values()))

    def boundary_power_w(self) -> float:
        pump_work = self._apply_boundary()
        heat = sum(volume.heat_w for volume in self.network.volumes.values())
        return heat + pump_work

    def _reject_if_boiling(self) -> None:
        self.network.refresh_properties()
        for volume in self.network.volumes.values():
            if volume.thermo.quality > 1.0e-4:
                raise DomainError("Петля %s вышла в пар, этот паспорт жидкий" % volume.name)

    def advance(self, dt_s: float, rtol: float = 1e-6, atol: float = 1e-6) -> dict:
        from core.solvers import SolverError, integrate

        if dt_s <= 0.0:
            raise DomainError("Шаг тепловой петли должен быть положительным")
        saved = self.pack().copy()
        saved_time = self.time_s
        try:
            t0 = self.time_s
            t1 = t0 + dt_s
            result = integrate(
                self.derivatives,
                saved,
                (t0, t1),
                rtol=rtol,
                atol=atol,
                max_step=dt_s,
                dense_output=True,
            )
            self.time_s = t1
            self.unpack(result.final)
            self._reject_if_boiling()
            times = np.linspace(t0, t1, 11)
            samples = [
                np.asarray(result.dense(float(time_s)), dtype=float).reshape(-1) for time_s in times
            ]
            self.unpack(samples[0])
            energy_0 = self.stored_energy_j()
            powers = []
            for state in samples:
                self.unpack(state)
                powers.append(self.boundary_power_w())
            self.unpack(samples[-1])
            supplied = 0.0
            for index in range(len(times) - 1):
                span = float(times[index + 1] - times[index])
                supplied += 0.5 * span * (powers[index] + powers[index + 1])
            self.last_energy_residual_j = self.stored_energy_j() - energy_0 - supplied
            self.time_s = t1
            self.unpack(result.final)
            return {
                "time_s": self.time_s,
                "mass_flow_kg_s": self.network.junctions["bottom"].mass_flow_kg_s,
                "omega": self.omega,
                "hot_temperature_k": self.network.volumes["hot_bot"].temperature_k,
                "cold_temperature_k": self.network.volumes["cold_top"].temperature_k,
            }
        except (DomainError, SolverError):
            self.unpack(saved)
            self.time_s = saved_time
            raise
