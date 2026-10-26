"""Сосредоточенная установка ВВЭР/PWR: один эквивалентный контур.

Это не четырёхпетлевая геометрическая копия блока и не plant reference.
Стационар собран так, чтобы невязки массы и энергии обращались в ноль
при номинальных параметрах модели.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.status import Applicability, DomainError, ModelPassport, ModelStatus
from physics.depletion.chains import DecayHeat
from physics.hydraulics.network import FlowNetwork, Junction, Volume
from physics.materials.iapws import properties_pt, saturation_properties
from physics.neutronics.point_kinetics import PointKinetics

PASSPORT = ModelPassport(
    model_id="vver-lumped-loop",
    version="2.0.0",
    status=ModelStatus.VERIFIED,
    phenomenon="Эквивалентная петля ВВЭР: точечная кинетика, твэл, первичный контур, ПГ, турбина, конденсатор",
    approximations=(
        "одна эквивалентная петля вместо четырёх",
        "узлы на одной отметке: движущий напор естественной циркуляции в этот паспорт не входит",
        "эффективный UA парогенератора согласован с номинальной точкой",
        "турбина — постоянный изоэнтропийный КПД и отделённый насыщенный пар",
        "обратные связи реактивности редуцированы и обнулены в номинальной точке",
    ),
    applicability="Подогрев первичного контура без объёмного кипения; вторичный контур в равновесной двухфазной области при p < 16.5 МПа",
    data_sources=(
        "IF97 для воды",
        "шестигрупповые параметры теплового U-235 как вход точечной модели",
        "номиналы 15.7 МПа, 290/320 °C и 6.4 МПа — ориентиры класса ВВЭР-1000, не паспорт блока",
    ),
    notes=(
        "verified означает замкнутые балансы и эталоны подсистем. Это не validated и не plant_referenced.",
    ),
)

WORTH = 0.008  # абсолютная интегральная эффективность, параметр модели


def rod_integral_worth(insertion: float) -> float:
    """Гладкая интегральная кривая, 0 на извлечённом упоре и 1 на нижнем."""
    z = min(1.0, max(0.0, insertion))
    return z * z * (3.0 - 2.0 * z)


def differential_worth(insertion: float) -> float:
    z = min(1.0, max(0.0, insertion))
    return 6.0 * z * (1.0 - z)


@dataclass
class PlantSnapshot:
    time_s: float
    state: np.ndarray
    scram_latched: bool
    throttle: float
    motor_on: bool


class VVERPlant:
    def __init__(self) -> None:
        self.passport = PASSPORT
        self.kinetics = PointKinetics.thermal_u235(power_nominal_w=1.0)
        self.decay = DecayHeat.illustrative()
        self.p_primary = 15.7e6
        self.t_cold = 290.0 + 273.15
        self.t_hot = 320.0 + 273.15
        self.p_secondary = 6.4e6
        self.p_condenser = 8.0e3
        self.t_sink = 293.15
        self.mass_flow_nom = 18000.0
        self.t_fuel_ref = 900.0
        self.rod_ref = 0.25
        self.boron_ppm_ref = 1000.0
        self.doppler_coef = -1.44e-3  # 1/sqrt(K), см. теорию
        self.density_coef = 1.5e-4  # dk/k на (кг/м^3)
        self.boron_coef = -8.0e-5  # dk/k на ppm
        self.throttle = 1.0
        self.motor_on = True
        self.scram_latched = False
        self.scram_speed = 0.2  # доля хода в секунду
        self.rod_command = 0.0
        self.time_s = 0.0
        self.applicability = Applicability.IN_DOMAIN
        self.message = ""
        self.network = self._build_network()
        self._nominalize()
        self.decay_state = self.decay.equilibrium(self.fission_power_nom)
        self.kinetics_state = self.kinetics.initial_state(1.0)
        self.omega = self.omega_nom
        self.rod = self.rod_ref
        self.fuel_temperature = self.t_fuel_ref

    def _build_network(self) -> FlowNetwork:
        volumes = {
            "core": Volume("core", 18.0, 0.0),
            "hot_leg": Volume("hot_leg", 8.0, 0.0),
            "sg_primary": Volume("sg_primary", 22.0, 0.0),
            "cold_leg": Volume("cold_leg", 8.0, 0.0),
            "pressurizer": Volume("pressurizer", 55.0, 0.0),
            "sg_secondary": Volume("sg_secondary", 120.0, 0.0),
            "condenser": Volume("condenser", 80.0, 0.0),
        }

        def junction(name, up, down, k_loss=0.25, area=0.8):
            return Junction(name, up, down, area, 12.0, k_loss)

        junctions = {
            "pump": junction("pump", "cold_leg", "core"),
            "core_out": junction("core_out", "core", "hot_leg"),
            "to_sg": junction("to_sg", "hot_leg", "sg_primary"),
            "to_cold": junction("to_cold", "sg_primary", "cold_leg"),
            "surge": junction("surge", "hot_leg", "pressurizer", k_loss=5.0, area=0.05),
            "steam": Junction(
                "steam",
                "sg_secondary",
                "condenser",
                0.5,
                5.0,
                0.0,
                enthalpy_mode="saturated_vapor",
                boron_carry=0.0,
                flow_law="target",
            ),
            "feed": Junction(
                "feed",
                "condenser",
                "sg_secondary",
                0.2,
                8.0,
                0.0,
                enthalpy_mode="saturated_liquid",
                flow_law="target",
            ),
        }
        return FlowNetwork(volumes, junctions)

    def _nominalize(self) -> None:
        net = self.network
        cold = properties_pt(self.p_primary, self.t_cold)
        hot = properties_pt(self.p_primary, self.t_hot)
        h_cold = cold.h_j_kg
        h_hot = hot.h_j_kg
        self.network.volumes["sg_secondary"].set_quality(self.p_secondary, 0.12)
        self.network.volumes["condenser"].set_quality(self.p_condenser, 1.0e-4)
        p_cold = p_core = p_hot = p_sg = self.p_primary
        self.dp_pump_nom = 0.0
        for _ in range(3):
            self._assign_primary_enthalpy(p_cold, p_core, p_hot, p_sg, h_cold, h_hot)
            net.volumes["pressurizer"].set_quality(p_hot, 0.35)
            net.refresh_properties()
            drops = {}
            for name in ("pump", "core_out", "to_sg", "to_cold"):
                junction = net.junctions[name]
                junction.mass_flow_kg_s = self.mass_flow_nom
                donor = net.volumes[junction.upstream]
                drops[name] = junction.friction_dp(donor.thermo.rho_kg_m3)
            self.dp_pump_nom = float(sum(drops.values()))
            p_cold = self.p_primary
            p_core = p_cold - drops["pump"] + self.dp_pump_nom
            p_hot = p_core - drops["core_out"]
            p_sg = p_hot - drops["to_sg"]
        self._assign_primary_enthalpy(p_cold, p_core, p_hot, p_sg, h_cold, h_hot)
        net.volumes["pressurizer"].set_quality(p_hot, 0.35)
        net.junctions["surge"].mass_flow_kg_s = 0.0
        net.junctions["pump"].pump_pressure_pa = self.dp_pump_nom
        net.refresh_properties()
        self.rho_ref = net.volumes["cold_leg"].thermo.rho_kg_m3
        self.h_hot = h_hot
        self.h_cold = h_cold
        self.pump_power_w = self.mass_flow_nom * self.dp_pump_nom / self.rho_ref
        self.q_transferred_w = self.mass_flow_nom * (h_hot - h_cold)
        self.fission_power_nom = self.q_transferred_w - self.pump_power_w
        self.kinetics.power_nominal_w = self.fission_power_nom
        _liquid_sg, vapor_sg = saturation_properties(self.p_secondary)
        liquid_c, _vapor_c = saturation_properties(self.p_condenser)
        self.h_g = vapor_sg.h_j_kg
        self.h_g_current = self.h_g
        self.h_feed = liquid_c.h_j_kg + liquid_c.v_m3_kg * (self.p_secondary - self.p_condenser)
        self.p_sg_nom = self.p_secondary
        self.m_steam_nom = self.q_transferred_w / (self.h_g - self.h_feed)
        self.eta_turbine = 0.85
        self.h_turbine_nom = self.turbine_outlet_enthalpy(self.p_secondary)
        steam = net.junctions["steam"]
        feed = net.junctions["feed"]
        steam.mass_flow_kg_s = self.m_steam_nom
        steam.flow_target_kg_s = self.m_steam_nom
        steam.delivery_enthalpy_j_kg = self.h_turbine_nom
        feed.mass_flow_kg_s = self.m_steam_nom
        feed.flow_target_kg_s = self.m_steam_nom
        feed.delivery_enthalpy_j_kg = self.h_feed
        t_sat = net.volumes["sg_secondary"].temperature_k
        self.ua_sg = self.q_transferred_w / (net.volumes["sg_primary"].temperature_k - t_sat)
        q_sink = self.m_steam_nom * (self.h_turbine_nom - liquid_c.h_j_kg)
        t_cond = net.volumes["condenser"].temperature_k
        self.ua_condenser = q_sink / max(t_cond - self.t_sink, 1.0)
        self.ua_fuel = self.fission_power_nom / (
            self.t_fuel_ref - net.volumes["core"].temperature_k
        )
        self.omega_nom = 157.0
        self.inertia = 2.0e4
        hyd_torque = self.pump_power_w / self.omega_nom
        self.loss_coef = 0.05 * hyd_torque / self.omega_nom
        self.drive_torque = hyd_torque + self.loss_coef * self.omega_nom
        self.fuel_capacity = 2.5e8

    def _assign_primary_enthalpy(self, p_cold, p_core, p_hot, p_sg, h_cold, h_hot) -> None:
        net = self.network
        net.volumes["cold_leg"].set_ph(p_cold, h_cold, self.boron_ppm_ref)
        net.volumes["sg_primary"].set_ph(p_sg, h_cold, self.boron_ppm_ref)
        net.volumes["core"].set_ph(p_core, h_hot, self.boron_ppm_ref)
        net.volumes["hot_leg"].set_ph(p_hot, h_hot, self.boron_ppm_ref)

    def _assign_primary(self, p_cold: float, p_core: float, p_hot: float, p_sg: float) -> None:
        net = self.network
        net.volumes["cold_leg"].set_liquid(p_cold, self.t_cold, self.boron_ppm_ref)
        net.volumes["core"].set_liquid(p_core, self.t_hot, self.boron_ppm_ref)
        net.volumes["hot_leg"].set_liquid(p_hot, self.t_hot, self.boron_ppm_ref)
        net.volumes["sg_primary"].set_liquid(p_sg, self.t_cold, self.boron_ppm_ref)

    def turbine_outlet_enthalpy(self, sg_pressure_pa: float) -> float:
        _liquid, vapor = saturation_properties(sg_pressure_pa)
        liquid_c, vapor_c = saturation_properties(self.p_condenser)
        quality = (vapor.s_j_kg_k - liquid_c.s_j_kg_k) / (vapor_c.s_j_kg_k - liquid_c.s_j_kg_k)
        if quality < 0.0 or quality > 1.0:
            raise DomainError("Изоэнтропийное расширение покинуло двухфазную область конденсатора")
        h_is = liquid_c.h_j_kg + quality * (vapor_c.h_j_kg - liquid_c.h_j_kg)
        return vapor.h_j_kg - self.eta_turbine * (vapor.h_j_kg - h_is)

    def electrical_power_w(self) -> float:
        steam = self.network.junctions["steam"]
        if steam.delivery_enthalpy_j_kg is None:
            return 0.0
        drop = self.h_g_current - steam.delivery_enthalpy_j_kg
        return max(steam.mass_flow_kg_s, 0.0) * drop * 0.98

    def pack(self) -> np.ndarray:
        extra = np.concatenate(
            [
                [self.kinetics_state[0]],
                self.kinetics_state[1:],
                self.decay_state,
                [self.fuel_temperature, self.omega, self.rod],
            ]
        )
        return np.concatenate([extra, self.network.pack()])

    def unpack(self, state: np.ndarray) -> None:
        self.kinetics_state = state[:7].copy()
        self.decay_state = state[7:10].copy()
        self.fuel_temperature = float(state[10])
        self.omega = float(state[11])
        self.rod = float(state[12])
        self.network.unpack(state[13:])

    def reactivity(self) -> float:
        net = self.network
        density = net.volumes["cold_leg"].thermo.rho_kg_m3
        boron_ppm = net.volumes["core"].boron_fraction * 1.0e6
        rho_rod = -WORTH * (rod_integral_worth(self.rod) - rod_integral_worth(self.rod_ref))
        rho_fuel = self.doppler_coef * (np.sqrt(self.fuel_temperature) - np.sqrt(self.t_fuel_ref))
        rho_mod = self.density_coef * (density - self.rho_ref)
        rho_boron = self.boron_coef * (boron_ppm - self.boron_ppm_ref)
        return float(rho_rod + rho_fuel + rho_mod + rho_boron)

    def derivatives(self, _time: float, state: np.ndarray) -> np.ndarray:
        self.unpack(state)
        self.network.refresh_properties()
        self._check_domain()
        self._apply_heat_and_actuators()
        rho = self.reactivity()
        kin = self.kinetics.derivatives(self.kinetics_state[0], self.kinetics_state[1:], rho, 0.0)
        fission = self.kinetics.power_w(self.kinetics_state[0])
        d_decay = self.decay.derivatives(self.decay_state, fission)
        total = self.decay.total_power_w(fission, self.decay_state)
        t_coolant = self.network.volumes["core"].temperature_k
        d_tf = (total - self.ua_fuel * (self.fuel_temperature - t_coolant)) / self.fuel_capacity
        torque_hyd = 0.0 if abs(self.omega) < 1.0 else self._hydraulic_torque()
        torque_drive = self.drive_torque if self.motor_on else 0.0
        d_omega = (torque_drive - torque_hyd - self.loss_coef * self.omega) / self.inertia
        if self.scram_latched:
            rod_speed = -self.scram_speed
        else:
            rod_speed = self.rod_command
        if self.rod <= 0.0 and rod_speed < 0.0:
            rod_speed = 0.0
        if self.rod >= 1.0 and rod_speed > 0.0:
            rod_speed = 0.0
        d_rod = rod_speed
        net_dot = self.network.derivatives(self.network.pack())
        # derivatives() распаковывает сеть заново из pack, состояние уже то же.
        return np.concatenate([kin, d_decay, [d_tf, d_omega, d_rod], net_dot])

    def _hydraulic_torque(self) -> float:
        pump = self.network.junctions["pump"]
        rho = self.network.volumes["cold_leg"].thermo.rho_kg_m3
        if pump.mass_flow_kg_s < 0.0:
            raise DomainError("Момент насоса при обратном расходе не входит в этот паспорт")
        # Напор масштабируется квадратом скорости. Номинал соответствует omega_nom.
        speed_ratio = self.omega / self.omega_nom
        dp = self.dp_pump_nom * speed_ratio**2
        pump.pump_pressure_pa = dp
        return pump.mass_flow_kg_s * dp / (rho * self.omega)

    def _apply_heat_and_actuators(self) -> None:
        net = self.network
        fission = self.kinetics.power_w(self.kinetics_state[0])
        total = self.decay.total_power_w(fission, self.decay_state)
        t_sat = net.volumes["sg_secondary"].temperature_k
        q_sg = self.ua_sg * (net.volumes["sg_primary"].temperature_k - t_sat)
        net.volumes["core"].heat_w = total
        net.volumes["hot_leg"].heat_w = 0.0
        net.volumes["sg_primary"].heat_w = -q_sg
        net.volumes["cold_leg"].heat_w = 0.0
        net.volumes["pressurizer"].heat_w = 0.0
        net.volumes["sg_secondary"].heat_w = q_sg
        t_cond = net.volumes["condenser"].temperature_k
        net.volumes["condenser"].heat_w = -self.ua_condenser * max(t_cond - self.t_sink, 0.0)
        _liquid, vapor = saturation_properties(net.volumes["sg_secondary"].pressure_pa)
        self.h_g_current = vapor.h_j_kg
        steam = net.junctions["steam"]
        steam.flow_target_kg_s = (
            self.m_steam_nom
            * self.throttle
            * (net.volumes["sg_secondary"].pressure_pa / self.p_sg_nom)
        )
        steam.delivery_enthalpy_j_kg = self.turbine_outlet_enthalpy(
            net.volumes["sg_secondary"].pressure_pa
        )
        feed = net.junctions["feed"]
        feed.flow_target_kg_s = max(steam.mass_flow_kg_s, 0.0)
        liquid_c, _vapor_c = saturation_properties(self.p_condenser)
        feed.delivery_enthalpy_j_kg = liquid_c.h_j_kg + liquid_c.v_m3_kg * (
            net.volumes["sg_secondary"].pressure_pa - self.p_condenser
        )
        self._hydraulic_torque()

    def _check_domain(self) -> None:
        for name in ("core", "hot_leg", "sg_primary", "cold_leg"):
            quality = self.network.volumes[name].thermo.quality
            if quality > 1e-4:
                raise DomainError(
                    "Объёмное кипение в первичном контуре %s вне паспорта ВВЭР" % name
                )
        if self.fuel_temperature > 2800.0 or self.fuel_temperature < 300.0:
            raise DomainError("Температура топлива вне области теплофизики модели")

    def advance(self, dt_s: float, rtol: float = 1e-5, atol: float = 1e-4) -> dict:
        from core.solvers import integrate

        y0 = self.pack()
        result = integrate(
            self.derivatives,
            y0,
            (self.time_s, self.time_s + dt_s),
            rtol=rtol,
            atol=atol,
            max_step=dt_s,
        )
        self.time_s += dt_s
        self.unpack(result.final)
        self.network.refresh_properties()
        return self.observation()

    def observation(self) -> dict:
        net = self.network
        fission = self.kinetics.power_w(self.kinetics_state[0])
        total = self.decay.total_power_w(fission, self.decay_state)
        return {
            "time_s": self.time_s,
            "fission_power_w": fission,
            "thermal_power_w": total,
            "electrical_power_w": self.electrical_power_w(),
            "fuel_temperature_k": self.fuel_temperature,
            "rod_insertion": self.rod,
            "reactivity": self.reactivity(),
            "primary_pressure_pa": net.volumes["pressurizer"].pressure_pa,
            "sg_pressure_pa": net.volumes["sg_secondary"].pressure_pa,
            "mass_flow_kg_s": net.junctions["pump"].mass_flow_kg_s,
            "steam_flow_kg_s": net.junctions["steam"].mass_flow_kg_s,
            "scram_latched": self.scram_latched,
            "applicability": self.applicability.value,
        }

    def snapshot(self) -> PlantSnapshot:
        return PlantSnapshot(
            self.time_s, self.pack().copy(), self.scram_latched, self.throttle, self.motor_on
        )

    def restore(self, snapshot: PlantSnapshot) -> None:
        self.time_s = snapshot.time_s
        self.scram_latched = snapshot.scram_latched
        self.throttle = snapshot.throttle
        self.motor_on = snapshot.motor_on
        self.unpack(snapshot.state)
        self.network.refresh_properties()
