"""Сосредоточенная установка ВВЭР/PWR: один эквивалентный контур.

Это не четырёхпетлевая геометрическая копия блока и не plant reference.
Стационар собран так, чтобы невязки массы и энергии обращались в ноль
при номинальных параметрах модели.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from components.condenser import DynamicCondenser
from components.heat_exchanger import HeatExchangerWall
from control.regulation import ProtectionLogic, high_trip_time
from core.status import Applicability, DomainError, ModelPassport, ModelStatus
from physics.depletion.chains import DecayHeat
from physics.hydraulics.network import BoundaryState, FlowNetwork, Junction, Volume
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
        "стенка ПГ хранит температуру: два UA в ряду в номинале дают прежний общий коэффициент",
        "теплоёмкость стенки — параметр модели с постоянной времени 2 с, не масса труб блока",
        "турбина — постоянный изоэнтропийный КПД и отделённый насыщенный пар",
        "конденсатор — двухфазный объём: давление из (rho, u), не заданный вакуум",
        "питание — кривая напора и квадратичного сопротивления; при нехватке напора расход равен нулю",
        "сброс пара мимо турбины открывается выше 7 МПа; это параметр модели, не уставка блока",
        "нагреватель и впрыск компенсатора включаются вне мёртвой зоны вокруг номинального давления",
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


def worth_between(start: float, end: float) -> float:
    """Приращение интегральной кривой. Оно совпадает с интегралом дифференциальной эффективности."""
    return rod_integral_worth(end) - rod_integral_worth(start)


def doppler_reactivity(temperature_k: float, reference_k: float, coefficient: float) -> float:
    if temperature_k <= 0.0 or reference_k <= 0.0:
        raise DomainError("Температура Доплера должна быть положительной")
    return coefficient * (np.sqrt(temperature_k) - np.sqrt(reference_k))


def doppler_slope_per_k(temperature_k: float, coefficient: float) -> float:
    if temperature_k <= 0.0:
        raise DomainError("Температура Доплера должна быть положительной")
    return coefficient / (2.0 * np.sqrt(temperature_k))


@dataclass
class PlantSnapshot:
    time_s: float
    state: np.ndarray
    scram_latched: bool
    throttle: float
    motor_on: bool
    rod_command: float = 0.0
    protection_timer_s: float = 0.0
    protection_latched: bool = False
    feed_bias_kg_s: float = 0.0
    protection_setpoint_w: float = 0.0
    protection_delay_s: float = 0.0


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
        self.feed_bias_kg_s = 0.0
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
        self.fuel_to_coolant_w = self.fission_power_nom
        self.power_protection = ProtectionLogic(setpoint=1.25 * self.fission_power_nom, delay_s=2.0)
        self._power_channel_bias = (-0.001, 0.0, 0.001)
        self.last_protection_event_s = None
        self.last_energy_ledger = None

    def _build_network(self) -> FlowNetwork:
        volumes = {
            "core": Volume("core", 18.0, 0.0),
            "hot_leg": Volume("hot_leg", 8.0, 0.0),
            "sg_primary": Volume("sg_primary", 22.0, 0.0),
            "cold_leg": Volume("cold_leg", 8.0, 0.0),
            "pressurizer": Volume("pressurizer", 55.0, 0.0),
            "sg_secondary": Volume("sg_secondary", 120.0, 0.0),
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
                enthalpy_mode="donor",
                flow_law="target",
                flow_time_constant_s=0.05,
            ),
            "spray": Junction(
                "spray",
                "cold_leg",
                "pressurizer",
                0.02,
                6.0,
                0.0,
                enthalpy_mode="donor",
                flow_law="target",
                flow_time_constant_s=0.5,
            ),
            "dump": Junction(
                "dump",
                "sg_secondary",
                "condenser",
                0.3,
                4.0,
                0.0,
                enthalpy_mode="saturated_vapor",
                boron_carry=0.0,
                flow_law="target",
                flow_time_constant_s=0.2,
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
        self._condenser_pressure_pa = self.p_condenser
        liquid_pool, vapor_pool = saturation_properties(self.p_condenser)
        # Доля пара задаёт запас до границы купола, а не измеренный объём блока.
        quality = 0.05
        specific_volume = (1.0 - quality) * liquid_pool.v_m3_kg + quality * vapor_pool.v_m3_kg
        self.condenser = DynamicCondenser.from_quality(
            60000.0 * specific_volume, self.p_condenser, quality
        )
        self.hotwell_mass_kg = self.condenser.mass_kg
        self.hotwell_energy_j = self.condenser.energy_j
        self.hotwell_temperature_k = liquid_pool.T_k
        self.network.boundaries["condenser"] = BoundaryState(
            self.p_condenser, liquid_pool.h_j_kg, liquid_pool.rho_kg_m3, 0.0
        )
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
        self.sg_wall_ua_primary = 2.0 * self.ua_sg
        self.sg_wall_ua_secondary = 2.0 * self.ua_sg
        self.sg_wall_time_constant_s = 2.0
        self.sg_wall_capacity = self.sg_wall_time_constant_s * (
            self.sg_wall_ua_primary + self.sg_wall_ua_secondary
        )
        primary_temperature = net.volumes["sg_primary"].temperature_k
        self.sg_wall_temperature = (
            self.sg_wall_ua_primary * primary_temperature + self.sg_wall_ua_secondary * t_sat
        ) / (self.sg_wall_ua_primary + self.sg_wall_ua_secondary)
        self.sg_wall = HeatExchangerWall(
            self.sg_wall_capacity,
            self.sg_wall_ua_primary,
            self.sg_wall_ua_secondary,
            self.sg_wall_temperature,
        )
        h_bulk = self.network.boundaries["condenser"].enthalpy_j_kg
        q_sink = self.m_steam_nom * (self.h_turbine_nom - h_bulk)
        self.ua_condenser = q_sink / max(self.hotwell_temperature_k - self.t_sink, 1.0)
        feed_density = self.network.boundaries["condenser"].density_kg_m3
        static_pa = self.p_secondary - self.p_condenser
        friction_pa = 0.02 * static_pa
        feed.loss_coefficient = (
            friction_pa * 2.0 * feed_density * feed.area_m2**2 / self.m_steam_nom**2
        )
        self.feed_dp_nom_pa = static_pa + friction_pa
        feed.pump_pressure_pa = self.feed_dp_nom_pa
        feed.flow_law = "target"
        feed.flow_time_constant_s = 0.2
        self.ua_fuel = self.fission_power_nom / (
            self.t_fuel_ref - net.volumes["core"].temperature_k
        )
        self.omega_nom = 157.0
        self.inertia = 2.0e4
        hyd_torque = self.pump_power_w / self.omega_nom
        self.loss_coef = 0.05 * hyd_torque / self.omega_nom
        self.drive_torque = hyd_torque + self.loss_coef * self.omega_nom
        self.fuel_capacity = 2.5e8
        self.condenser_mass_ref_kg = self.hotwell_mass_kg
        self.feed_level_gain = 0.05
        self.p_dump_open_pa = 7.0e6
        self.dump_band_pa = 0.05 * self.p_dump_open_pa
        self.p_pressurizer_ref_pa = net.volumes["pressurizer"].pressure_pa
        self.pressurizer_deadband_pa = 0.15e6
        self.pressurizer_heater_span_pa = 0.5e6
        self.pressurizer_spray_span_pa = 0.5e6
        self.pressurizer_heater_max_w = 2.0e6
        self.pressurizer_spray_max_kg_s = 30.0
        self._pressurizer_heater_w = 0.0
        spray = net.junctions["spray"]
        spray.mass_flow_kg_s = 0.0
        spray.flow_target_kg_s = 0.0
        dump = net.junctions["dump"]
        dump.mass_flow_kg_s = 0.0
        dump.flow_target_kg_s = 0.0
        dump.delivery_enthalpy_j_kg = self.h_g

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
        liquid_c, vapor_c = saturation_properties(self._condenser_pressure_pa)
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
        return np.concatenate(
            [
                extra,
                self.network.pack(),
                [self.hotwell_mass_kg, self.hotwell_energy_j, self.sg_wall_temperature],
            ]
        )

    def unpack(self, state: np.ndarray) -> None:
        self.kinetics_state = state[:7].copy()
        self.decay_state = state[7:10].copy()
        self.fuel_temperature = float(state[10])
        self.omega = float(state[11])
        self.rod = float(state[12])
        self.network.unpack(state[13:-3])
        self.hotwell_mass_kg = float(state[-3])
        self.hotwell_energy_j = float(state[-2])
        self.sg_wall_temperature = float(state[-1])
        self.sg_wall.temperature_k = self.sg_wall_temperature

    def reactivity(self) -> float:
        net = self.network
        density = net.volumes["cold_leg"].thermo.rho_kg_m3
        boron_ppm = net.volumes["core"].boron_fraction * 1.0e6
        rho_rod = -WORTH * (rod_integral_worth(self.rod) - rod_integral_worth(self.rod_ref))
        rho_fuel = doppler_reactivity(self.fuel_temperature, self.t_fuel_ref, self.doppler_coef)
        rho_mod = self.density_coef * (density - self.rho_ref)
        rho_boron = self.boron_coef * (boron_ppm - self.boron_ppm_ref)
        return float(rho_rod + rho_fuel + rho_mod + rho_boron)

    def set_boron_ppm(self, ppm: float) -> None:
        """Одинаковая концентрация в объёмах первого контура. Масса бора следует за массой воды."""
        if ppm < 0.0:
            raise DomainError("Концентрация бора отрицательна")
        for name in ("core", "hot_leg", "sg_primary", "cold_leg", "pressurizer"):
            volume = self.network.volumes[name]
            volume.boron_kg = ppm * 1.0e-6 * volume.mass_kg

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
        d_tf = (total - self.fuel_to_coolant_w) / self.fuel_capacity
        torque_hyd = 0.0 if abs(self.omega) < 1.0 else self._hydraulic_torque()
        torque_drive = self.drive_torque if self.motor_on else 0.0
        d_omega = (torque_drive - torque_hyd - self.loss_coef * self.omega) / self.inertia
        if self.scram_latched:
            rod_speed = self.scram_speed
        else:
            rod_speed = self.rod_command
        if self.rod <= 0.0 and rod_speed < 0.0:
            rod_speed = 0.0
        if self.rod >= 1.0 and rod_speed > 0.0:
            rod_speed = 0.0
        d_rod = rod_speed
        self._sync_hotwell()
        net_dot = self.network.derivatives(self.network.pack())
        steam_flow = self.network.junctions["steam"].mass_flow_kg_s
        dump_flow = self.network.junctions["dump"].mass_flow_kg_s
        feed_flow = self.network.junctions["feed"].mass_flow_kg_s
        h_steam = self.network.junctions["steam"].delivery_enthalpy_j_kg
        h_dump = self.network.junctions["dump"].delivery_enthalpy_j_kg
        h_liquid = self.network.boundaries["condenser"].enthalpy_j_kg
        heat = -self.ua_condenser * max(self.hotwell_temperature_k - self.t_sink, 0.0)
        d_mass = steam_flow + dump_flow - feed_flow
        d_energy = steam_flow * h_steam + dump_flow * h_dump - feed_flow * h_liquid + heat
        return np.concatenate(
            [kin, d_decay, [d_tf, d_omega, d_rod], net_dot, [d_mass, d_energy, self._wall_rate_k_s]]
        )

    def _feed_curve_kg_s(self, feed) -> float:
        """Равновесие напора насоса, противодавления ПГ и квадратичного сопротивления.

        Если напора не хватает, обратный клапан держит нулевой расход.
        """
        back = (
            self.network.volumes["sg_secondary"].pressure_pa
            - self.network.boundaries["condenser"].pressure_pa
        )
        available = feed.pump_pressure_pa - back
        if available <= 0.0 or feed.loss_coefficient <= 0.0:
            return 0.0
        density = self.network.boundaries["condenser"].density_kg_m3
        return feed.area_m2 * (2.0 * density * available / feed.loss_coefficient) ** 0.5

    def _sync_hotwell(self) -> None:
        if self.hotwell_mass_kg <= 1.0:
            raise DomainError("Конденсатор осушен")
        self.condenser.mass_kg = self.hotwell_mass_kg
        self.condenser.energy_j = self.hotwell_energy_j
        state = self.condenser.state()
        if state.quality <= 1.0e-4 or state.quality >= 1.0 - 1.0e-4:
            raise DomainError("Динамический конденсатор потерял одну из фаз")
        self._condenser_pressure_pa = state.p_pa
        self.hotwell_temperature_k = state.T_k
        liquid, _vapor = saturation_properties(state.p_pa)
        boundary = self.network.boundaries["condenser"]
        boundary.enthalpy_j_kg = liquid.h_j_kg
        boundary.density_kg_m3 = liquid.rho_kg_m3
        boundary.pressure_pa = state.p_pa

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
        t_primary = net.volumes["sg_primary"].temperature_k
        t_sat = net.volumes["sg_secondary"].temperature_k
        self.sg_wall.temperature_k = self.sg_wall_temperature
        q_to_wall, q_to_secondary = self.sg_wall.heat_flows_w(t_primary, t_sat)
        self._wall_rate_k_s = self.sg_wall.derivative_k_s(t_primary, t_sat)
        q_fuel = self.ua_fuel * (self.fuel_temperature - net.volumes["core"].temperature_k)
        self.fuel_to_coolant_w = q_fuel
        net.volumes["core"].heat_w = q_fuel
        net.volumes["hot_leg"].heat_w = 0.0
        net.volumes["sg_primary"].heat_w = -q_to_wall
        net.volumes["cold_leg"].heat_w = 0.0
        heater_w, spray_kg_s = self._pressurizer_controls(net.volumes["pressurizer"].pressure_pa)
        self._pressurizer_heater_w = heater_w
        net.volumes["pressurizer"].heat_w = heater_w
        net.junctions["spray"].flow_target_kg_s = spray_kg_s
        net.volumes["sg_secondary"].heat_w = q_to_secondary
        self._sync_hotwell()
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
        dump = net.junctions["dump"]
        overpressure = net.volumes["sg_secondary"].pressure_pa - self.p_dump_open_pa
        opened = 0.0 if overpressure <= 0.0 else overpressure / self.dump_band_pa
        dump.flow_target_kg_s = min(self.m_steam_nom, self.m_steam_nom * opened)
        dump.delivery_enthalpy_j_kg = self.h_g_current
        feed = net.junctions["feed"]
        level_error = self.hotwell_mass_kg - self.condenser_mass_ref_kg
        demand = max(
            0.0,
            steam.mass_flow_kg_s
            + dump.mass_flow_kg_s
            + self.feed_level_gain * level_error
            + self.feed_bias_kg_s,
        )
        speed = 0.0 if self.m_steam_nom <= 0.0 else demand / self.m_steam_nom
        feed.pump_pressure_pa = self.feed_dp_nom_pa * speed * speed
        feed.flow_target_kg_s = self._feed_curve_kg_s(feed)
        liquid_c, _vapor_c = saturation_properties(self._condenser_pressure_pa)
        static_lift = liquid_c.v_m3_kg * (
            net.volumes["sg_secondary"].pressure_pa - self._condenser_pressure_pa
        )
        delivered = liquid_c.h_j_kg + static_lift
        if feed.pump_pressure_pa > 0.0 and liquid_c.rho_kg_m3 > 0.0:
            delivered -= feed.pump_pressure_pa / liquid_c.rho_kg_m3
        feed.delivery_enthalpy_j_kg = delivered
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

    def thermal_power_w_from_state(self, state: np.ndarray) -> float:
        fission = self.kinetics.power_w(float(state[0]))
        return self.decay.total_power_w(fission, np.asarray(state[7:10], dtype=float))

    def account_rates_w(self, dot: np.ndarray) -> dict:
        """Мощности запасов после вызова derivatives: топливо, теплоноситель, бак, стенка, вал."""
        coolant = 0.0
        for index in range(len(self.network.volumes)):
            coolant += float(dot[13 + 3 * index + 1])
        return {
            "fuel_w": self.fuel_capacity * float(dot[10]),
            "coolant_w": coolant,
            "hotwell_w": float(dot[-2]),
            "wall_w": self.sg_wall_capacity * float(dot[-1]),
            "shaft_w": self.inertia * self.omega * float(dot[11]),
        }

    def stored_accounts_j(self) -> dict:
        return {
            "fuel_j": self.fuel_capacity * self.fuel_temperature,
            "coolant_j": self.network.total_energy(),
            "hotwell_j": self.hotwell_energy_j,
            "wall_j": self.sg_wall_capacity * self.sg_wall_temperature,
            "shaft_j": 0.5 * self.inertia * self.omega**2,
        }

    def boundary_terms_w(self) -> dict:
        """Внешние потоки. Напор насоса остаётся внутри пары «вал — теплоноситель»."""
        fission = self.kinetics.power_w(float(self.kinetics_state[0]))
        total = self.decay.total_power_w(fission, self.decay_state)
        steam = self.network.junctions["steam"]
        feed = self.network.junctions["feed"]
        turbine = steam.mass_flow_kg_s * (self.h_g_current - steam.delivery_enthalpy_j_kg)
        boundary = self.network.boundaries["condenser"]
        delivered = 0.0 if feed.delivery_enthalpy_j_kg is None else feed.delivery_enthalpy_j_kg
        feed_work = feed.mass_flow_kg_s * (delivered - boundary.enthalpy_j_kg)
        if (
            feed.mass_flow_kg_s > 0.0
            and feed.pump_pressure_pa > 0.0
            and boundary.density_kg_m3 > 0.0
        ):
            feed_work += feed.mass_flow_kg_s * feed.pump_pressure_pa / boundary.density_kg_m3
        condenser = -self.ua_condenser * max(self.hotwell_temperature_k - self.t_sink, 0.0)
        motor = self.drive_torque * self.omega if self.motor_on else 0.0
        loss = self.loss_coef * self.omega**2
        return {
            "nuclear_w": total,
            "motor_w": motor,
            "shaft_loss_w": -loss,
            "feed_work_w": feed_work,
            "turbine_work_w": -turbine,
            "condenser_w": condenser,
            "pressurizer_heater_w": self._pressurizer_heater_w,
        }

    def set_sink_temperature_k(self, temperature_k: float) -> None:
        if temperature_k < 273.15 or temperature_k > 373.15:
            raise DomainError("Температура стока вне 273.15–373.15 K")
        self.t_sink = float(temperature_k)

    def _pressurizer_controls(self, pressure_pa: float) -> tuple:
        """Нагреватель только греет, впрыск только добавляет жидкость. В мёртвой зоне оба нули."""
        low = self.p_pressurizer_ref_pa - self.pressurizer_deadband_pa
        high = self.p_pressurizer_ref_pa + self.pressurizer_deadband_pa
        if pressure_pa < low:
            fraction = min(1.0, (low - pressure_pa) / self.pressurizer_heater_span_pa)
            return self.pressurizer_heater_max_w * fraction, 0.0
        if pressure_pa > high:
            fraction = min(1.0, (pressure_pa - high) / self.pressurizer_spray_span_pa)
            return 0.0, self.pressurizer_spray_max_kg_s * fraction
        return 0.0, 0.0

    def _reject_if_out_of_domain(self) -> None:
        self.network.refresh_properties()
        self._sync_hotwell()
        self._check_domain()
        for volume in self.network.volumes.values():
            mismatch = abs(volume.volume_mismatch) > 0.02 * volume.thermo.v_m3_kg
            if abs(volume.energy_mismatch_j_kg) > 5.0e3 or mismatch:
                raise DomainError(
                    "Принятый шаг вывел объём %s из равновесной области" % volume.name
                )

    def _state_at(self, result, t0: float, t1: float, time_s: float) -> np.ndarray:
        if result.dense is not None:
            return np.asarray(result.dense(float(time_s)), dtype=float).reshape(-1)
        if t1 == t0:
            return np.asarray(result.y[:, 0], dtype=float).copy()
        times = np.asarray(result.t, dtype=float)
        if time_s <= times[0]:
            return np.asarray(result.y[:, 0], dtype=float).copy()
        if time_s >= times[-1]:
            return np.asarray(result.y[:, -1], dtype=float).copy()
        index = int(np.searchsorted(times, time_s) - 1)
        span = float(times[index + 1] - times[index])
        weight = 0.0 if span == 0.0 else (time_s - float(times[index])) / span
        left = result.y[:, index]
        right = result.y[:, index + 1]
        return np.asarray((1.0 - weight) * left + weight * right, dtype=float)

    def _mesh_times(self, result, t0: float, t1: float) -> list:
        nodes = result.dense.ts if result.dense is not None else result.t
        times = [t0]
        for node in nodes:
            moment = float(node)
            if t0 + 1.0e-9 < moment < t1 - 1.0e-9:
                times.append(moment)
        times.append(t1)
        refined = [times[0]]
        for left, right in zip(times, times[1:]):
            refined.append(0.5 * (left + right))
            refined.append(right)
        return refined

    def _protection_trip_time(self, result, t0: float, t1: float):
        if self.power_protection.latched or self.scram_latched:
            self.power_protection.latched = True
            self.scram_latched = True
            return None

        def value_at(time_s: float) -> float:
            return self.thermal_power_w_from_state(self._state_at(result, t0, t1, time_s))

        for left, right in zip(
            self._mesh_times(result, t0, t1), self._mesh_times(result, t0, t1)[1:]
        ):
            if right <= left:
                continue
            found = high_trip_time(self.power_protection, value_at, left, right, right - left)
            if found is None:
                continue
            return min(t1, max(t0, float(found)))
        return None

    def _quadrature_times(self, result, t0: float, t1: float) -> list:
        uniform = np.linspace(t0, t1, 11)
        merged = {float(t0), float(t1)}
        merged.update(float(item) for item in self._mesh_times(result, t0, t1))
        merged.update(float(item) for item in uniform)
        return sorted(merged)

    def _samples(self, result, t0: float, t1: float) -> list:
        return [
            (time_s, self._state_at(result, t0, t1, time_s).copy())
            for time_s in self._quadrature_times(result, t0, t1)
        ]

    def energy_ledger(self, samples: list) -> dict:
        """Приращение запасов против трапеций внешних потоков вдоль принятой траектории."""
        self.unpack(samples[0][1])
        stored_0 = self.stored_accounts_j()
        times = []
        flows = []
        for time_s, state in samples:
            self.derivatives(float(time_s), state)
            times.append(float(time_s))
            flows.append(self.boundary_terms_w())
        self.unpack(samples[-1][1])
        stored_1 = self.stored_accounts_j()
        integrated = {key: 0.0 for key in flows[0]}
        for index in range(len(times) - 1):
            span = times[index + 1] - times[index]
            for key in integrated:
                integrated[key] += 0.5 * span * (flows[index][key] + flows[index + 1][key])
        delta = {key: stored_1[key] - stored_0[key] for key in stored_0}
        residual = sum(delta.values()) - sum(integrated.values())
        return {
            "delta_j": delta,
            "supplied_j": integrated,
            "residual_j": float(residual),
            "interval_s": times[-1] - times[0],
        }

    def advance(self, dt_s: float, rtol: float = 1e-5, atol: float = 1e-4) -> dict:
        from core.solvers import SolverError, integrate

        if dt_s <= 0.0:
            raise DomainError("Шаг установки должен быть положительным")
        saved = self.snapshot()
        try:
            t0 = self.time_s
            t1 = t0 + dt_s
            y0 = self.pack()
            first = integrate(
                self.derivatives,
                y0,
                (t0, t1),
                rtol=rtol,
                atol=atol,
                max_step=dt_s,
                dense_output=True,
            )
            trip = self._protection_trip_time(first, t0, t1)
            samples = self._samples(first, t0, t1)
            event_time = None
            if trip is not None and trip < t1 - 1.0e-10:
                event_time = trip
                self.unpack(y0 if trip <= t0 + 1.0e-10 else self._state_at(first, t0, t1, trip))
                self.time_s = trip
                self.power_protection.latched = True
                self.power_protection.timer_s = self.power_protection.delay_s
                self.scram_latched = True
                self._reject_if_out_of_domain()
                samples = [item for item in samples if item[0] <= trip + 1.0e-12]
                samples.append((trip, self.pack().copy()))
                if t1 > trip + 1.0e-10:
                    second = integrate(
                        self.derivatives,
                        self.pack(),
                        (trip, t1),
                        rtol=rtol,
                        atol=atol,
                        max_step=t1 - trip,
                        dense_output=True,
                    )
                    self.time_s = t1
                    self.unpack(second.final)
                    samples.extend(self._samples(second, trip, t1)[1:])
            else:
                self.time_s = t1
                self.unpack(first.final)
                if trip is not None:
                    event_time = trip
                    self.power_protection.latched = True
                    self.scram_latched = True
            self._reject_if_out_of_domain()
            self.last_energy_ledger = self.energy_ledger(samples)
            self.last_protection_event_s = event_time
            return self.observation()
        except (DomainError, SolverError):
            self.restore(saved)
            raise

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
            "feed_flow_kg_s": net.junctions["feed"].mass_flow_kg_s,
            "dump_flow_kg_s": net.junctions["dump"].mass_flow_kg_s,
            "pressurizer_heater_w": self._pressurizer_heater_w,
            "energy_residual_j": None
            if self.last_energy_ledger is None
            else self.last_energy_ledger["residual_j"],
            "scram_latched": self.scram_latched,
            "applicability": self.applicability.value,
        }

    def snapshot(self) -> PlantSnapshot:
        return PlantSnapshot(
            self.time_s,
            self.pack().copy(),
            self.scram_latched,
            self.throttle,
            self.motor_on,
            self.rod_command,
            self.power_protection.timer_s,
            self.power_protection.latched,
            self.feed_bias_kg_s,
            self.power_protection.setpoint,
            self.power_protection.delay_s,
        )

    def restore(self, snapshot: PlantSnapshot) -> None:
        self.time_s = snapshot.time_s
        self.scram_latched = snapshot.scram_latched
        self.throttle = snapshot.throttle
        self.motor_on = snapshot.motor_on
        self.rod_command = snapshot.rod_command
        self.feed_bias_kg_s = snapshot.feed_bias_kg_s
        self.power_protection.timer_s = snapshot.protection_timer_s
        self.power_protection.latched = snapshot.protection_latched
        self.power_protection.setpoint = snapshot.protection_setpoint_w
        self.power_protection.delay_s = snapshot.protection_delay_s
        self.unpack(snapshot.state)
        self.network.refresh_properties()
        self._sync_hotwell()
