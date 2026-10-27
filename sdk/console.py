"""Пульт исследователя поверх той же сессии, что SDK и сценарии.

Стержень не переносится мгновенно: ползунок задаёт цель, скорость ограничена.
Пространственный поток и ксеноновые колебания этот паспорт не рисует.
"""

from __future__ import annotations

from control.regulation import PID
from core.status import DomainError, SolverError
from sdk.api import create_session, step_session


class ControlBridge:
    def __init__(self, console: "VVERConsole") -> None:
        self.console = console
        self.auto_power = False
        self.auto_boron = False
        nominal_mw = console.core.fission_power_nom / 1.0e6
        self.target_power = nominal_mw

    @property
    def rod_position(self) -> float:
        return self.console.core.rod

    @rod_position.setter
    def rod_position(self, value: float) -> None:
        self.console.rod_target = min(1.0, max(0.0, float(value)))

    @property
    def boron_concentration(self) -> float:
        return self.console.core.network.volumes["core"].boron_fraction * 1.0e6

    @boron_concentration.setter
    def boron_concentration(self, value: float) -> None:
        self.console.core.set_boron_ppm(float(value))


class ActuatorBridge:
    def __init__(self, console: "VVERConsole") -> None:
        self.console = console

    @property
    def turbine_throttle(self) -> float:
        return self.console.core.throttle

    @turbine_throttle.setter
    def turbine_throttle(self, value: float) -> None:
        self.console.core.throttle = min(1.0, max(0.0, float(value)))

    @property
    def feedwater_pump_speed(self) -> float:
        nom = self.console.core.m_steam_nom
        return 1.0 + self.console.core.feed_bias_kg_s / nom

    @feedwater_pump_speed.setter
    def feedwater_pump_speed(self, value: float) -> None:
        fraction = min(1.5, max(0.0, float(value)))
        self.console.core.feed_bias_kg_s = (fraction - 1.0) * self.console.core.m_steam_nom

    @property
    def motor_on(self) -> bool:
        return self.console.core.motor_on

    @motor_on.setter
    def motor_on(self, value: bool) -> None:
        self.console.core.motor_on = bool(value)


class VVERConsole:
    def __init__(self, rod: float | None = None, boron_ppm: float | None = None) -> None:
        self.core = create_session("vver")
        self.control = ControlBridge(self)
        self.plant = ActuatorBridge(self)
        self.power_pid = PID(kp=0.4, ki=0.05, kd=0.0, output_limit=0.2)
        self.rod_target = self.core.rod
        self.message = ""
        self.out_of_domain = False
        if rod is not None:
            self.core.rod = min(1.0, max(0.0, float(rod)))
            self.rod_target = self.core.rod
        if boron_ppm is not None:
            self.core.set_boron_ppm(float(boron_ppm))

    @property
    def time(self) -> float:
        return self.core.time_s

    @property
    def rod_position(self) -> float:
        return self.core.rod

    def step(self, dt: float = 0.1) -> dict:
        thermal = self.core.decay.total_power_w(
            self.core.kinetics.power_w(self.core.kinetics_state[0]), self.core.decay_state
        )
        error = (thermal - self.control.target_power * 1.0e6) / self.core.fission_power_nom
        commands: dict = {
            "throttle": self.core.throttle,
            "feed_bias_kg_s": self.core.feed_bias_kg_s,
        }
        if self.core.scram_latched:
            commands["scram"] = True
        elif self.control.auto_power:
            commands["rod_command"] = self.power_pid.step(error, dt)
        else:
            commands["rod_command"] = max(-1.0, min(1.0, (self.rod_target - self.core.rod) / dt))
        if self.control.auto_boron:
            ppm = self.control.boron_concentration + 20.0 * error * dt
            commands["boron_ppm"] = max(0.0, ppm)
        try:
            observed = step_session(self.core, dt, commands)
            self.out_of_domain = False
            self.message = ""
        except (DomainError, SolverError) as exc:
            self.out_of_domain = True
            self.message = str(exc)
            observed = self.core.observation()
        return {
            "thermal_power_mw": observed["thermal_power_w"] / 1.0e6,
            "max_fuel_temp": observed["fuel_temperature_k"] - 273.15,
            "turbine_power_mw": observed["electrical_power_w"] / 1.0e6,
            "sg_pressure": observed["sg_pressure_pa"],
            "out_of_domain": self.out_of_domain,
            "message": self.message,
            "rod": observed["rod_insertion"],
        }
