import torch
from .neutronics_1d import Neutronics1D
from .thermal_1d import ThermalModel1D
from .poisoning_1d import XenonIodine1D
from .control_1d import ControlSystem1D
from .depletion_1d import FuelDepletion1D
from .plant_balance import PlantBalance
from .reactor_configs import ReactorConfig, VVER_1000, REACTORS


class ReactorSimulator1D:
    def __init__(self, config_name="vver", num_nodes=50, device="cpu", dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.num_nodes = num_nodes

        # Load Config
        if isinstance(config_name, str):
            self.config = REACTORS.get(config_name, VVER_1000)
        elif isinstance(config_name, ReactorConfig):
            self.config = config_name
        else:
            self.config = VVER_1000

        # Initialize Physics Modules
        self.neutronics = Neutronics1D(
            config=self.config, num_nodes=num_nodes, device=device, dtype=dtype
        )
        self.thermal = ThermalModel1D(
            config=self.config, num_nodes=num_nodes, device=device, dtype=dtype
        )
        self.poisoning = XenonIodine1D(num_nodes=num_nodes, device=device, dtype=dtype)
        self.fuel = FuelDepletion1D(num_nodes=num_nodes, device=device, dtype=dtype)
        self.control = ControlSystem1D(device=device, dtype=dtype)
        self.plant = PlantBalance(dt=0.05)

        # Simulation State
        self.time = 0.0
        self.dt = 0.05  # s

        # Control Inputs
        self.target_power = 100.0  # %
        self.manual_control = False
        self.pump_signal = 1.0  # 0.0 to 1.0

    def step(self, dt=None):
        if dt is None:
            dt = self.dt

        self.time += dt
        power_percent, power_distribution = self._power_from_flux()

        # 1. Control System uses the power at the beginning of the step.
        self.control.step(
            target_power=self.target_power,
            current_power=power_percent,
            dt=dt,
            manual=self.manual_control,
        )

        # 2. Thermal-Hydraulics & Plant Balance
        hot_leg_temp_k = self.thermal.temp_coolant[-1].item()
        hot_leg_temp_c = hot_leg_temp_k - 273.15
        flow_rate = self.thermal.mass_flow.item()

        cold_leg_temp_c = self.plant.step(hot_leg_temp_c, flow_rate, dt)
        cold_leg_temp_k = cold_leg_temp_c + 273.15

        self.thermal.pump_speed.fill_(self.pump_signal)
        fuel_temp, coolant_temp = self.thermal(power_distribution, dt, inlet_temp_k=cold_leg_temp_k)

        # 3. Reactivity Feedback
        reactivity_feedback = self.thermal.get_reactivity_feedback()

        # 4. Poisoning
        # poisoning returns delta_Sigma_a vector
        xe_poisoning_sigma = self.poisoning(
            self.neutronics.phi, dt, nu_sigma_f=self.neutronics.nu_Sigma_f
        )

        # 5. Neutronics
        # Pass all components to neutronics forward
        self.neutronics.forward(
            dt=dt,
            rod_pos=self.control.rod_position,
            boron_ppm=self.control.boron_concentration,
            feedback_rho=reactivity_feedback,
            xenon_sigma=xe_poisoning_sigma,
        )
        power_percent, power_distribution = self._power_from_flux()

        return {
            "time": self.time,
            "power": power_percent,  # % of nominal thermal power after the neutronics update
            "total_power": self.plant.electric_power,  # For UI backward compatibility (though this is MW now)
            "turbine_power_mw": self.plant.electric_power,
            "thermal_power_mw": torch.sum(power_distribution).item() / 1e6,
            "flux_profile": self.neutronics.phi.cpu().numpy(),
            "fuel_temp_profile": fuel_temp.cpu().numpy(),
            "coolant_temp_profile": coolant_temp.cpu().numpy(),
            "void_profile": self.thermal.void_fraction.cpu().numpy(),
            "rod_position": self.control.rod_position,
            "boron": self.control.boron_concentration,
            "mass_flow": self.thermal.mass_flow.item(),
            "max_fuel_temp": torch.max(fuel_temp).item(),
            "max_clad_temp": torch.max(self.thermal.temp_clad).item(),
            "max_void_fraction": torch.max(self.thermal.void_fraction).item(),
            "sg_pressure": self.plant.sg_pressure,
            "sg_level": self.plant.sg_level,
        }

    def _power_from_flux(self):
        """Тепловая мощность из потока. Номинальный средний поток 1e13 соответствует 100%."""
        nominal_average_flux = 1.0e13
        nominal_total_power = self.config.thermal_power * 1.0e6
        power_factor = (nominal_total_power / self.num_nodes) / nominal_average_flux
        distribution = self.neutronics.phi * power_factor
        thermal_w = torch.sum(distribution).item()
        return 100.0 * thermal_w / nominal_total_power, distribution

    def burnup_step(self, days):
        """Шаг выгорания и передача изменения сечений в нейтронику."""
        seconds = days * 24 * 3600
        self.fuel.step(self.neutronics.phi, seconds)
        self.neutronics.set_fuel_feedback(self.fuel.get_cross_section_changes())

    def set_pump_speed(self, speed):
        self.pump_signal = max(0.0, min(1.0, speed))
