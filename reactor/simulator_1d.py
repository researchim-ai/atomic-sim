
"""
1D Reactor Simulator (Spatial Kinetics)
Объединяет 1D нейтронную кинетику, 1D теплогидравлику и 1D ксеноновые колебания.
"""

import torch
import numpy as np
from .neutronics_1d import Neutronics1D
from .thermal_1d import ThermalModel1D
from .poisoning_1d import XenonIodine1D
from .control_1d import ControlSystem1D
from .depletion_1d import FuelDepletion1D
from .reactor_configs import VVER_1000, REACTORS

class ReactorSimulator1D:
    def __init__(self, config_name='vver', num_nodes=50, device='cpu', dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.num_nodes = num_nodes
        
        # Load Config
        self.config = REACTORS.get(config_name, VVER_1000)
        
        self.neutronics = Neutronics1D(config=self.config, num_nodes=num_nodes, device=device, dtype=dtype)
        self.thermal = ThermalModel1D(config=self.config, num_nodes=num_nodes, device=device, dtype=dtype)
        self.poisoning = XenonIodine1D(num_nodes=num_nodes, device=device, dtype=dtype)
        self.fuel = FuelDepletion1D(num_nodes=num_nodes, device=device, dtype=dtype)
        self.control = ControlSystem1D(device=device, dtype=dtype)
        
        # Инициализация равновесного ксенона под начальный поток
        self.poisoning.reset(flux_profile=self.neutronics.phi)
        
        self.time = 0.0
        self.rod_position = 0.2 # Start slightly inserted (20%)
        self.rod_speed = 0.0 # units/sec
        self.boron_concentration = 100.0 # ppm (Low initial boron)
        
        # Для записи истории
        self.history = []
        
    def step(self, dt=0.01, flow_factor=1.0):
        # 0. Автоматическое управление
        current_power_MW = torch.sum(self.neutronics.phi).item() * (60.0 / 1e13) 
        
        ctrl_rod_speed, ctrl_boron_change = self.control.step(
            dt, current_power_MW, self.rod_position, self.boron_concentration
        )
        
        if self.control.auto_power:
            self.rod_speed = ctrl_rod_speed
            
        if self.control.auto_boron:
            self.boron_concentration += ctrl_boron_change * dt
        
        # 1. Физическое движение стержней
        # Use speed from config if needed, but here we use abstract speed from controller
        self.rod_position += self.rod_speed * dt
        self.rod_position = max(0.0, min(1.0, self.rod_position))
        self.boron_concentration = max(0.0, self.boron_concentration)
        
        # 2. Получение обратной связи от теплофизики и ксенона
        reactivity_feedback = self.thermal.compute_reactivity_feedback()
        xenon_absorption = self.poisoning.get_absorption_cross_section()
        fuel_feedback = self.fuel.get_cross_section_changes()
        
        # 3. Шаг нейтроники
        self.neutronics.update_cross_sections(
            self.rod_position, 
            boron_ppm=self.boron_concentration,
            temp_feedback=reactivity_feedback,
            xenon_absorption=xenon_absorption,
            fuel_feedback=fuel_feedback
        )
        
        avg_power_unit = self.neutronics.step(dt)
        
        # 4. Шаг Ксенона
        self.poisoning.step(dt, self.neutronics.phi, self.neutronics.nu_Sigma_f)
        
        # 5. Конвертация профиля потока в профиль мощности (МВт)
        flux_profile = self.neutronics.phi
        conversion = 60.0 / 1e13 
        power_profile_MW = flux_profile * conversion
        
        # 6. Шаг теплофизики
        self.thermal.step(dt, power_profile_MW, flow_factor=flow_factor)
        
        self.time += dt
        
        return self.get_state()

    def burnup_step(self, time_hours):
        dt_seconds = time_hours * 3600.0
        flux = self.neutronics.phi
        self.fuel.step(flux, dt_seconds)
        return self.fuel.get_state()

    def get_state(self):
        n_state = self.neutronics.get_state()
        t_state = self.thermal.get_state()
        p_state = self.poisoning.get_state()
        f_state = self.fuel.get_state()
        
        return {
            'time': self.time,
            'rod_position': self.rod_position,
            'boron_ppm': self.boron_concentration,
            'total_power': torch.sum(self.neutronics.phi).item() * (60.0 / 1e13), 
            'config_name': self.config.name,
            'config_type': self.config.type_str,
            **n_state,
            **t_state,
            **p_state,
            **f_state
        }

    def set_rod_speed(self, speed):
        self.rod_speed = speed
