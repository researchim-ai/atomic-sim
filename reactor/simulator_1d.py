"""
1D Reactor Simulator (Spatial Kinetics)
Объединяет 1D нейтронную кинетику, 1D теплогидравлику и 1D ксеноновые колебания.
"""

import torch
import numpy as np
from .neutronics_1d import Neutronics1D
from .thermal_1d import ThermalModel1D
from .poisoning_1d import XenonIodine1D

class ReactorSimulator1D:
    def __init__(self, num_nodes=50, device='cpu', dtype=torch.float64):
        self.device = device
        self.dtype = dtype
        self.num_nodes = num_nodes
        
        self.neutronics = Neutronics1D(num_nodes=num_nodes, device=device, dtype=dtype)
        self.thermal = ThermalModel1D(num_nodes=num_nodes, device=device, dtype=dtype)
        self.poisoning = XenonIodine1D(num_nodes=num_nodes, device=device, dtype=dtype)
        
        # Инициализация равновесного ксенона под начальный поток
        self.poisoning.reset(flux_profile=self.neutronics.phi)
        
        self.time = 0.0
        self.rod_position = 0.0 # 0.0 (out) to 1.0 (in)
        self.rod_speed = 0.0 # units/sec
        
        # Для записи истории
        self.history = []
        
    def step(self, dt=0.01):
        # 1. Управление стержнями
        self.rod_position += self.rod_speed * dt
        self.rod_position = max(0.0, min(1.0, self.rod_position))
        
        # 2. Получение обратной связи от теплофизики и ксенона
        # reactivity_feedback: вектор (N,)
        reactivity_feedback = self.thermal.compute_reactivity_feedback()
        
        # xenon_absorption: вектор (N,) Sigma_Xe
        xenon_absorption = self.poisoning.get_absorption_cross_section()
        
        # 3. Шаг нейтроники
        # Передаем позицию стержней, темп. связь и ксенон
        self.neutronics.update_cross_sections(
            self.rod_position, 
            temp_feedback=reactivity_feedback,
            xenon_absorption=xenon_absorption
        )
        
        # Делаем шаг нейтроники
        avg_power_unit = self.neutronics.step(dt)
        
        # 4. Шаг Ксенона
        # Используем текущий поток и сечение деления
        self.poisoning.step(dt, self.neutronics.phi, self.neutronics.nu_Sigma_f)
        
        # 5. Конвертация профиля потока в профиль мощности (МВт)
        flux_profile = self.neutronics.phi
        conversion = 60.0 / 1e13 # 60 MW per node at 1e13 flux
        power_profile_MW = flux_profile * conversion
        
        # 6. Шаг теплофизики
        self.thermal.step(dt, power_profile_MW)
        
        self.time += dt
        
        return self.get_state()

    def get_state(self):
        n_state = self.neutronics.get_state()
        t_state = self.thermal.get_state()
        p_state = self.poisoning.get_state()
        
        return {
            'time': self.time,
            'rod_position': self.rod_position,
            'total_power': torch.sum(self.neutronics.phi).item() * (60.0 / 1e13), 
            **n_state,
            **t_state,
            **p_state
        }

    def set_rod_speed(self, speed):
        """Скорость движения стержней (доли полной высоты в секунду)"""
        self.rod_speed = speed
