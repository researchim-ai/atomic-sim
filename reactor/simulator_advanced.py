"""
Расширенный симулятор с поддержкой отравления ксеноном и событий
Для генерации реалистичных датасетов
"""

import torch
import numpy as np
from .neutronics import NeutronKinetics
from .thermal import ThermalModel
from .control import ControlSystem
from .poisoning import XenonIodineKinetics
from .events import EventManager


class ReactorSimulatorAdvanced:
    """
    Расширенный симулятор ядерного реактора с:
    - Отравлением ксеноном/йодом
    - Системой событий
    - Расширенным логированием
    """
    
    def __init__(self, device='cpu', dtype=torch.float64, enable_xenon=True):
        """
        Args:
            device: устройство для вычислений
            dtype: тип данных
            enable_xenon: включить модель отравления ксеноном
        """
        self.device = device
        self.dtype = dtype
        self.enable_xenon = enable_xenon
        
        # Создание подсистем
        self.neutronics = NeutronKinetics(device=device, dtype=dtype)
        self.thermal = ThermalModel(device=device, dtype=dtype)
        self.control = ControlSystem(device=device, dtype=dtype)
        
        if enable_xenon:
            self.xenon = XenonIodineKinetics(device=device, dtype=dtype)
        else:
            self.xenon = None
        
        # Менеджер событий
        self.event_manager = EventManager()
        
        # Время
        self.time = 0.0
        self.dt = 0.001
        
        # История с расширенной информацией
        self.history = {
            'time': [],
            'power': [],
            'reactivity': [],
            'reactivity_rods': [],
            'reactivity_temp': [],
            'reactivity_xenon': [],
            'T_fuel': [],
            'T_coolant': [],
            'rod_position': [],
        }
        
        if enable_xenon:
            self.history['I_concentration'] = []
            self.history['Xe_concentration'] = []
        
        self.recording = False
    
    def reset(self, equilibrium_xenon=False):
        """
        Сброс симулятора
        
        Args:
            equilibrium_xenon: начать с равновесной концентрацией Xe
        """
        self.neutronics.reset()
        self.thermal.reset()
        self.control.reset()
        
        if self.xenon:
            if equilibrium_xenon:
                self.xenon.reset_equilibrium(100.0)
            else:
                self.xenon.reset()
        
        self.time = 0.0
        self.event_manager = EventManager()
        
        self.history = {k: [] for k in self.history.keys()}
        self.recording = False
    
    def start_recording(self):
        """Начать запись истории"""
        self.recording = True
    
    def stop_recording(self):
        """Остановить запись истории"""
        self.recording = False
    
    def step(self, manual_rod_direction=None):
        """Выполнить один шаг симуляции"""
        # Проверить события
        self.event_manager.check_events(self.time, self)
        
        # Получить текущую мощность
        current_power = (self.neutronics.n * self.neutronics.P0).item()
        
        # Шаг системы управления
        rod_reactivity = self.control.step(self.dt, current_power, manual_rod_direction)
        
        # Шаг тепловой модели
        temp_reactivity = self.thermal.step(self.dt, current_power)
        
        # Шаг отравления ксеноном
        xenon_reactivity = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if self.xenon:
            xenon_reactivity = self.xenon.step(self.dt, current_power)
        
        # Полная реактивность
        total_reactivity = rod_reactivity + temp_reactivity + xenon_reactivity
        
        # Шаг нейтронной кинетики
        power = self.neutronics.step(self.dt, total_reactivity)
        
        # Обновить время
        self.time += self.dt
        
        # Записать в историю
        if self.recording:
            self.history['time'].append(self.time)
            self.history['power'].append(power.item())
            self.history['reactivity'].append(total_reactivity.item())
            self.history['reactivity_rods'].append(rod_reactivity.item())
            self.history['reactivity_temp'].append(temp_reactivity.item())
            self.history['reactivity_xenon'].append(xenon_reactivity.item())
            self.history['T_fuel'].append(self.thermal.T_fuel.item())
            self.history['T_coolant'].append(self.thermal.T_coolant.item())
            self.history['rod_position'].append(self.control.rod_positions.mean().item())
            
            if self.xenon:
                self.history['I_concentration'].append(self.xenon.I.item())
                self.history['Xe_concentration'].append(self.xenon.Xe.item())
        
        # Получить состояние
        state = self.get_state()
        
        return state
    
    def run(self, duration, callback=None, callback_interval=1.0, show_progress=True):
        """
        Запустить симуляцию
        
        Args:
            duration: длительность (секунды)
            callback: функция обратного вызова
            callback_interval: интервал callback
            show_progress: показывать прогресс-бар
        
        Returns:
            history
        """
        from tqdm import trange
        
        self.start_recording()
        
        steps = int(duration / self.dt)
        last_callback_time = 0.0
        scram_triggered = False
        
        iterator = trange(steps, desc="Simulation", disable=not show_progress)
        
        for i in iterator:
            state = self.step()
            
            # Обновить прогресс-бар
            if show_progress and i % 1000 == 0:
                iterator.set_postfix({
                    'P': f"{state['power']:.1f}MW",
                    'T': f"{state['T_fuel']:.0f}C"
                })
            
            # Callback
            if callback and (self.time - last_callback_time) >= callback_interval:
                callback(state)
                last_callback_time = self.time
            
            # Проверка безопасности
            safety = self.thermal.check_safety_limits()
            if not safety['safe'] and not scram_triggered:
                print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Нарушение безопасности на t={self.time:.2f}s")
                if safety['fuel_overheat']:
                    print(f"   Перегрев топлива: {self.thermal.T_fuel.item():.1f}°C")
                if safety['coolant_boiling']:
                    print(f"   Кипение теплоносителя: {self.thermal.T_coolant.item():.1f}°C")
                
                self.control.scram()
                print("   >>> Выполнен аварийный SCRAM <<<\n")
                scram_triggered = True
        
        self.stop_recording()
        
        return self.get_history()
    
    def get_state(self):
        """Получить текущее состояние всех систем"""
        neutron_state = self.neutronics.get_state()
        thermal_state = self.thermal.get_state()
        control_state = self.control.get_state()
        safety_state = self.thermal.check_safety_limits()
        
        state = {
            'time': self.time,
            **neutron_state,
            **thermal_state,
            **control_state,
            **safety_state,
        }
        
        if self.xenon:
            state.update(self.xenon.get_state())
        
        return state
    
    def get_history(self):
        """Получить историю в виде numpy массивов"""
        return {
            key: np.array(values) for key, values in self.history.items()
        }
    
    def set_auto_control(self, enabled, target_power=None):
        """Включить/выключить автоматическое управление"""
        self.control.set_auto_control(enabled, target_power)
    
    def scram(self):
        """Аварийная остановка"""
        self.control.scram()
    
    def set_coolant_flow(self, flow_rate):
        """Установить расход теплоносителя"""
        self.thermal.flow_rate = torch.tensor(
            flow_rate, device=self.device, dtype=self.dtype
        )

