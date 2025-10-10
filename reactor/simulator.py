"""
Главный симулятор атомного реактора
Объединяет нейтронную физику, тепловую модель и систему управления
"""

import torch
import numpy as np
from .neutronics import NeutronKinetics
from .thermal import ThermalModel
from .control import ControlSystem


class ReactorSimulator:
    """
    Полный симулятор атомного реактора
    
    Интегрирует:
    - Нейтронную кинетику
    - Тепловую модель
    - Систему управления
    - Обратные связи
    """
    
    def __init__(self, device='cpu', dtype=torch.float64):
        """
        Инициализация симулятора
        
        Args:
            device: устройство для вычислений ('cpu' или 'cuda')
            dtype: тип данных для вычислений
        """
        self.device = device
        self.dtype = dtype
        
        # Создание подсистем
        self.neutronics = NeutronKinetics(device=device, dtype=dtype)
        self.thermal = ThermalModel(device=device, dtype=dtype)
        self.control = ControlSystem(device=device, dtype=dtype)
        
        # Время симуляции
        self.time = 0.0
        self.dt = 0.001  # шаг по времени (секунды) - уменьшен для стабильности жесткой системы
        
        # История для записи данных
        self.history = {
            'time': [],
            'power': [],
            'reactivity': [],
            'T_fuel': [],
            'T_coolant': [],
            'rod_position': [],
        }
        
        self.recording = False
    
    def reset(self):
        """Сброс симулятора к начальному состоянию"""
        self.neutronics.reset()
        self.thermal.reset()
        self.control.reset()
        
        self.time = 0.0
        self.history = {
            'time': [],
            'power': [],
            'reactivity': [],
            'T_fuel': [],
            'T_coolant': [],
            'rod_position': [],
        }
    
    def start_recording(self):
        """Начать запись истории"""
        self.recording = True
        self.history = {
            'time': [],
            'power': [],
            'reactivity': [],
            'T_fuel': [],
            'T_coolant': [],
            'rod_position': [],
        }
    
    def stop_recording(self):
        """Остановить запись истории"""
        self.recording = False
    
    def step(self, manual_rod_direction=None, external_reactivity=0.0):
        """
        Выполнить один шаг симуляции
        
        Args:
            manual_rod_direction: ручное управление стержнями (-1, 0, +1)
            external_reactivity: внешняя реактивность (в долях β)
        
        Returns:
            state: текущее состояние системы
        """
        # Установить внешнюю реактивность
        self.control.external_reactivity = torch.tensor(
            external_reactivity, device=self.device, dtype=self.dtype
        )
        
        # Получить текущую мощность
        current_power = (self.neutronics.n * self.neutronics.P0).item()
        
        # Шаг системы управления (получить реактивность от стержней)
        rod_reactivity = self.control.step(self.dt, current_power, manual_rod_direction)
        
        # Шаг тепловой модели (получить температурную обратную связь)
        temp_reactivity = self.thermal.step(self.dt, current_power)
        
        # Полная реактивность
        total_reactivity = rod_reactivity + temp_reactivity
        
        # Шаг нейтронной кинетики
        power = self.neutronics.step(self.dt, total_reactivity)
        
        # Обновить время
        self.time += self.dt
        
        # Записать в историю
        if self.recording:
            self.history['time'].append(self.time)
            self.history['power'].append(power.item())
            self.history['reactivity'].append(total_reactivity.item())
            self.history['T_fuel'].append(self.thermal.T_fuel.item())
            self.history['T_coolant'].append(self.thermal.T_coolant.item())
            self.history['rod_position'].append(self.control.rod_positions.mean().item())
        
        # Получить состояние
        state = self.get_state()
        
        return state
    
    def run(self, duration, callback=None, callback_interval=1.0):
        """
        Запустить симуляцию на заданное время
        
        Args:
            duration: длительность симуляции (секунды)
            callback: функция обратного вызова для мониторинга
            callback_interval: интервал вызова callback (секунды)
        
        Returns:
            history: история симуляции
        """
        self.start_recording()
        
        steps = int(duration / self.dt)
        last_callback_time = 0.0
        scram_triggered = False
        
        for i in range(steps):
            state = self.step()
            
            # Вызов callback
            if callback is not None and (self.time - last_callback_time) >= callback_interval:
                callback(state)
                last_callback_time = self.time
            
            # Проверка безопасности (SCRAM только один раз)
            safety = self.thermal.check_safety_limits()
            if not safety['safe'] and not scram_triggered:
                print(f"\n⚠️  ПРЕДУПРЕЖДЕНИЕ: Нарушение пределов безопасности на t={self.time:.2f}s")
                if safety['fuel_overheat']:
                    print(f"   Перегрев топлива: {self.thermal.T_fuel.item():.1f}°C")
                if safety['coolant_boiling']:
                    print(f"   Кипение теплоносителя: {self.thermal.T_coolant.item():.1f}°C")
                
                # Автоматический SCRAM при критических условиях
                self.control.scram()
                print("   >>> Выполнен аварийный SCRAM <<<\n")
                scram_triggered = True
        
        self.stop_recording()
        
        return self.get_history()
    
    def get_state(self):
        """
        Получить текущее состояние всех систем
        
        Returns:
            dict с полным состоянием реактора
        """
        neutron_state = self.neutronics.get_state()
        thermal_state = self.thermal.get_state()
        control_state = self.control.get_state()
        safety_state = self.thermal.check_safety_limits()
        
        return {
            'time': self.time,
            **neutron_state,
            **thermal_state,
            **control_state,
            **safety_state,
        }
    
    def get_history(self):
        """
        Получить историю симуляции в виде numpy массивов
        
        Returns:
            dict с историей параметров
        """
        return {
            key: np.array(values) for key, values in self.history.items()
        }
    
    def print_state(self):
        """Вывести текущее состояние в консоль"""
        state = self.get_state()
        
        print(f"\n{'='*60}")
        print(f"Время: {state['time']:.2f} с")
        print(f"{'='*60}")
        print(f"Нейтронная физика:")
        print(f"  Мощность:           {state['power']:.2f} МВт")
        print(f"  Плотность нейтронов: {state['neutron_density']:.4f}")
        print(f"\nТемпературы:")
        print(f"  Топливо:            {state['T_fuel']:.1f} °C")
        print(f"  Оболочка:           {state['T_clad']:.1f} °C")
        print(f"  Теплоноситель:      {state['T_coolant']:.1f} °C")
        print(f"\nУправление:")
        print(f"  Реактивность (стержни): {state['rod_reactivity']:.4f} β")
        print(f"  Средняя позиция стержней: {state['avg_rod_position']*100:.1f}%")
        print(f"  Авторегулирование:  {'ВКЛ' if state['auto_control'] else 'ВЫКЛ'}")
        if state['auto_control']:
            print(f"  Целевая мощность:   {state['target_power']:.2f} МВт")
        print(f"\nБезопасность:")
        print(f"  Статус:             {'✓ НОРМА' if state['safe'] else '✗ АВАРИЯ'}")
        if state['fuel_overheat']:
            print(f"  ⚠️  ПЕРЕГРЕВ ТОПЛИВА!")
        if state['coolant_boiling']:
            print(f"  ⚠️  КИПЕНИЕ ТЕПЛОНОСИТЕЛЯ!")
        print(f"{'='*60}\n")
    
    # Удобные методы управления
    
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
    
    def insert_rods(self):
        """Вставить управляющие стержни (снизить мощность)"""
        return -1
    
    def withdraw_rods(self):
        """Извлечь управляющие стержни (увеличить мощность)"""
        return 1
    
    def hold_rods(self):
        """Удерживать стержни в текущем положении"""
        return 0

