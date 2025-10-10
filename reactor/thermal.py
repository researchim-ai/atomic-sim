"""
Тепловая модель реактора
Моделирует распределение температуры и систему охлаждения
"""

import torch
import torch.nn as nn


class ThermalModel(nn.Module):
    """
    Упрощенная тепловая модель реактора с тремя узлами:
    - Топливо
    - Оболочка
    - Теплоноситель
    
    Уравнения теплопередачи:
    dT_fuel/dt = Q/C_fuel - h1*(T_fuel - T_clad)
    dT_clad/dt = h1*(T_fuel - T_clad) - h2*(T_clad - T_coolant)
    dT_coolant/dt = h2*(T_clad - T_coolant) - (T_coolant - T_in)*flow_rate/V_coolant
    
    где:
    Q - тепловыделение от делений
    C - теплоемкость
    h - коэффициенты теплопередачи
    T_in - температура входящего теплоносителя
    """
    
    def __init__(self, device='cpu', dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Физические параметры
        
        # Теплоемкости (МДж/К)
        self.register_buffer('C_fuel', torch.tensor(200.0, device=device, dtype=dtype))
        self.register_buffer('C_clad', torch.tensor(20.0, device=device, dtype=dtype))
        self.register_buffer('C_coolant', torch.tensor(100.0, device=device, dtype=dtype))
        
        # Коэффициенты теплопередачи (МВт/К)
        self.register_buffer('h1', torch.tensor(20.0, device=device, dtype=dtype))  # топливо-оболочка
        self.register_buffer('h2', torch.tensor(30.0, device=device, dtype=dtype))  # оболочка-теплоноситель
        
        # Параметры охлаждения
        self.register_buffer('T_in', torch.tensor(280.0, device=device, dtype=dtype))  # температура входа (°C)
        self.register_buffer('h_flow', torch.tensor(100.0, device=device, dtype=dtype))  # коэффициент потока (МВт/К)
        self.register_buffer('flow_rate_nominal', torch.tensor(1.0, device=device, dtype=dtype))  # номинальный расход
        
        # Коэффициенты обратной связи по температуре (доплеровский эффект)
        # Отрицательная обратная связь для стабильности
        self.register_buffer('alpha_fuel', torch.tensor(-5e-5, device=device, dtype=dtype))  # (Δρ/β)/K
        self.register_buffer('alpha_coolant', torch.tensor(-3e-5, device=device, dtype=dtype))  # (Δρ/β)/K
        
        # Начальные температуры (°C) - приблизительное стационарное состояние
        # Для 100 МВт при T_in=280°C и h_flow=100:
        # h_flow*(T_coolant - T_in) ≈ 100 МВт -> T_coolant ≈ 281°C
        self.T_fuel_0 = torch.tensor(287.0, device=device, dtype=dtype)
        self.T_clad_0 = torch.tensor(284.0, device=device, dtype=dtype)
        self.T_coolant_0 = torch.tensor(281.0, device=device, dtype=dtype)
        
        # Состояние
        self.T_fuel = None
        self.T_clad = None
        self.T_coolant = None
        self.flow_rate = None
        
        self.reset()
    
    def reset(self):
        """Сброс к начальному состоянию"""
        self.T_fuel = self.T_fuel_0.clone()
        self.T_clad = self.T_clad_0.clone()
        self.T_coolant = self.T_coolant_0.clone()
        self.flow_rate = self.flow_rate_nominal.clone()
    
    def compute_derivatives(self, T_fuel, T_clad, T_coolant, power, flow_rate):
        """
        Вычисление производных температур
        
        Args:
            T_fuel: температура топлива (°C)
            T_clad: температура оболочки (°C)
            T_coolant: температура теплоносителя (°C)
            power: мощность реактора (МВт)
            flow_rate: относительный расход теплоносителя
        
        Returns:
            производные температур
        """
        # Тепловыделение в топливе (МВт)
        Q = power
        
        # Производная температуры топлива
        dT_fuel = Q / self.C_fuel - self.h1 * (T_fuel - T_clad) / self.C_fuel
        
        # Производная температуры оболочки
        dT_clad = (self.h1 * (T_fuel - T_clad) - self.h2 * (T_clad - T_coolant)) / self.C_clad
        
        # Производная температуры теплоносителя
        dT_coolant = (self.h2 * (T_clad - T_coolant) - 
                     self.h_flow * (T_coolant - self.T_in) * flow_rate) / self.C_coolant
        
        return dT_fuel, dT_clad, dT_coolant
    
    def step(self, dt, power, flow_rate=None):
        """
        Шаг интегрирования
        
        Args:
            dt: шаг по времени (секунды)
            power: мощность реактора (МВт)
            flow_rate: относительный расход теплоносителя (опционально)
        
        Returns:
            reactivity_feedback: обратная связь по реактивности
        """
        if flow_rate is not None:
            self.flow_rate = torch.tensor(flow_rate, device=self.device, dtype=self.dtype)
        
        # RK4 интегрирование
        T_f0 = self.T_fuel.clone()
        T_c0 = self.T_clad.clone()
        T_co0 = self.T_coolant.clone()
        
        # k1
        dTf1, dTc1, dTco1 = self.compute_derivatives(T_f0, T_c0, T_co0, power, self.flow_rate)
        
        # k2
        dTf2, dTc2, dTco2 = self.compute_derivatives(
            T_f0 + 0.5*dt*dTf1, T_c0 + 0.5*dt*dTc1, T_co0 + 0.5*dt*dTco1,
            power, self.flow_rate
        )
        
        # k3
        dTf3, dTc3, dTco3 = self.compute_derivatives(
            T_f0 + 0.5*dt*dTf2, T_c0 + 0.5*dt*dTc2, T_co0 + 0.5*dt*dTco2,
            power, self.flow_rate
        )
        
        # k4
        dTf4, dTc4, dTco4 = self.compute_derivatives(
            T_f0 + dt*dTf3, T_c0 + dt*dTc3, T_co0 + dt*dTco3,
            power, self.flow_rate
        )
        
        # Обновление состояния
        self.T_fuel = T_f0 + dt/6.0 * (dTf1 + 2*dTf2 + 2*dTf3 + dTf4)
        self.T_clad = T_c0 + dt/6.0 * (dTc1 + 2*dTc2 + 2*dTc3 + dTc4)
        self.T_coolant = T_co0 + dt/6.0 * (dTco1 + 2*dTco2 + 2*dTco3 + dTco4)
        
        # Вычисление обратной связи по реактивности
        delta_T_fuel = self.T_fuel - self.T_fuel_0
        delta_T_coolant = self.T_coolant - self.T_coolant_0
        
        reactivity_feedback = (self.alpha_fuel * delta_T_fuel + 
                              self.alpha_coolant * delta_T_coolant)
        
        return reactivity_feedback
    
    def get_state(self):
        """Получить текущее состояние"""
        return {
            'T_fuel': self.T_fuel.item(),
            'T_clad': self.T_clad.item(),
            'T_coolant': self.T_coolant.item(),
            'flow_rate': self.flow_rate.item(),
        }
    
    def check_safety_limits(self):
        """
        Проверка предельных значений безопасности
        
        Returns:
            dict с флагами нарушений
        """
        return {
            'fuel_overheat': self.T_fuel.item() > 1200.0,  # температура плавления топлива
            'coolant_boiling': self.T_coolant.item() > 320.0,  # кипение воды под давлением
            'safe': self.T_fuel.item() <= 1200.0 and self.T_coolant.item() <= 320.0,
        }

