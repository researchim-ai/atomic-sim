"""
Модель нейтронной кинетики реактора
Использует уравнения точечной кинетики с учетом запаздывающих нейтронов
"""

import torch
import torch.nn as nn


class NeutronKinetics(nn.Module):
    """
    Модель точечной кинетики реактора с 6 группами запаздывающих нейтронов
    
    Основные уравнения:
    dn/dt = (ρ - β)/Λ * n + Σ λ_i * C_i
    dC_i/dt = β_i/Λ * n - λ_i * C_i
    
    где:
    n - плотность нейтронов
    ρ - реактивность
    β - доля запаздывающих нейтронов
    Λ - время жизни мгновенных нейтронов
    C_i - концентрация предшественников i-й группы
    λ_i - постоянная распада i-й группы
    β_i - доля i-й группы запаздывающих нейтронов
    """
    
    def __init__(self, device='cpu', dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Физические параметры для типичного теплового реактора
        # 6 групп запаздывающих нейтронов
        self.n_groups = 6
        
        # Доли запаздывающих нейтронов для каждой группы
        beta_values = torch.tensor([
            0.000215,  # группа 1
            0.001424,  # группа 2
            0.001274,  # группа 3
            0.002568,  # группа 4
            0.000748,  # группа 5
            0.000273,  # группа 6
        ], device=device, dtype=dtype)
        
        # Постоянные распада (1/с)
        lambda_values = torch.tensor([
            0.0124,   # группа 1
            0.0305,   # группа 2
            0.111,    # группа 3
            0.301,    # группа 4
            1.14,     # группа 5
            3.01,     # группа 6
        ], device=device, dtype=dtype)
        
        self.register_buffer('beta_i', beta_values)
        self.register_buffer('lambda_i', lambda_values)
        
        # Общая доля запаздывающих нейтронов
        self.register_buffer('beta', torch.sum(beta_values))
        
        # Время жизни мгновенных нейтронов (секунды)
        self.register_buffer('Lambda', torch.tensor(5e-5, device=device, dtype=dtype))
        
        # Начальная мощность (условные единицы)
        self.P0 = torch.tensor(100.0, device=device, dtype=dtype)
        
        # Состояние системы
        self.n = None  # плотность нейтронов (относительно начальной)
        self.C = None  # концентрации предшественников
        
        self.reset()
    
    def reset(self):
        """Сброс состояния реактора к начальному"""
        self.n = torch.ones(1, device=self.device, dtype=self.dtype)
        
        # Начальные концентрации предшественников (стационарное состояние)
        self.C = self.beta_i / (self.lambda_i * self.Lambda)
    
    def compute_derivatives(self, n, C, reactivity):
        """
        Вычисление производных для уравнений кинетики
        
        Args:
            n: плотность нейтронов
            C: концентрации предшественников [n_groups]
            reactivity: реактивность (в долях β)
        
        Returns:
            dn_dt: производная плотности нейтронов
            dC_dt: производные концентраций предшественников
        """
        # Реактивность в абсолютных единицах
        rho = reactivity * self.beta
        
        # Производная плотности нейтронов
        dn_dt = (rho - self.beta) / self.Lambda * n + torch.sum(self.lambda_i * C)
        
        # Производные концентраций предшественников
        dC_dt = self.beta_i / self.Lambda * n - self.lambda_i * C
        
        return dn_dt, dC_dt
    
    def step(self, dt, reactivity):
        """
        Шаг интегрирования методом Рунге-Кутты 4-го порядка
        
        Args:
            dt: шаг по времени (секунды)
            reactivity: реактивность (в долях β)
        
        Returns:
            power: текущая мощность реактора
        """
        # RK4 интегрирование
        n0 = self.n.clone()
        C0 = self.C.clone()
        
        # k1
        dn1, dC1 = self.compute_derivatives(n0, C0, reactivity)
        
        # k2
        n_mid = n0 + 0.5 * dt * dn1
        C_mid = C0 + 0.5 * dt * dC1
        dn2, dC2 = self.compute_derivatives(n_mid, C_mid, reactivity)
        
        # k3
        n_mid = n0 + 0.5 * dt * dn2
        C_mid = C0 + 0.5 * dt * dC2
        dn3, dC3 = self.compute_derivatives(n_mid, C_mid, reactivity)
        
        # k4
        n_end = n0 + dt * dn3
        C_end = C0 + dt * dC3
        dn4, dC4 = self.compute_derivatives(n_end, C_end, reactivity)
        
        # Обновление состояния
        self.n = n0 + dt / 6.0 * (dn1 + 2*dn2 + 2*dn3 + dn4)
        self.C = C0 + dt / 6.0 * (dC1 + 2*dC2 + 2*dC3 + dC4)
        
        # Ограничение снизу (нейтронная плотность не может быть отрицательной)
        self.n = torch.clamp(self.n, min=1e-10)
        self.C = torch.clamp(self.C, min=1e-10)
        
        # Мощность пропорциональна плотности нейтронов
        power = self.n * self.P0
        
        return power
    
    def get_state(self):
        """Получить текущее состояние системы"""
        return {
            'neutron_density': self.n.item(),
            'power': (self.n * self.P0).item(),
            'precursor_concentrations': self.C.cpu().numpy(),
        }

