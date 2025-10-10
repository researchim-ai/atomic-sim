"""
Модель отравления ксеноном и йодом
Моделирует динамику Xe-135 и I-135, критичную для реализма реактора
"""

import torch
import torch.nn as nn


class XenonIodineKinetics(nn.Module):
    """
    Кинетика I-135 и Xe-135 (основной поглотитель нейтронов)
    
    Уравнения:
    dI/dt = γ_I * Σ_f * φ - λ_I * I
    dXe/dt = γ_Xe * Σ_f * φ + λ_I * I - λ_Xe * Xe - σ_a,Xe * φ * Xe
    
    где:
    φ - нейтронный поток (пропорционален мощности)
    γ - выходы на деление
    λ - константы распада
    σ_a,Xe - эффективное сечение поглощения Xe-135
    """
    
    def __init__(self, device='cpu', dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Физические константы
        # I-135: период полураспада ~6.57 часов
        self.register_buffer('lambda_I', torch.tensor(2.9e-5, device=device, dtype=dtype))  # 1/с
        
        # Xe-135: период полураспада ~9.14 часов
        self.register_buffer('lambda_Xe', torch.tensor(2.1e-5, device=device, dtype=dtype))  # 1/с
        
        # Эффективное микроскопическое сечение поглощения Xe-135 (см²)
        # Нормализовано на номинальный поток
        self.register_buffer('sigma_Xe', torch.tensor(2.6e-18, device=device, dtype=dtype))
        
        # Независимые выходы на деление
        self.register_buffer('gamma_I', torch.tensor(0.061, device=device, dtype=dtype))   # I-135
        self.register_buffer('gamma_Xe', torch.tensor(0.003, device=device, dtype=dtype))  # Xe-135
        
        # Коэффициент реактивности от Xe (настраивается)
        # Отрицательный - Xe поглощает нейтроны
        self.register_buffer('alpha_xe', torch.tensor(-8e-6, device=device, dtype=dtype))  # (Δρ/β) на единицу концентрации
        
        # Нормировка потока на номинальную мощность
        self.P_nominal = torch.tensor(100.0, device=device, dtype=dtype)  # МВт
        
        # Состояние
        self.I = None   # концентрация I-135 (относительные единицы)
        self.Xe = None  # концентрация Xe-135 (относительные единицы)
        
        self.reset()
    
    def reset(self):
        """Сброс к начальному состоянию"""
        # Начинаем с нулевых концентраций (свежее топливо)
        self.I = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.Xe = torch.tensor(0.0, device=self.device, dtype=self.dtype)
    
    def reset_equilibrium(self, power):
        """
        Установить равновесные концентрации для заданной мощности
        
        Args:
            power: мощность реактора (МВт)
        """
        phi = power / self.P_nominal
        
        # Равновесные концентрации при постоянной мощности
        self.I = self.gamma_I * phi / self.lambda_I
        
        # Для Xe: gamma_Xe*phi + lambda_I*I - lambda_Xe*Xe - sigma_Xe*phi*Xe = 0
        # Квадратное уравнение относительно Xe
        a = self.sigma_Xe * phi
        b = self.lambda_Xe
        c = -(self.gamma_Xe * phi + self.lambda_I * self.I)
        
        if a > 1e-10:
            # Физический корень (положительный)
            discriminant = b**2 - 4*a*c
            self.Xe = (-b + torch.sqrt(discriminant)) / (2*a)
        else:
            # При низкой мощности
            self.Xe = -(c / b) if b > 1e-10 else torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        self.Xe = torch.clamp(self.Xe, min=0.0)
    
    def compute_derivatives(self, I, Xe, power):
        """
        Вычислить производные концентраций
        
        Args:
            I: концентрация йода
            Xe: концентрация ксенона
            power: мощность реактора (МВт)
        
        Returns:
            dI_dt, dXe_dt
        """
        # Нормализованный поток
        phi = power / self.P_nominal
        
        # Производная I-135
        dI_dt = self.gamma_I * phi - self.lambda_I * I
        
        # Производная Xe-135
        # Источники: прямое образование + распад I
        # Стоки: распад Xe + выгорание в потоке
        dXe_dt = (self.gamma_Xe * phi + 
                 self.lambda_I * I - 
                 self.lambda_Xe * Xe - 
                 self.sigma_Xe * phi * Xe)
        
        return dI_dt, dXe_dt
    
    def step(self, dt, power, method='rk4'):
        """
        Шаг интегрирования
        
        Args:
            dt: шаг по времени (секунды)
            power: текущая мощность реактора (МВт)
            method: метод интегрирования ('euler' или 'rk4')
        
        Returns:
            reactivity_feedback: вклад в реактивность от Xe
        """
        if method == 'euler':
            # Простой явный Эйлер
            dI_dt, dXe_dt = self.compute_derivatives(self.I, self.Xe, power)
            self.I = torch.clamp(self.I + dt * dI_dt, min=0.0)
            self.Xe = torch.clamp(self.Xe + dt * dXe_dt, min=0.0)
            
        else:  # rk4
            # Рунге-Кутта 4-го порядка
            I0, Xe0 = self.I.clone(), self.Xe.clone()
            
            # k1
            dI1, dXe1 = self.compute_derivatives(I0, Xe0, power)
            
            # k2
            I_mid = I0 + 0.5 * dt * dI1
            Xe_mid = Xe0 + 0.5 * dt * dXe1
            dI2, dXe2 = self.compute_derivatives(I_mid, Xe_mid, power)
            
            # k3
            I_mid = I0 + 0.5 * dt * dI2
            Xe_mid = Xe0 + 0.5 * dt * dXe2
            dI3, dXe3 = self.compute_derivatives(I_mid, Xe_mid, power)
            
            # k4
            I_end = I0 + dt * dI3
            Xe_end = Xe0 + dt * dXe3
            dI4, dXe4 = self.compute_derivatives(I_end, Xe_end, power)
            
            # Финальное обновление
            self.I = torch.clamp(I0 + dt / 6.0 * (dI1 + 2*dI2 + 2*dI3 + dI4), min=0.0)
            self.Xe = torch.clamp(Xe0 + dt / 6.0 * (dXe1 + 2*dXe2 + 2*dXe3 + dXe4), min=0.0)
        
        # Вклад в реактивность (отрицательный - Xe поглощает)
        reactivity_feedback = self.alpha_xe * self.Xe
        
        return reactivity_feedback
    
    def get_state(self):
        """Получить текущее состояние"""
        return {
            'I_concentration': self.I.item(),
            'Xe_concentration': self.Xe.item(),
            'Xe_reactivity': (self.alpha_xe * self.Xe).item(),
        }

