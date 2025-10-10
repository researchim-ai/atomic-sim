"""
Система управления реактором
Включает управляющие стержни и автоматическое регулирование мощности
"""

import torch
import torch.nn as nn


class ControlSystem(nn.Module):
    """
    Система управления реактором с управляющими стержнями
    и автоматическим регулятором мощности
    
    Управляющие стержни влияют на реактивность:
    ρ = ρ_external + ρ_control_rods + ρ_temperature_feedback
    """
    
    def __init__(self, device='cpu', dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # Параметры управляющих стержней
        self.n_rods = 10  # количество стержней
        
        # Реактивность одного полностью вставленного стержня (в долях β)
        self.register_buffer('rod_worth', torch.tensor(-2.0, device=device, dtype=dtype))
        
        # Позиции стержней (0 = полностью вставлен, 1 = полностью извлечен)
        self.rod_positions = None
        
        # Скорость движения стержней (относительные единицы/секунду)
        self.register_buffer('rod_speed', torch.tensor(0.10, device=device, dtype=dtype))
        
        # ПИД-регулятор
        self.use_auto_control = False
        self.target_power = torch.tensor(100.0, device=device, dtype=dtype)
        
        # Параметры ПИД (настроены для стабильного управления)
        self.register_buffer('Kp', torch.tensor(0.02, device=device, dtype=dtype))  # пропорциональный
        self.register_buffer('Ki', torch.tensor(0.002, device=device, dtype=dtype))  # интегральный
        self.register_buffer('Kd', torch.tensor(0.01, device=device, dtype=dtype))  # дифференциальный
        
        self.integral_error = torch.tensor(0.0, device=device, dtype=dtype)
        self.previous_error = torch.tensor(0.0, device=device, dtype=dtype)
        
        # Внешняя реактивность (например, от выгорания топлива)
        self.external_reactivity = torch.tensor(0.0, device=device, dtype=dtype)
        
        self.reset()
    
    def reset(self):
        """Сброс к начальному состоянию"""
        # Начальная конфигурация: стержни полностью извлечены для критичности
        # При rod_position = 1.0, реактивность = 0 (точная критичность)
        # Это ЕДИНСТВЕННАЯ позиция, дающая стационарное состояние при n=1.0
        # Стержни можно вставлять для снижения мощности, но не извлекать дальше
        # Для увеличения мощности выше 100 МВт потребуется внешняя положительная реактивность
        self.rod_positions = torch.ones(self.n_rods, device=self.device, dtype=self.dtype) * 1.0
        
        self.integral_error = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.previous_error = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.external_reactivity = torch.tensor(0.0, device=self.device, dtype=self.dtype)
    
    def set_rod_position(self, rod_index, position):
        """
        Установить позицию конкретного стержня
        
        Args:
            rod_index: индекс стержня (0 до n_rods-1)
            position: целевая позиция (0 до 1)
        """
        position = max(0.0, min(1.0, position))
        self.rod_positions[rod_index] = torch.tensor(position, device=self.device, dtype=self.dtype)
    
    def move_rods(self, direction, dt):
        """
        Переместить все стержни
        
        Args:
            direction: направление (-1 = вставить, +1 = извлечь, 0 = не двигать)
            dt: шаг по времени
        """
        if direction == 0:
            return
        
        delta = direction * self.rod_speed * dt
        self.rod_positions = torch.clamp(self.rod_positions + delta, 0.0, 1.0)
    
    def compute_rod_reactivity(self):
        """
        Вычислить реактивность от управляющих стержней
        
        Returns:
            реактивность в долях β
        """
        # Суммарная реактивность от всех стержней
        # Полностью извлеченный стержень (position=1) не вносит реактивности
        # Полностью вставленный стержень (position=0) вносит rod_worth
        total_reactivity = torch.sum((1.0 - self.rod_positions) * self.rod_worth)
        
        return total_reactivity
    
    def auto_control_step(self, current_power, dt):
        """
        Шаг автоматического ПИД-регулятора
        
        Args:
            current_power: текущая мощность
            dt: шаг по времени
        
        Returns:
            направление движения стержней
        """
        if not self.use_auto_control:
            return 0
        
        # Ошибка регулирования
        error = self.target_power - current_power
        
        # Интегральная составляющая
        self.integral_error += error * dt
        # Ограничение интегральной составляющей (anti-windup)
        self.integral_error = torch.clamp(self.integral_error, -100.0, 100.0)
        
        # Дифференциальная составляющая
        derivative = (error - self.previous_error) / dt if dt > 0 else torch.tensor(0.0)
        
        # ПИД-выход
        control_signal = (self.Kp * error + 
                         self.Ki * self.integral_error + 
                         self.Kd * derivative)
        
        self.previous_error = error
        
        # Преобразование управляющего сигнала в направление движения стержней
        # Положительная ошибка (нужно больше мощности) -> извлечь стержни (+1)
        # Отрицательная ошибка (нужно меньше мощности) -> вставить стержни (-1)
        
        threshold = 0.1  # порог для движения стержней (низкий для чувствительного управления)
        if control_signal > threshold:
            return 1  # извлечь
        elif control_signal < -threshold:
            return -1  # вставить
        else:
            return 0  # не двигать
    
    def step(self, dt, current_power, manual_direction=None):
        """
        Шаг системы управления
        
        Args:
            dt: шаг по времени
            current_power: текущая мощность реактора
            manual_direction: ручное управление стержнями (опционально)
        
        Returns:
            total_reactivity: полная реактивность
        """
        # Определить направление движения стержней
        if manual_direction is not None:
            direction = manual_direction
        else:
            direction = self.auto_control_step(current_power, dt)
        
        # Переместить стержни
        self.move_rods(direction, dt)
        
        # Вычислить реактивность от стержней
        rod_reactivity = self.compute_rod_reactivity()
        
        # Полная реактивность
        total_reactivity = self.external_reactivity + rod_reactivity
        
        return total_reactivity
    
    def scram(self):
        """
        Аварийная остановка реактора (SCRAM)
        Все стержни мгновенно полностью вставляются
        """
        self.rod_positions = torch.zeros(self.n_rods, device=self.device, dtype=self.dtype)
    
    def set_auto_control(self, enabled, target_power=None):
        """
        Включить/выключить автоматическое управление
        
        Args:
            enabled: включить (True) или выключить (False)
            target_power: целевая мощность (опционально)
        """
        self.use_auto_control = enabled
        if target_power is not None:
            self.target_power = torch.tensor(target_power, device=self.device, dtype=self.dtype)
        
        # Сброс интегральной составляющей при переключении
        if enabled:
            self.integral_error = torch.tensor(0.0, device=self.device, dtype=self.dtype)
            self.previous_error = torch.tensor(0.0, device=self.device, dtype=self.dtype)
    
    def get_state(self):
        """Получить текущее состояние"""
        return {
            'rod_positions': self.rod_positions.cpu().numpy(),
            'avg_rod_position': self.rod_positions.mean().item(),
            'rod_reactivity': self.compute_rod_reactivity().item(),
            'external_reactivity': self.external_reactivity.item(),
            'auto_control': self.use_auto_control,
            'target_power': self.target_power.item() if self.use_auto_control else None,
        }

