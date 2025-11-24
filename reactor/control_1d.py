"""
Automatic Control System (1D)
Реализует алгоритмы управления мощностью и формой поля (Axial Offset)
используя управляющие стержни (Rods) и борное регулирование (Boron).
"""

import torch
import torch.nn as nn

class ControlSystem1D(nn.Module):
    """
    Система управления реактором.
    
    Цели:
    1. Поддерживать заданную мощность (Power Control) с помощью стержней.
    2. Поддерживать Axial Offset (Ao) около 0, используя бор для возврата стержней в оптимальную позицию.
       (Стратегия: Rods контролируют мощность, Boron контролирует положение Rods).
    """
    
    def __init__(self, device='cpu', dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # === Настройки PID регулятора мощности (Rods) ===
        self.target_power = 3000.0 # MW
        # Уменьшаем коэффициенты для стабильности
        self.kp_p = 0.00002  # Proportional gain
        self.ki_p = 0.000005 # Integral gain
        self.kd_p = 0.0001  # Derivative gain
        
        self.integral_error_p = 0.0
        self.prev_error_p = 0.0
        
        # === Настройки контроля Бора (Boron Shim) ===
        # Цель: держать стержни в позиции inserted_ref (например, 20% внутри)
        # Если стержни глубже -> уменьшить бор (dilute) -> стержни пойдут вверх
        # Если стержни выше -> увеличить бор (borate) -> стержни пойдут вниз
        self.rod_ref_pos = 0.2 # Оптимальная позиция стержней (20%)
        self.boron_rate = 0.1 # ppm/sec (скорость изменения концентрации) - уменьшена с 0.5
        self.boron_deadband = 0.05 # Зона нечувствительности (5% позиции стержней)
        
        self.auto_power = False
        self.auto_boron = False
        
    def step(self, dt, current_power, current_rod_pos, current_boron):
        """
        Вычисляет управляющие воздействия.
        
        Returns:
            rod_speed: скорость движения стержней (-1.0 .. 1.0 units/sec)
            boron_change: изменение концентрации бора (ppm/sec)
        """
        rod_speed = 0.0
        boron_change = 0.0
        
        # 1. Контроль мощности (Стержни)
        if self.auto_power:
            error = self.target_power - current_power
            self.integral_error_p += error * dt
            derivative = (error - self.prev_error_p) / dt if dt > 0 else 0.0
            
            # PID output = desired reactivity change speed -> mapped to rod speed
            # Output > 0 means we need MORE power -> withdraw rods (speed < 0)
            # Output < 0 means we need LESS power -> insert rods (speed > 0)
            pid_out = self.kp_p * error + self.ki_p * self.integral_error_p + self.kd_p * derivative
            
            # PID выдает желаемую скорость изменения реактивности.
            # Преобразуем в скорость стержней.
            # withdraw (power up) -> rod_speed negative (pos decreases toward 0)
            # insert (power down) -> rod_speed positive (pos increases toward 1)
            
            # Если pid_out > 0 (хотим больше мощности), нужно вытаскивать стержни (speed < 0)
            rod_speed = -pid_out 
            
            # Ограничение скорости
            rod_speed = max(-0.2, min(0.2, rod_speed)) # Макс 20% высоты в секунду
            
            self.prev_error_p = error
            
        # 2. Контроль формы поля / Позиции стержней (Бор)
        if self.auto_boron:
            # Ошибка позиции стержней
            rod_error = current_rod_pos - self.rod_ref_pos
            
            # Если стержни слишком глубоко (rod_error > deadband)
            # Значит в реакторе слишком много реактивности скомпенсировано стержнями?
            # Нет. Чтобы стержни вышли (уменьшить rod_pos), нужно добавить реактивности чем-то другим.
            # Нужно УМЕНЬШИТЬ бор (dilute).
            
            # Если стержни слишком высоко (rod_error < -deadband)
            # Значит реактор "тупой", стержни почти вышли.
            # Нужно УВЕЛИЧИТЬ бор (borate), чтобы заставить стержни пойти внутрь для компенсации.
            
            if rod_error > self.boron_deadband:
                # Rods too deep (e.g. 0.8 > 0.2) -> Need to withdraw rods.
                # To withdraw rods (add reactivity), we must subtract reactivity elsewhere.
                # So we ADD Boron (poison).
                boron_change = self.boron_rate
            elif rod_error < -self.boron_deadband:
                # Rods too high (e.g. 0.0 < 0.2) -> Need to insert rods.
                # To insert rods (remove reactivity), we must add reactivity elsewhere.
                # So we REMOVE Boron (dilute).
                boron_change = -self.boron_rate
            else:
                boron_change = 0.0
                
        return rod_speed, boron_change

    def set_target_power(self, power):
        self.target_power = power
        self.integral_error_p = 0.0

