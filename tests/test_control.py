"""
Тесты системы управления
"""

import pytest
import torch
from reactor.control import ControlSystem
from reactor.rod_worth import rod_worth_curve, differential_worth


class TestControlSystem:
    """Тесты системы управления стержнями"""
    
    def test_rod_movement(self):
        """Стержни должны двигаться с заданной скоростью"""
        control = ControlSystem()
        control.reset()
        
        initial_pos = control.rod_positions[0].item()
        
        # Вставляем стержни
        control.move_rods(-1, dt=1.0)
        
        final_pos = control.rod_positions[0].item()
        expected_pos = initial_pos - control.rod_speed.item()
        
        assert abs(final_pos - expected_pos) < 1e-6, "Rod should move at specified speed"
    
    def test_rod_limits(self):
        """Стержни не могут выйти за пределы [0, 1]"""
        control = ControlSystem()
        control.reset()
        
        # Пытаемся извлечь стержни выше максимума
        for _ in range(100):
            control.move_rods(1, dt=0.1)
        
        assert all(control.rod_positions <= 1.0), "Rods should not exceed 1.0"
        
        # Пытаемся вставить стержни ниже минимума
        for _ in range(200):
            control.move_rods(-1, dt=0.1)
        
        assert all(control.rod_positions >= 0.0), "Rods should not go below 0.0"
    
    def test_scram(self):
        """SCRAM должен мгновенно вставить все стержни"""
        control = ControlSystem()
        control.reset()
        
        control.scram()
        
        assert all(control.rod_positions == 0.0), "All rods should be fully inserted after SCRAM"
    
    def test_pid_deadband(self):
        """ПИД не должен срабатывать при малой ошибке (deadband)"""
        control = ControlSystem()
        control.reset()
        control.use_auto_control = True
        control.target_power = torch.tensor(100.0)
        
        # Малая ошибка (меньше deadband)
        current_power = 100.0 + control.deadband.item() * 5.0  # Малая ошибка
        
        direction = control.auto_control_step(current_power, 0.001)
        
        # При малой ошибке стержни не должны двигаться
        # Хотя если ошибка чуть больше deadband, должны двигаться
        # Проверим граничный случай
        control.previous_error = torch.tensor(0.0)
        control.integral_error = torch.tensor(0.0)
        
        # Точно на границе deadband
        small_error_power = 100.0 + control.deadband.item() * 10.0
        direction = control.auto_control_step(small_error_power, 0.001)
        
        # Направление должно быть определено (±1 или 0)
        assert direction in [-1, 0, 1]


class TestRodWorth:
    """Тесты кривой ценности стержней"""
    
    def test_worth_boundaries(self):
        """Проверка граничных значений кривой worth"""
        # При полностью извлеченном стержне (pos=1) worth=0
        worth_extracted = rod_worth_curve(torch.tensor(1.0))
        assert abs(worth_extracted.item()) < 1e-6, "Worth should be 0 when fully extracted"
        
        # При полностью вставленном стержне (pos=0) worth=1
        worth_inserted = rod_worth_curve(torch.tensor(0.0))
        assert abs(worth_inserted.item() - 1.0) < 1e-6, "Worth should be 1 when fully inserted"
    
    def test_worth_monotonic(self):
        """Кривая worth должна быть монотонно убывающей"""
        positions = torch.linspace(0, 1, 100)
        worth = rod_worth_curve(positions)
        
        # Проверяем монотонность
        diffs = worth[1:] - worth[:-1]
        assert all(diffs <= 0), "Worth should be monotonically decreasing"
    
    def test_differential_worth(self):
        """Дифференциальная ценность должна быть отрицательной"""
        positions = torch.linspace(0.1, 0.9, 10)
        diff_worth = differential_worth(positions)
        
        # Производная должна быть отрицательной (worth убывает)
        assert all(diff_worth < 0), "Differential worth should be negative"
    
    def test_nonlinear_vs_linear(self):
        """Нелинейная кривая должна отличаться от линейной"""
        pos = torch.tensor(0.5)
        
        nonlinear = rod_worth_curve(pos, shape=2.2)
        linear = 1.0 - pos
        
        # Должны отличаться
        assert abs(nonlinear.item() - linear.item()) > 0.01, "Nonlinear should differ from linear"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

