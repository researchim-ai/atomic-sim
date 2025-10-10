"""
Тесты физических моделей
Проверка корректности физических уравнений и законов сохранения
"""

import pytest
import torch
import numpy as np
from reactor import ReactorSimulator
from reactor.neutronics import NeutronKinetics
from reactor.thermal import ThermalModel
from reactor.poisoning import XenonIodineKinetics


class TestNeutronics:
    """Тесты нейтронной кинетики"""
    
    def test_critical_state(self):
        """Критическое состояние: при ρ=0 мощность должна быть стабильной"""
        neutronics = NeutronKinetics()
        neutronics.reset()
        
        initial_n = neutronics.n.item()
        
        # 10 секунд при нулевой реактивности
        for _ in range(10000):
            neutronics.step(0.001, reactivity=0.0)
        
        final_n = neutronics.n.item()
        
        # Плотность нейтронов должна остаться примерно такой же
        assert abs(final_n - initial_n) < 0.01, f"n changed from {initial_n} to {final_n}"
    
    def test_positive_reactivity(self):
        """Положительная реактивность → рост мощности"""
        neutronics = NeutronKinetics()
        neutronics.reset()
        
        initial_power = (neutronics.n * neutronics.P0).item()
        
        # Небольшая положительная реактивность
        for _ in range(1000):
            neutronics.step(0.001, reactivity=0.1)  # +0.1 β
        
        final_power = (neutronics.n * neutronics.P0).item()
        
        assert final_power > initial_power, "Power should increase with positive reactivity"
    
    def test_negative_reactivity(self):
        """Отрицательная реактивность → падение мощности"""
        neutronics = NeutronKinetics()
        neutronics.reset()
        
        initial_power = (neutronics.n * neutronics.P0).item()
        
        # Отрицательная реактивность
        for _ in range(1000):
            neutronics.step(0.001, reactivity=-0.5)  # -0.5 β
        
        final_power = (neutronics.n * neutronics.P0).item()
        
        assert final_power < initial_power, "Power should decrease with negative reactivity"
    
    def test_beta_sum(self):
        """Проверка суммы долей запаздывающих нейтронов"""
        neutronics = NeutronKinetics()
        
        beta_sum = torch.sum(neutronics.beta_i).item()
        beta_total = neutronics.beta.item()
        
        assert abs(beta_sum - beta_total) < 1e-10, "Sum of beta_i should equal beta"


class TestThermal:
    """Тесты тепловой модели"""
    
    def test_energy_balance(self):
        """Проверка энергетического баланса"""
        thermal = ThermalModel()
        thermal.reset()
        
        power = 100.0  # МВт
        dt = 0.001
        
        # Один шаг
        initial_energy = (thermal.C_fuel * thermal.T_fuel + 
                         thermal.C_clad * thermal.T_clad + 
                         thermal.C_coolant * thermal.T_coolant)
        
        thermal.step(dt, power)
        
        final_energy = (thermal.C_fuel * thermal.T_fuel + 
                       thermal.C_clad * thermal.T_clad + 
                       thermal.C_coolant * thermal.T_coolant)
        
        # Энергия должна увеличиться примерно на power * dt
        energy_added = (final_energy - initial_energy).item()
        expected = power * dt
        
        # Допускаем 50% погрешность (часть энергии уходит в охлаждение)
        assert energy_added > 0, "Energy should increase"
        assert energy_added < expected * 2, "Energy increase shouldn't be too large"
    
    def test_cooling(self):
        """Проверка охлаждения"""
        thermal = ThermalModel()
        thermal.reset()
        
        # Нагреваем систему
        for _ in range(1000):
            thermal.step(0.001, 200.0)
        
        high_temp = thermal.T_fuel.item()
        
        # Охлаждаем (нулевая мощность)
        for _ in range(5000):
            thermal.step(0.001, 0.0)
        
        low_temp = thermal.T_fuel.item()
        
        assert low_temp < high_temp, "Temperature should decrease without power"
    
    def test_negative_feedback(self):
        """Проверка отрицательной температурной обратной связи"""
        thermal = ThermalModel()
        thermal.reset()
        
        initial_T = thermal.T_fuel.item()
        
        # Повышаем температуру
        for _ in range(1000):
            thermal.step(0.001, 150.0)
        
        reactivity_fb = thermal.step(0.001, 150.0)
        
        # Обратная связь должна быть отрицательной
        assert reactivity_fb.item() < 0, "Temperature feedback should be negative"


class TestXenonPoisoning:
    """Тесты модели отравления ксеноном"""
    
    def test_xenon_buildup(self):
        """Ксенон накапливается при работе реактора"""
        xenon = XenonIodineKinetics()
        xenon.reset()
        
        initial_Xe = xenon.Xe.item()
        
        # Работа на мощности
        for _ in range(10000):  # ~10 секунд
            xenon.step(0.001, power=100.0)
        
        final_Xe = xenon.Xe.item()
        
        assert final_Xe > initial_Xe, "Xenon should build up during operation"
    
    def test_xenon_decay(self):
        """Ксенон распадается после останова"""
        xenon = XenonIodineKinetics()
        xenon.reset_equilibrium(100.0)  # Начинаем с равновесия
        
        peak_Xe = xenon.Xe.item()
        
        # Останов реактора
        for _ in range(20000):  # ~20 секунд
            xenon.step(0.001, power=0.0)
        
        final_Xe = xenon.Xe.item()
        
        # Ксенон должен сначала вырасти (из йода), потом упасть
        # Проверяем что в итоге упал
        assert final_Xe < peak_Xe, "Xenon should eventually decay"
    
    def test_negative_reactivity(self):
        """Ксенон вносит отрицательную реактивность"""
        xenon = XenonIodineKinetics()
        xenon.reset()
        
        # Накапливаем ксенон
        for _ in range(10000):
            xenon.step(0.001, power=100.0)
        
        reactivity = (xenon.alpha_xe * xenon.Xe).item()
        
        assert reactivity < 0, "Xenon reactivity should be negative"


class TestIntegrated:
    """Интеграционные тесты полного симулятора"""
    
    def test_steady_state(self):
        """Симулятор должен оставаться стабильным при критичности"""
        sim = ReactorSimulator()
        sim.reset()
        
        initial_power = sim.get_state()['power']
        
        # 10 секунд без управления
        for _ in range(10000):
            sim.step(manual_rod_direction=0)
        
        final_power = sim.get_state()['power']
        
        # Мощность не должна сильно измениться
        error = abs(final_power - initial_power) / initial_power
        assert error < 0.05, f"Power drifted by {error*100:.1f}%"
    
    def test_scram_effectiveness(self):
        """SCRAM должен быстро снижать мощность"""
        sim = ReactorSimulator()
        sim.reset()
        
        # Выполняем SCRAM
        sim.scram()
        
        initial_power = sim.get_state()['power']
        
        # 5 секунд после SCRAM
        for _ in range(5000):
            sim.step()
        
        final_power = sim.get_state()['power']
        
        # Мощность должна значительно упасть
        assert final_power < initial_power * 0.5, "SCRAM should reduce power significantly"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

