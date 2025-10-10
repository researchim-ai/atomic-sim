"""
Интеграционные тесты для расширенных возможностей
"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor.simulator_advanced import ReactorSimulatorAdvanced
from reactor.events import PumpCoastdownEvent, ReactivityInsertionEvent
from envs.gym_reactor import ReactorEnv


class TestAdvancedSimulator:
    """Тесты расширенного симулятора"""
    
    def test_with_xenon(self):
        """Тест с моделью отравления ксеноном"""
        sim = ReactorSimulatorAdvanced(enable_xenon=True)
        sim.reset()
        
        # Короткая симуляция
        history = sim.run(duration=100.0, show_progress=False)
        
        assert len(history['time']) > 0
        assert 'Xe_concentration' in history
        assert 'I_concentration' in history
    
    def test_events(self):
        """Тест системы событий"""
        sim = ReactorSimulatorAdvanced(enable_xenon=False)
        sim.reset()
        
        # Добавляем событие
        event = ReactivityInsertionEvent(trigger_time=5.0, reactivity=0.5, duration=10.0)
        sim.event_manager.add_event(event)
        
        # Симуляция
        history = sim.run(duration=20.0, show_progress=False)
        
        # Проверяем что реактивность менялась
        assert len(history['time']) > 0
        
        # В районе t=5-15 должна быть повышенная реактивность
        mid_idx = len(history['time']) // 2
        assert history['power'][mid_idx] > 100.0  # Мощность должна вырасти


class TestGymEnvironment:
    """Тесты Gymnasium обёртки"""
    
    def test_env_creation(self):
        """Тест создания среды"""
        env = ReactorEnv()
        assert env.action_space is not None
        assert env.observation_space is not None
    
    def test_reset(self):
        """Тест сброса среды"""
        env = ReactorEnv()
        obs, info = env.reset(seed=42)
        
        assert obs is not None
        assert len(obs) == 5  # power, T_fuel, T_coolant, reactivity, rod_pos
        assert info is not None
    
    def test_step(self):
        """Тест шага среды"""
        env = ReactorEnv(target_power=95.0)
        obs, info = env.reset(seed=42)
        
        # Выполняем случайное действие
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    
    def test_episode(self):
        """Тест полного эпизода"""
        env = ReactorEnv(target_power=95.0, max_steps=1000)
        obs, info = env.reset(seed=42)
        
        done = False
        steps = 0
        total_reward = 0.0
        
        while not done and steps < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        
        assert steps > 0
        assert total_reward != 0.0  # Должна быть какая-то награда


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

