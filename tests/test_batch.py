"""
Тесты батч-симулятора
"""

import pytest
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor.batch_simulator import BatchReactorSimulator


class TestBatchSimulator:
    """Тесты батч-симуляции"""
    
    def test_creation(self):
        """Тест создания батч-симулятора"""
        batch_size = 8
        sim = BatchReactorSimulator(batch_size=batch_size, device='cpu', dtype=torch.float32)
        
        assert sim.batch_size == batch_size
        assert sim.neutronics.n.shape == (batch_size,)
        assert sim.rod_positions.shape == (batch_size, sim.n_rods)
    
    def test_reset(self):
        """Тест сброса батча"""
        sim = BatchReactorSimulator(batch_size=4, device='cpu', dtype=torch.float32)
        
        # Изменяем состояние
        sim.neutronics.n.fill_(2.0)
        
        # Сбрасываем
        sim.reset_batch()
        
        assert torch.all(sim.neutronics.n == 1.0)
    
    def test_partial_reset(self):
        """Тест частичного сброса батча"""
        sim = BatchReactorSimulator(batch_size=4, device='cpu', dtype=torch.float32)
        
        sim.neutronics.n.fill_(2.0)
        
        # Сбросить только элементы 0 и 2
        sim.reset_batch(indices=[0, 2])
        
        assert sim.neutronics.n[0].item() == 1.0
        assert sim.neutronics.n[1].item() == 2.0
        assert sim.neutronics.n[2].item() == 1.0
        assert sim.neutronics.n[3].item() == 2.0
    
    def test_batch_step(self):
        """Тест батч-шага"""
        batch_size = 4
        sim = BatchReactorSimulator(batch_size=batch_size, device='cpu', dtype=torch.float32)
        sim.reset_batch()
        
        # Случайные действия для каждого элемента батча
        actions = torch.randint(-1, 2, (batch_size,), dtype=torch.float32)
        
        obs, rewards, dones = sim.step_batch(actions)
        
        assert obs.shape == (batch_size, 5)
        assert rewards.shape == (batch_size,)
        assert dones.shape == (batch_size,)
    
    def test_run_batch(self):
        """Тест полного прогона батча"""
        batch_size = 8
        n_steps = 1000
        
        sim = BatchReactorSimulator(batch_size=batch_size, device='cpu', dtype=torch.float32)
        
        trajectory = sim.run_batch(n_steps)
        
        assert trajectory['observations'].shape == (batch_size, n_steps, 5)
        assert trajectory['actions'].shape == (batch_size, n_steps)
        assert trajectory['rewards'].shape == (batch_size, n_steps)
        assert trajectory['dones'].shape == (batch_size, n_steps)
    
    def test_performance(self):
        """Тест производительности батч-симуляции"""
        import time
        
        batch_size = 16
        n_steps = 10000
        
        sim = BatchReactorSimulator(batch_size=batch_size, device='cpu', dtype=torch.float32)
        
        start = time.time()
        trajectory = sim.run_batch(n_steps)
        elapsed = time.time() - start
        
        total_steps = batch_size * n_steps
        steps_per_sec = total_steps / elapsed
        
        print(f"\n  Батч: {batch_size}, Шагов: {n_steps}")
        print(f"  Производительность: {steps_per_sec:,.0f} шагов/сек")
        
        # Должен быть быстрее одиночного симулятора
        assert steps_per_sec > 40000, "Batch simulator should be fast"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])

