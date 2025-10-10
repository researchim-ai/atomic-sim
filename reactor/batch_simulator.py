"""
Батч-симулятор для параллельной генерации данных
Использует векторизацию PyTorch для ускорения
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List


class BatchNeutronKinetics(nn.Module):
    """
    Батч-версия нейтронной кинетики
    Обрабатывает B независимых симуляций одновременно
    """
    
    def __init__(self, batch_size: int, device='cpu', dtype=torch.float32):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        self.n_groups = 6
        
        # Параметры (общие для всего батча)
        beta_values = torch.tensor([
            0.000215, 0.001424, 0.001274, 0.002568, 0.000748, 0.000273
        ], device=device, dtype=dtype)
        
        lambda_values = torch.tensor([
            0.0124, 0.0305, 0.111, 0.301, 1.14, 3.01
        ], device=device, dtype=dtype)
        
        self.register_buffer('beta_i', beta_values)
        self.register_buffer('lambda_i', lambda_values)
        self.register_buffer('beta', torch.sum(beta_values))
        self.register_buffer('Lambda', torch.tensor(5e-5, device=device, dtype=dtype))
        self.P0 = torch.tensor(100.0, device=device, dtype=dtype)
        
        # Состояния батча [B] и [B, n_groups]
        self.n = torch.ones(batch_size, device=device, dtype=dtype)
        self.C = self.beta_i.unsqueeze(0).expand(batch_size, -1) / (self.lambda_i * self.Lambda)
    
    def reset_batch(self, indices: Optional[List[int]] = None):
        """Сбросить указанные элементы батча"""
        if indices is None:
            # Сбросить весь батч
            self.n.fill_(1.0)
            self.C = self.beta_i.unsqueeze(0).expand(self.batch_size, -1) / (self.lambda_i * self.Lambda)
        else:
            # Сбросить только указанные
            self.n[indices] = 1.0
            for idx in indices:
                self.C[idx] = self.beta_i / (self.lambda_i * self.Lambda)
    
    def compute_derivatives(self, n, C, reactivity):
        """
        Вычислить производные для батча
        
        Args:
            n: [B] плотности нейтронов
            C: [B, n_groups] концентрации предшественников
            reactivity: [B] реактивности
        
        Returns:
            dn_dt: [B]
            dC_dt: [B, n_groups]
        """
        rho = reactivity * self.beta
        
        # dn/dt для каждого элемента батча
        dn_dt = (rho - self.beta) / self.Lambda * n + torch.sum(self.lambda_i * C, dim=1)
        
        # dC/dt для каждого элемента
        dC_dt = self.beta_i.unsqueeze(0) / self.Lambda * n.unsqueeze(1) - self.lambda_i * C
        
        return dn_dt, dC_dt
    
    def step_batch(self, dt, reactivity):
        """
        Батч-шаг RK4
        
        Args:
            dt: scalar или [B] шаг времени
            reactivity: [B] реактивности
        
        Returns:
            power: [B] мощности
        """
        n0 = self.n.clone()
        C0 = self.C.clone()
        
        # RK4
        dn1, dC1 = self.compute_derivatives(n0, C0, reactivity)
        
        n_mid = n0 + 0.5 * dt * dn1
        C_mid = C0 + 0.5 * dt * dC1.unsqueeze(-1) if dC1.dim() == 1 else C0 + 0.5 * dt * dC1
        dn2, dC2 = self.compute_derivatives(n_mid, C_mid, reactivity)
        
        n_mid = n0 + 0.5 * dt * dn2
        C_mid = C0 + 0.5 * dt * dC2.unsqueeze(-1) if dC2.dim() == 1 else C0 + 0.5 * dt * dC2
        dn3, dC3 = self.compute_derivatives(n_mid, C_mid, reactivity)
        
        n_end = n0 + dt * dn3
        C_end = C0 + dt * dC3.unsqueeze(-1) if dC3.dim() == 1 else C0 + dt * dC3
        dn4, dC4 = self.compute_derivatives(n_end, C_end, reactivity)
        
        # Обновление
        self.n = n0 + dt / 6.0 * (dn1 + 2*dn2 + 2*dn3 + dn4)
        self.C = C0 + dt / 6.0 * (dC1 + 2*dC2 + 2*dC3 + dC4)
        
        # Ограничения
        self.n = torch.clamp(self.n, min=1e-10)
        self.C = torch.clamp(self.C, min=1e-10)
        
        power = self.n * self.P0
        
        return power


class BatchReactorSimulator:
    """
    Батч-симулятор для параллельной генерации датасетов
    
    Обрабатывает B независимых симуляций одновременно
    Использует векторизацию PyTorch для ускорения
    """
    
    def __init__(self, batch_size: int, device='cpu', dtype=torch.float32):
        """
        Args:
            batch_size: количество параллельных симуляций
            device: 'cpu' или 'cuda'
            dtype: torch.float32 (быстрее) или torch.float64 (точнее)
        """
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        
        # Батч-версия нейтронной кинетики
        self.neutronics = BatchNeutronKinetics(batch_size, device, dtype)
        
        # Для упрощения пока используем скалярные параметры
        self.dt = 0.001
        self.time = torch.zeros(batch_size, device=device, dtype=dtype)
        
        # Позиции стержней [B, n_rods]
        self.n_rods = 10
        self.rod_positions = torch.ones(batch_size, self.n_rods, device=device, dtype=dtype)
        self.rod_worth = torch.tensor(-2.0, device=device, dtype=dtype)
        
        # Температуры [B]
        self.T_fuel = torch.ones(batch_size, device=device, dtype=dtype) * 287.0
        self.T_coolant = torch.ones(batch_size, device=device, dtype=dtype) * 281.0
        
        # История (опционально, может быть большой)
        self.enable_history = False
        self.history = None
    
    def reset_batch(self, indices: Optional[List[int]] = None):
        """
        Сбросить батч или его части
        
        Args:
            indices: индексы для сброса (None = все)
        """
        if indices is None:
            indices = list(range(self.batch_size))
        
        self.neutronics.reset_batch(indices)
        
        self.time[indices] = 0.0
        self.rod_positions[indices] = 1.0
        self.T_fuel[indices] = 287.0
        self.T_coolant[indices] = 281.0
    
    def step_batch(self, actions: torch.Tensor):
        """
        Батч-шаг симуляции
        
        Args:
            actions: [B] действия (direction: -1, 0, +1)
        
        Returns:
            observations: [B, obs_dim]
            rewards: [B]
            dones: [B] bool
        """
        # Вычислить реактивность от стержней [B]
        worth_factors = 1.0 - self.rod_positions.mean(dim=1)  # Упрощенно
        rod_reactivity = worth_factors * self.rod_worth * self.n_rods
        
        # Температурная обратная связь (упрощенно)
        alpha_fuel = -5e-5
        temp_reactivity = alpha_fuel * (self.T_fuel - 287.0)
        
        # Полная реактивность [B]
        total_reactivity = rod_reactivity + temp_reactivity
        
        # Шаг нейтронной кинетики
        power = self.neutronics.step_batch(self.dt, total_reactivity)
        
        # Упрощенное обновление температуры (без полной теплотехники)
        # dT/dt ≈ power / C_fuel - (T - T_ambient) / tau
        C_fuel = 200.0
        tau = 50.0  # Постоянная времени охлаждения
        T_ambient = 281.0
        
        dT_fuel = power / C_fuel - (self.T_fuel - T_ambient) / tau
        self.T_fuel = torch.clamp(self.T_fuel + self.dt * dT_fuel, min=250.0, max=1500.0)
        
        # Обновить позиции стержней [B, n_rods]
        # actions: [B], нужно применить ко всем стержням
        rod_speed = 0.10
        delta = actions.unsqueeze(1) * rod_speed * self.dt
        self.rod_positions = torch.clamp(self.rod_positions + delta, 0.0, 1.0)
        
        # Обновить время
        self.time += self.dt
        
        # Observations [B, 5]
        observations = torch.stack([
            power,
            self.T_fuel,
            self.T_coolant,
            rod_reactivity,
            self.rod_positions.mean(dim=1)
        ], dim=1)
        
        # Rewards (примерная целевая мощность 100 МВт) [B]
        target_power = 100.0
        rewards = -torch.abs(power - target_power) / target_power
        
        # Dones (нарушение безопасности) [B]
        dones = (self.T_fuel > 1200.0) | (self.T_coolant > 320.0)
        
        return observations, rewards, dones
    
    def run_batch(self, n_steps: int, policy_fn=None):
        """
        Запустить батч симуляций
        
        Args:
            n_steps: количество шагов
            policy_fn: функция политики (obs -> actions) или None для случайных
        
        Returns:
            trajectory: dict с траекториями [B, T, ...]
        """
        observations_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        
        # Сброс
        self.reset_batch()
        
        for step in range(n_steps):
            # Получить текущие observations
            power = self.neutronics.n * self.neutronics.P0
            obs = torch.stack([
                power,
                self.T_fuel,
                self.T_coolant,
                torch.zeros(self.batch_size, device=self.device, dtype=self.dtype),
                self.rod_positions.mean(dim=1)
            ], dim=1)
            
            # Выбрать действия
            if policy_fn is None:
                # Случайная политика
                actions = torch.randint(-1, 2, (self.batch_size,), device=self.device, dtype=self.dtype)
            else:
                actions = policy_fn(obs)
            
            # Шаг
            obs_next, rewards, dones = self.step_batch(actions)
            
            # Сохранить
            observations_list.append(obs.cpu())
            actions_list.append(actions.cpu())
            rewards_list.append(rewards.cpu())
            dones_list.append(dones.cpu())
            
            # Сбросить завершенные эпизоды
            done_indices = torch.where(dones)[0].tolist()
            if done_indices:
                self.reset_batch(done_indices)
        
        # Собрать траектории
        trajectory = {
            'observations': torch.stack(observations_list, dim=1),  # [B, T, obs_dim]
            'actions': torch.stack(actions_list, dim=1),  # [B, T]
            'rewards': torch.stack(rewards_list, dim=1),  # [B, T]
            'dones': torch.stack(dones_list, dim=1),  # [B, T]
        }
        
        return trajectory


if __name__ == '__main__':
    # Тест батч-симулятора
    print("Тест батч-симулятора...")
    
    batch_size = 16
    n_steps = 10000
    
    sim = BatchReactorSimulator(batch_size=batch_size, device='cpu', dtype=torch.float32)
    
    print(f"Батч размер: {batch_size}")
    print(f"Шагов: {n_steps}")
    
    import time
    start = time.time()
    
    trajectory = sim.run_batch(n_steps)
    
    elapsed = time.time() - start
    total_steps = batch_size * n_steps
    
    print(f"\nВремя: {elapsed:.2f}с")
    print(f"Всего шагов: {total_steps:,}")
    print(f"Производительность: {total_steps/elapsed:,.0f} шагов/сек")
    print(f"\nТраектории:")
    print(f"  observations: {trajectory['observations'].shape}")
    print(f"  actions: {trajectory['actions'].shape}")
    print(f"  rewards: {trajectory['rewards'].shape}")
    
    print("\n✓ Батч-симулятор работает!")

