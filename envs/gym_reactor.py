"""
Gymnasium обёртка для симулятора реактора
Для обучения с подкреплением (RL) и генерации offline-датасетов
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from reactor import ReactorSimulator


class ReactorEnv(gym.Env):
    """
    Среда Gymnasium для управления атомным реактором
    
    Observation Space:
        - power (МВт)
        - T_fuel (°C)
        - T_coolant (°C)
        - rod_reactivity (β)
        - avg_rod_position (0-1)
    
    Action Space:
        Discrete(3): 
        - 0: вставить стержни (снизить мощность)
        - 1: удерживать стержни
        - 2: извлечь стержни (повысить мощность)
    
    Reward:
        - Штраф за отклонение от целевой мощности
        - Большой штраф за нарушение безопасности
        - Малый штраф за движение стержней (экономия ресурса)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 10}
    
    def __init__(
        self,
        target_power=100.0,
        obs_noise=0.0,
        max_steps=200_000,
        reward_scale=1.0,
        safety_penalty=100.0,
        render_mode=None
    ):
        """
        Args:
            target_power: целевая мощность (МВт)
            obs_noise: стандартное отклонение шума наблюдений (доля)
            max_steps: максимальное количество шагов в эпизоде
            reward_scale: масштаб награды
            safety_penalty: штраф за нарушение безопасности
            render_mode: режим рендеринга
        """
        super().__init__()
        
        self.sim = ReactorSimulator(device='cpu')
        self.target_power = target_power
        self.obs_noise = obs_noise
        self.max_steps = max_steps
        self.reward_scale = reward_scale
        self.safety_penalty = safety_penalty
        self.render_mode = render_mode
        
        self.step_count = 0
        self.episode_return = 0.0
        
        # Action space: discrete actions for rod control
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [power, T_fuel, T_coolant, rod_reactivity, avg_rod_position]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -20.0, 0.0], dtype=np.float32),
            high=np.array([500.0, 1500.0, 400.0, 1.0, 1.0], dtype=np.float32),
            shape=(5,),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Сброс среды"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.step_count = 0
        self.episode_return = 0.0
        
        self.sim.reset()
        
        # Опционально: случайная начальная конфигурация
        if options and options.get('random_init', False):
            # Случайная позиция стержней
            init_pos = 0.9 + 0.1 * np.random.rand()
            self.sim.control.rod_positions.fill_(init_pos)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def _get_obs(self):
        """Получить наблюдение"""
        state = self.sim.get_state()
        
        obs = np.array([
            state['power'],
            state['T_fuel'],
            state['T_coolant'],
            state['rod_reactivity'],
            state['avg_rod_position'],
        ], dtype=np.float32)
        
        # Добавляем шум если нужно
        if self.obs_noise > 0:
            noise = np.random.normal(0, self.obs_noise, obs.shape).astype(np.float32)
            obs = obs * (1.0 + noise)
        
        return obs
    
    def _get_info(self):
        """Получить дополнительную информацию"""
        state = self.sim.get_state()
        return {
            'time': state['time'],
            'safe': state['safe'],
            'fuel_overheat': state.get('fuel_overheat', False),
            'coolant_boiling': state.get('coolant_boiling', False),
        }
    
    def step(self, action):
        """Выполнить действие"""
        # Преобразовать discrete action в направление стержней
        action_map = {0: -1, 1: 0, 2: 1}  # вставить, держать, извлечь
        rod_direction = action_map[int(action)]
        
        # Выполнить шаг симуляции
        state = self.sim.step(manual_rod_direction=rod_direction)
        
        self.step_count += 1
        
        # Вычислить награду
        reward = self._compute_reward(state, rod_direction)
        self.episode_return += reward
        
        # Проверить терминальность
        terminated = not state['safe']  # Завершить если небезопасно
        truncated = self.step_count >= self.max_steps  # Ограничение по времени
        
        obs = self._get_obs()
        info = self._get_info()
        info['episode_return'] = self.episode_return
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, state, rod_direction):
        """
        Вычислить награду
        
        Компоненты:
        1. Tracking error: штраф за отклонение от целевой мощности
        2. Safety: большой штраф за нарушение безопасности
        3. Action cost: малый штраф за движение стержней
        """
        # 1. Tracking error
        power_error = abs(state['power'] - self.target_power) / self.target_power
        tracking_reward = -power_error
        
        # 2. Safety penalty
        safety_reward = -self.safety_penalty if not state['safe'] else 0.0
        
        # 3. Action cost (штраф за частые движения)
        action_cost = -0.001 * abs(rod_direction)
        
        # Суммарная награда
        total_reward = (tracking_reward + safety_reward + action_cost) * self.reward_scale
        
        return total_reward
    
    def render(self):
        """Отрисовка состояния"""
        if self.render_mode == "human":
            state = self.sim.get_state()
            print(f"\n{'='*60}")
            print(f"Step: {self.step_count} | Time: {state['time']:.2f}s")
            print(f"Power: {state['power']:.2f} МВт (target: {self.target_power:.2f})")
            print(f"T_fuel: {state['T_fuel']:.1f}°C | T_cool: {state['T_coolant']:.1f}°C")
            print(f"Rods: {state['avg_rod_position']*100:.1f}% | Safe: {state['safe']}")
            print(f"Return: {self.episode_return:.2f}")
            print(f"{'='*60}")
    
    def close(self):
        """Закрыть среду"""
        pass


class ReactorEnvContinuous(ReactorEnv):
    """
    Версия с непрерывными действиями
    
    Action Space:
        Box(1): скорость движения стержней [-1, 1]
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Continuous action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
    
    def step(self, action):
        """Выполнить действие с непрерывным управлением"""
        # action в диапазоне [-1, 1]
        action_value = float(np.clip(action[0], -1.0, 1.0))
        
        # Преобразовать в дискретное направление
        if action_value > 0.2:
            rod_direction = 1
        elif action_value < -0.2:
            rod_direction = -1
        else:
            rod_direction = 0
        
        # Выполнить шаг
        state = self.sim.step(manual_rod_direction=rod_direction)
        
        self.step_count += 1
        
        # Вычислить награду с учетом величины действия
        reward = self._compute_reward(state, action_value)
        self.episode_return += reward
        
        terminated = not state['safe']
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_obs()
        info = self._get_info()
        info['episode_return'] = self.episode_return
        
        return obs, reward, terminated, truncated, info


# Регистрация сред в Gymnasium
try:
    import torch
    
    gym.register(
        id='AtomicReactor-v0',
        entry_point='envs.gym_reactor:ReactorEnv',
        max_episode_steps=200_000,
    )
    
    gym.register(
        id='AtomicReactorContinuous-v0',
        entry_point='envs.gym_reactor:ReactorEnvContinuous',
        max_episode_steps=200_000,
    )
except:
    pass  # Регистрация опциональна


if __name__ == "__main__":
    # Тестовый прогон
    import torch
    
    print("Тестирование Reactor Gymnasium Environment...")
    
    env = ReactorEnv(target_power=95.0, render_mode="human")
    obs, info = env.reset(seed=42)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Initial obs: {obs}")
    
    # Несколько случайных шагов
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
        if i % 5 == 0:
            env.render()
        
        if term or trunc:
            break
    
    env.close()
    print("\n✓ Тест завершен успешно!")

