# 🚀 Расширенные возможности

Этот документ описывает продвинутые функции симулятора для генерации датасетов и обучения ML/RL моделей.

## 🔬 Физические модели

### 1. Отравление ксеноном и йодом

Модель Xe-135 и I-135 для реалистичного поведения реактора:

```python
from reactor.simulator_advanced import ReactorSimulatorAdvanced

# Создание с поддержкой ксенона
sim = ReactorSimulatorAdvanced(enable_xenon=True)
sim.reset(equilibrium_xenon=True)  # Начать с равновесным ксеноном

history = sim.run(duration=50000.0)  # Длительная симуляция

# Анализ отравления
print(f"Концентрация Xe: {sim.xenon.Xe.item():.6f}")
print(f"Реактивность от Xe: {sim.xenon.get_state()['Xe_reactivity']:.6f} β")
```

**Эффект ксеноновой ямы:**
- При снижении мощности Xe сначала растет (из накопленного I-135)
- Создает отрицательную реактивность, затрудняя повторный пуск
- Критично для реалистичного поведения

### 2. Нелинейная кривая ценности стержней

S-образная кривая worth для реалистичного моделирования:

```python
from reactor.rod_worth import rod_worth_curve, differential_worth

# Вычислить worth для позиции
position = torch.tensor(0.5)
worth = rod_worth_curve(position, shape=2.2)

# Дифференциальная ценность (эффективность движения)
diff = differential_worth(position)
```

Используется автоматически в `ControlSystem.compute_rod_reactivity()`.

### 3. Система событий

Моделирование аварийных и штатных событий:

```python
from reactor.events import (
    PumpCoastdownEvent,
    ReactivityInsertionEvent,
    StuckRodEvent,
    RodEjectionEvent,
    LossOfHeatSinkEvent
)

sim = ReactorSimulatorAdvanced()
sim.reset()

# Добавить события
sim.event_manager.add_event(
    PumpCoastdownEvent(trigger_time=30.0, time_constant=3.0)
)
sim.event_manager.add_event(
    StuckRodEvent(trigger_time=45.0, rod_index=3)
)

# Симуляция с событиями
history = sim.run(duration=100.0)
```

**Доступные события:**
- `PumpCoastdownEvent` - отказ насоса с экспоненциальным снижением
- `ReactivityInsertionEvent` - ввод внешней реактивности
- `StuckRodEvent` - заклинивание стержня
- `RodEjectionEvent` - выброс стержня (тяжелая авария)
- `LossOfHeatSinkEvent` - полная потеря теплосъёма

## 🤖 Reinforcement Learning

### Gymnasium интерфейс

```python
from envs.gym_reactor import ReactorEnv

# Создание среды
env = ReactorEnv(
    target_power=100.0,
    obs_noise=0.01,  # 1% шум измерений
    max_steps=200_000
)

# Стандартный цикл RL
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # Ваша политика
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()
```

**Observation space** (5 переменных):
- Мощность (МВт)
- Температура топлива (°C)
- Температура теплоносителя (°C)
- Реактивность от стержней (β)
- Средняя позиция стержней (0-1)

**Action space:**
- `ReactorEnv`: Discrete(3) - вставить/держать/извлечь
- `ReactorEnvContinuous`: Box(1) - непрерывное управление

**Reward function:**
- Tracking: штраф за отклонение от целевой мощности
- Safety: большой штраф за нарушение безопасности
- Action cost: малый штраф за движение стержней

### Регистрация в Gymnasium

```python
import gymnasium as gym

# Автоматически зарегистрированы:
env = gym.make('AtomicReactor-v0')  # Дискретные действия
env = gym.make('AtomicReactorContinuous-v0')  # Непрерывные
```

## 📊 Генерация датасетов

### Базовая генерация

```python
from generate_dataset import DatasetGenerator

generator = DatasetGenerator(output_dir='./datasets', seed=42)

# Генерация батча сценариев
trajectories = generator.generate_batch(
    n_scenarios=100,
    scenario_types=['random', 'auto_control', 'ramp', 'emergency'],
    duration=100.0
)

# Сохранение в различных форматах
generator.save_dataset(trajectories, name='my_dataset', format='parquet')
```

### Батч-симуляция (высокая производительность)

```python
from reactor.batch_simulator import BatchReactorSimulator

# Создать батч-симулятор
batch_sim = BatchReactorSimulator(
    batch_size=32,
    device='cpu',  # или 'cuda' для GPU
    dtype=torch.float32  # float32 быстрее чем float64
)

# Генерация траекторий
n_steps = 100_000
trajectory = batch_sim.run_batch(n_steps)

# trajectory['observations']: [32, 100000, 5]
# trajectory['actions']: [32, 100000]
# trajectory['rewards']: [32, 100000]
```

**Производительность:**
- Обычный симулятор: ~5,500 шагов/сек
- Батч-симулятор (B=16): ~125,000 шагов/сек (**×22 быстрее**)
- Батч-симулятор (B=32, GPU): ~500,000+ шагов/сек (**×90 быстрее**)

### Q&A генерация для LLM

```python
from dataset_builders.qa_from_traces import QAGenerator

# Загрузить траектории
import pandas as pd
df = pd.read_parquet('datasets/reactor_dataset.parquet')

# Генерация Q&A
generator = QAGenerator(language='ru')
qa_pairs = generator.generate_qa_pairs(df, n_samples=1000)

# Сохранение
generator.save_qa_dataset(qa_pairs, 'datasets/qa_dataset.jsonl')
```

**Типы вопросов:**
- Изменение мощности и причины
- Температурные режимы и безопасность
- Управление стержнями и реактивность
- Аварийные ситуации и SCRAM

### Формат датасета

**Parquet/CSV структура:**
```
scenario_id | scenario_type | step | time | power_MW | T_fuel_C | ... | safe | action
0           | emergency     | 100  | 0.1  | 142.3    | 650.1    | ... | True | 1
0           | emergency     | 101  | 0.101| 142.5    | 650.2    | ... | True | 1
...
```

**JSONL для Q&A:**
```json
{"id": 0, "question": "Мощность выросла на 15 МВт...", "answer": "Стержни были извлечены..."}
{"id": 1, "question": "Температура топлива 950°C...", "answer": "ПРЕДУПРЕЖДЕНИЕ! Приближается..."}
```

## ⚙️ Конфигурации

### Загрузка конфигов

```python
from reactor.config_loader import load_config

# Загрузить default config
config = load_config()

# Или из файла
config = load_config('configs/high_power.yaml')

# Использовать параметры
dt = config.get('simulator.dt', default=0.001)
Kp = config.get('control.Kp', default=0.02)
```

### Создание своего конфига

`configs/high_power.yaml`:
```yaml
simulator:
  device: cuda
  dt: 0.0005

neutronics:
  P0: 200.0  # Номинальная мощность 200 МВт

control:
  Kp: 0.03
  Ki: 0.003
  Kd: 0.015
  deadband: 0.05
```

## 📈 Сценарии

### YAML-сценарии

`scenarios/custom_scenario.yaml`:
```yaml
name: custom_scenario
duration: 200
seed: 123

timeline:
  - t: 0
    action: set_auto_control
    params:
      enabled: true
      target_power: 120

  - t: 60
    action: set_coolant_flow
    params:
      value: 0.5

  - t: 90
    action: scram_if
    params:
      condition: "T_fuel > 850"
```

## 🧪 Расширенное тестирование

### Запуск всех тестов

```bash
# Все тесты
pytest tests/ -v

# Только физические
pytest tests/test_physics.py -v

# С покрытием кода
pytest tests/ --cov=reactor --cov-report=html
```

### Категории тестов

1. **test_physics.py** - физические модели и законы сохранения
2. **test_control.py** - система управления и ПИД
3. **test_simulator.py** - интеграция и smoke-тесты
4. **test_integration.py** - расширенные функции (Xenon, события, Gym)

## 🎯 Производительность

### Benchmark

```bash
python -c "
from reactor.batch_simulator import BatchReactorSimulator
import time

sim = BatchReactorSimulator(batch_size=32, dtype=torch.float32)
start = time.time()
traj = sim.run_batch(100000)
elapsed = time.time() - start
print(f'Производительность: {32*100000/elapsed:,.0f} шагов/сек')
"
```

### Оптимизация

- **float32 vs float64**: float32 на 30-50% быстрее
- **Batch size**: оптимально 16-64 для CPU, 128-512 для GPU
- **GPU**: используйте `device='cuda'` для ускорения в 5-10x
- **torch.compile**: добавьте `torch.compile()` для PyTorch 2.0+

## 📚 Примеры

### Генерация большого датасета

```bash
# 1000 сценариев в Parquet
python generate_dataset.py --n-scenarios 1000 --duration 100 --format parquet --seed 42

# Q&A из траекторий
python dataset_builders/qa_from_traces.py --input datasets/reactor_dataset*.parquet --output datasets/qa.jsonl --n-samples 5000
```

### Обучение RL агента (пример с stable-baselines3)

```python
from stable_baselines3 import PPO
from envs.gym_reactor import ReactorEnv

env = ReactorEnv(target_power=95.0)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

model.save("reactor_ppo")
```

### Сценарные симуляции

```python
from reactor.simulator_advanced import ReactorSimulatorAdvanced
from reactor.events import create_loss_of_cooling_scenario

sim = ReactorSimulatorAdvanced()
sim.reset()

# Загрузить предопределенный сценарий
sim.event_manager = create_loss_of_cooling_scenario()

history = sim.run(duration=200.0)
```

## 🔧 Параметры ПИД

Настройка для вашей задачи:

```python
simulator.control.Kp = torch.tensor(0.03)  # Быстрее реакция
simulator.control.Ki = torch.tensor(0.005)  # Сильнее интегральная часть
simulator.control.Kd = torch.tensor(0.02)  # Больше демпфирование
simulator.control.deadband = torch.tensor(0.05)  # Более чувствительный
```

## 📦 Установка как пакет

```bash
# Установка в режиме разработки
pip install -e .

# Или обычная установка
pip install .

# С dev-зависимостями
pip install -e ".[dev]"
```

После установки доступна команда:
```bash
atomic-sim --n-scenarios 100 --duration 200 --format parquet
```

## 🎓 Для исследований

### Воспроизводимость

Всегда устанавливайте seed:
```python
import torch
import numpy as np

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

simulator = ReactorSimulator()
# ...
```

### Метаданные

Каждый датасет включает метаданные:
- Версия симулятора
- Git commit (если доступен)
- Seed для воспроизводимости
- Временная метка создания
- Параметры конфигурации

### Валидация данных

```python
# Проверка физической корректности
def validate_trajectory(df):
    # Энергетический баланс
    power_positive = (df['power_MW'] >= 0).all()
    
    # Температуры в физическом диапазоне
    temp_valid = (df['T_fuel_C'] > 0).all() and (df['T_fuel_C'] < 3000).all()
    
    # Реактивность в разумных пределах
    react_valid = (df['rod_reactivity_beta'] > -30).all()
    
    return power_positive and temp_valid and react_valid
```

## 📊 Статистика и анализ

### Анализ датасета

```python
import pandas as pd

df = pd.read_parquet('datasets/reactor_dataset.parquet')

print("Статистика датасета:")
print(f"  Всего записей: {len(df)}")
print(f"  Сценариев: {df['scenario_id'].nunique()}")
print(f"  Средняя мощность: {df['power_MW'].mean():.2f} МВт")
print(f"  Нарушений безопасности: {(~df['safe']).sum()}")

# Распределение по типам сценариев
print("\nТипы сценариев:")
print(df.groupby('scenario_type').size())
```

### Визуализация статистики

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Распределение мощности
df['power_MW'].hist(bins=50, ax=axes[0,0])
axes[0,0].set_title('Распределение мощности')

# Распределение температуры
df['T_fuel_C'].hist(bins=50, ax=axes[0,1])
axes[0,1].set_title('Распределение температуры топлива')

# Позиции стержней
df['rod_position_pct'].hist(bins=50, ax=axes[1,0])
axes[1,0].set_title('Распределение позиций стержней')

# Безопасность
df['safe'].value_counts().plot(kind='bar', ax=axes[1,1])
axes[1,1].set_title('Безопасность')

plt.tight_layout()
plt.savefig('dataset_statistics.png')
```

## 🚄 Оптимизация производительности

### Советы

1. **Используйте батч-симулятор** для массовой генерации
2. **float32 вместо float64** для скорости (если точность не критична)
3. **GPU** для батчей > 32
4. **Сэмплинг истории** (record_interval > 1) для экономии памяти
5. **torch.compile** для дополнительного ускорения (PyTorch 2.0+)

### Пример оптимизированной генерации

```python
import torch
from reactor.batch_simulator import BatchReactorSimulator

# GPU + float32 + большой батч
sim = BatchReactorSimulator(
    batch_size=64,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    dtype=torch.float32
)

# Генерация 6.4M шагов
trajectory = sim.run_batch(n_steps=100_000)

# Сохранение только каждого 10-го шага
trajectory_sampled = {
    k: v[:, ::10] for k, v in trajectory.items()
}
```

## 🔍 Отладка и диагностика

### Детальное логирование

```python
simulator = ReactorSimulatorAdvanced(enable_xenon=True)
simulator.reset()

def detailed_callback(state):
    print(f"t={state['time']:.1f}s:")
    print(f"  P={state['power']:.2f} MW, T_fuel={state['T_fuel']:.1f}°C")
    print(f"  ρ_total={state.get('reactivity', 0):.4f} β")
    if 'Xe_reactivity' in state:
        print(f"  ρ_Xe={state['Xe_reactivity']:.6f} β")

history = simulator.run(duration=100.0, callback=detailed_callback, callback_interval=10.0)
```

### Проверка численной стабильности

```python
# Проверить что n не становится отрицательным
assert (sim.neutronics.n > 0).all(), "n should always be positive"

# Проверить что реактивность в разумных пределах
total_react = history['reactivity']
assert (total_react > -10.0).all() and (total_react < 2.0).all()
```

## 📖 Дополнительные ресурсы

- **README.md** - основная документация
- **QUICKSTART.md** - быстрый старт
- **CHANGELOG.md** - история изменений
- **configs/** - примеры конфигураций
- **scenarios/** - примеры сценариев
- **examples/** - готовые примеры использования

## 🤝 Вклад в разработку

См. [CONTRIBUTING.md](CONTRIBUTING.md) для деталей.

## 📄 Цитирование

Если вы используете этот симулятор в исследованиях, пожалуйста, цитируйте:

```bibtex
@software{atomic_sim_2025,
  title = {Atomic Reactor Simulator: High-Fidelity Nuclear Reactor Simulation for ML/RL},
  author = {researchim-ai},
  year = {2025},
  url = {https://github.com/researchim-ai/atomic-sim},
  version = {1.1.0}
}
```

