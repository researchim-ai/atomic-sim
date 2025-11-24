<p align="center">
  <img src="src/imgs/atomic-sim.png" alt="Atomic Reactor Simulator" width="600"/>
</p>

# 🔬 Симулятор Атомного Реактора

Высокоточный симулятор атомного реактора на Python с использованием PyTorch. Включает физически корректные модели нейтронной кинетики, теплообмена и систему управления.

## 📋 Возможности

### Базовые
- **Нейтронная кинетика**: Уравнения точечной кинетики с 6 группами запаздывающих нейтронов
- **Тепловая модель**: Трехузловая модель (топливо, оболочка, теплоноситель)
- **Система управления**: Управляющие стержни с ручным и автоматическим (ПИД) управлением
- **Обратные связи**: Доплеровский эффект и температурная обратная связь по теплоносителю
- **Безопасность**: Мониторинг предельных значений и аварийная остановка (SCRAM)
- **Визуализация**: Графики в реальном времени и отчеты о симуляции

### Расширенные (v1.2.0+)
- **Пространственная кинетика (1D)**: Решение уравнения диффузии на осевой сетке (50+ узлов).
- **1D Теплогидравлика**: Учет профиля температур и переноса тепла теплоносителем (адвекция).
- **Выгорание топлива (Burnup)**: Моделирование изотопного состава (U-235, Pu-239) на протяжении кампании (18 мес).
- **Ксеноновые колебания**: Моделирование пространственной нестабильности (волны мощности).
- **Нелинейная worth-кривая**: S-образная зависимость реактивности от позиции стержней.
- **Система событий**: Аварии и возмущения (отказ насоса, выброс стержня, и т.д.).
- **Gymnasium интерфейс**: Готовые RL-среды для обучения агентов.
- **Батч-симуляция**: Векторизованная генерация датасетов (×22 быстрее).

## 🚀 Установка

### Базовая установка

```bash
# Клонировать репозиторий
git clone https://github.com/researchim-ai/atomic-sim.git
cd atomic-sim

# Установить зависимости
pip install -r requirements.txt
```

### Установка как пакет

```bash
# В режиме разработки
pip install -e .

# С dev-зависимостями
pip install -e ".[dev]"

# После установки доступна команда
atomic-sim --help
```

### Требования

**Базовые:**
- Python 3.8+
- PyTorch 2.0+
- NumPy 1.24+
- Matplotlib 3.7+
- SciPy 1.10+
- tqdm 4.65+

**Расширенные (для ML/RL):**
- Gymnasium 0.29+
- PyYAML 6.0+
- Pandas 2.0+
- PyArrow 12.0+

**Разработка:**
- pytest 7.4+
- pytest-cov 4.1+
- black 23.0+
- ruff 0.0.290+

## 📖 Быстрый старт

### Тестирование

```bash
# Запустить все тесты (34 теста)
pytest tests/ -v

# Быстрая проверка
python tests/test_simulator.py
```

### Базовая симуляция (0D)

```python
from reactor import ReactorSimulator
from reactor.visualization import plot_simulation_results

# Создание симулятора
simulator = ReactorSimulator(device='cpu')

# Запуск симуляции на 60 секунд
history = simulator.run(duration=60.0)

# Визуализация результатов
plot_simulation_results(history, save_path="results.png")
```

### Пространственная симуляция (1D)

```python
from reactor import ReactorSimulator1D

# Создание 1D симулятора
sim = ReactorSimulator1D(num_nodes=50)

# Шаг симуляции
state = sim.step(dt=0.01)
print(f"Axial Offset: {state['axial_offset']:.3f}")
```

## 🧪 Примеры

В директории `examples/` представлены детальные примеры:

### 1. Базовая симуляция
```bash
cd examples
python basic_simulation.py
```
Демонстрирует основные функции симулятора без активного управления.

### 2. Пространственная кинетика (1D)
```bash
python examples/spatial_kinetics_demo.py
```
Демонстрирует изменение профиля нейтронного потока при движении управляющих стержней.

### 3. Ксеноновые колебания
```bash
python examples/xenon_demo_1d.py
```
Симулирует 26 часов работы реактора, показывая возникновение и затухание пространственных ксеноновых волн.

### 4. Топливный цикл (Burnup)
```bash
python examples/burnup_demo.py
```
Демонстрирует 18-месячную кампанию реактора: падение концентрации U-235, наработку плутония и кривую снижения бора (Letdown Curve).

### 5. Автоматическое управление
```bash
python examples/control_demo.py
```
Показывает работу связки "Стержни + Бор" для управления мощностью и формой поля.

### 6. Аварийная ситуация
```bash
python emergency_scram.py
```
Симуляция потери охлаждения и аварийной остановки реактора.

### 7. Веб-интерфейс (UI)
```bash
python3 gui/app.py
```
Запускает профессиональный пульт оператора в браузере (по умолчанию localhost:8081) с живыми графиками, управлением стержнями и мониторингом безопасности.

## 🚀 Расширенные возможности

### Генерация датасетов

```bash
# Генерация 100 сценариев
python generate_dataset.py --n-scenarios 100 --duration 100 --format parquet

# Q&A для LLM
python dataset_builders/qa_from_traces.py --input datasets/*.parquet --output datasets/qa.jsonl
```

### Reinforcement Learning

```python
from envs.gym_reactor import ReactorEnv
import gymnasium as gym

# Создание среды
env = ReactorEnv(target_power=95.0)

# Или через Gymnasium
env = gym.make('AtomicReactor-v0')

obs, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
```

### Батч-симуляция (высокая производительность)

```python
from reactor.batch_simulator import BatchReactorSimulator

# 16 параллельных симуляций
sim = BatchReactorSimulator(batch_size=16, dtype=torch.float32)
trajectory = sim.run_batch(n_steps=100_000)

# Производительность: ~125,000 шагов/сек (×22 быстрее)
```

## 📚 Полная документация

Вся подробная документация находится в директории **[docs/](docs/)**:

- **[Быстрый старт](docs/QUICKSTART.md)** - начните здесь!
- **[Установка](docs/INSTALLATION.md)** - детальное руководство
- **[Архитектура](docs/ARCHITECTURE.md)** - структура и дизайн
- **[Математика](docs/MATHEMATICS.md)** - все уравнения
- **[Физика](docs/PHYSICS_MODELS.md)** - подробная физика (100+ формул)
- **[Расширенные функции](docs/ADVANCED_FEATURES.md)** - ML/RL, датасеты
- **[Производительность](docs/PERFORMANCE.md)** - оптимизация
- **[Contributing](docs/CONTRIBUTING.md)** - для разработчиков

### Генерация PDF документации

```bash
pip install -r requirements-docs.txt
playwright install chromium
mkdocs build
python scripts/export_pdf.py
```

Подробнее: [docs/HOW_TO_GENERATE_PDF.md](docs/HOW_TO_GENERATE_PDF.md)

## 💡 Примечания

⚠️ **Дисклеймер**: Этот симулятор создан в образовательных целях. Он использует упрощенные модели и не должен применяться для проектирования реальных ядерных установок без надлежащей валидации и экспертизы.

## 📞 Контакты

Для вопросов и предложений создайте issue в репозитории проекта.

---

**Разработано с использованием PyTorch 🔥**
