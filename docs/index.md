# Добро пожаловать в документацию Atomic Simulator!

<div align="center">
<img src="images/atomic-sim.png" alt="Atomic Reactor Simulator" style="max-width: 400px; width: 100%; height: auto;">
</div>

## 🔬 О проекте

**Atomic Simulator** - это высокоточный симулятор ядерного реактора, разработанный на Python с использованием PyTorch. Проект создан для:

- 🎓 **Обучения** физике ядерных реакторов
- 🤖 **Машинного обучения** и reinforcement learning
- 📊 **Генерации датасетов** для обучения LLM и RL моделей
- 🔬 **Исследований** в области управления реакторами
- 🛡️ **Анализа безопасности** и аварийных ситуаций

## ✨ Ключевые особенности

### Физически корректные модели

- ✅ Уравнения точечной кинетики с **6 группами запаздывающих нейтронов**
- ✅ Трехузловая **тепловая модель** (топливо, оболочка, теплоноситель)
- ✅ Отрицательные **температурные обратные связи** (Доплера, плотность)
- ✅ Модель **отравления Xe-135/I-135**
- ✅ Нелинейная **S-образная кривая** ценности стержней

### Система управления

- ✅ **10 управляющих стержней** с реалистичной динамикой
- ✅ **ПИД-регулятор** с антиwindup и deadband
- ✅ Ручное и автоматическое управление
- ✅ Аварийная остановка (**SCRAM**)

### ML/RL интеграция

- ✅ **Gymnasium интерфейс** (дискретные и непрерывные действия)
- ✅ **Батч-симуляция** (×22 ускорение через PyTorch векторизацию)
- ✅ **Генератор датасетов** (Parquet/JSONL/CSV)
- ✅ **Q&A генератор** для обучения LLM

### Качество и надежность

- ✅ **34 теста** (100% проходят)
- ✅ Полное **тестовое покрытие** физических моделей
- ✅ **CI/CD pipeline** для автоматической проверки
- ✅ **Production-ready** код

## 📊 Производительность

| Режим | Производительность | Применение |
|-------|-------------------|------------|
| Обычная симуляция | 5,500 шагов/сек | Обучение, разработка |
| Батч (B=16, CPU) | 125,000 шагов/сек | Генерация датасетов |
| Батч (B=64, GPU) | 500,000+ шагов/сек | Массовая генерация |

## 🚀 Быстрый старт

### Установка

```bash
git clone https://github.com/researchim-ai/atomic-sim.git
cd atomic-sim
pip install -r requirements.txt
```

### Первая симуляция

```python
from reactor import ReactorSimulator

sim = ReactorSimulator()
history = sim.run(duration=100.0)

# Визуализация
from reactor.visualization import plot_simulation_results
plot_simulation_results(history, save_path="results.png")
```

### Тестирование

```bash
# Все тесты
pytest tests/ -v

# Быстрая проверка
python tests/test_simulator.py
```

## 📚 Структура документации

### Для начинающих

1. **[Основные понятия](BASIC_CONCEPTS.md)** - что такое ядерный реактор простыми словами
2. **[Быстрый старт](QUICKSTART.md)** - первые шаги с симулятором
3. **[Установка](INSTALLATION.md)** - детальное руководство
4. **[Глоссарий терминов](GLOSSARY.md)** - объяснение основных понятий ядерной физики

### Технические детали

5. **[Архитектура](ARCHITECTURE.md)** - структура и дизайн системы
6. **[Математические модели](MATHEMATICS.md)** - все уравнения и выводы
7. **[Физические модели](PHYSICS_MODELS.md)** - подробное описание физики

### Для продвинутых пользователей

8. **[Расширенные возможности](ADVANCED_FEATURES.md)** - ML/RL, датасеты, события
9. **[Производительность](PERFORMANCE.md)** - оптимизация и benchmark

### Для разработчиков

10. **[Руководство по вкладу](CONTRIBUTING.md)** - как участвовать
11. **[История изменений](CHANGELOG.md)** - версии и обновления
12. **[Сводка проекта](PROJECT_SUMMARY.md)** - метрики и статистика

## 🎯 Примеры использования

### Обучение RL агента

```python
from envs.gym_reactor import ReactorEnv
from stable_baselines3 import PPO

env = ReactorEnv(target_power=95.0)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

### Генерация датасета

```bash
python generate_dataset.py --n-scenarios 1000 --duration 100 --format parquet
```

### Отравление ксеноном

```python
from reactor.simulator_advanced import ReactorSimulatorAdvanced

sim = ReactorSimulatorAdvanced(enable_xenon=True)
history = sim.run(duration=50000.0)  # 14 часов
```

## 📈 Статус проекта

| Метрика | Значение |
|---------|----------|
| Версия | 1.1.0 |
| Статус | Production Ready ✅ |
| Тесты | 34/34 (100%) ✅ |
| Строк кода | ~4,600 |
| Документация | Полная на русском |
| Лицензия | MIT |

## 🔗 Полезные ссылки

- [GitHub Repository](https://github.com/researchim-ai/atomic-sim)
- [Issue Tracker](https://github.com/researchim-ai/atomic-sim/issues)
- [Примеры кода](https://github.com/researchim-ai/atomic-sim/tree/main/examples)

## 📞 Контакты

Для вопросов и предложений:
- Создайте [Issue](https://github.com/researchim-ai/atomic-sim/issues)
- См. [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Начните с [Быстрого старта](QUICKSTART.md) →**

