<p align="center">

Текущий статус и очередь оставшихся работ: [локальная проверка и научные направления](docs/CURRENT_STATUS_RU.md).
  <img src="docs/assets/branding/atomic-sim.png" alt="Atomic Reactor Simulator" width="600"/>
</p>

# 🔬 Симулятор Атомного Реактора


[Точность и энергия управления](docs/THERMAL_ENERGY_CONTROL_RU.md): train-only калибровка линейной политики и PI, 64 калибровочных и 48 оценочных эпизодов с учётом задержанной энергии после управления. Раздельные IAE/энергия и фиксированный безразмерный objective. 10 новых тестов; всего 1196 тестов прошли. Один полный test с хвостом точно повторён из wheel.

[Обучение теплового агента](docs/THERMAL_AGENT_TRAINING_RU.md): Gymnasium-среда с задержкой/скоростью нагревателя и шумом, train-only линейная imitation-политика, 12 test-эпизодов и два переноса сетки. Средняя IAE ниже фиксированного PI на 4.1%, энергия выше в 2.20 раза; общее превосходство не заявляется. 15 новых тестов, всего 1186 тестов прошли; выбранный эпизод точно повторён из wheel.

[Моделирование решений для агента](docs/AGENT_DECISION_MODELING_RU.md): новый atomic_model_decision оценивает ветви из одного checkpoint, объясняет ограничения/отказы и применяет выбранную ветвь после полного replay и проверки актуальности. Шесть квалификационных путей, 28 новых тестов; всего 1171 тест прошёл. Один полный путь точно повторён из wheel. Физические исходники сохранены; уникальность относительно всех конкурентов и непрерывная гарантия ограничений не заявлены.

[Сравнительный стенд моделёров и агентов](docs/MODELER_COMPARISON_RU.md): 6/6 независимых нагреваемых траекторий OpenModelica/MSL, 36 статических и 480 маршрутных EOS-проверок CoolProp; 240 test-эпизодов hold/PID/rule с пересчётом отчёта. В повторном EOS-замере CoolProp IF97 быстрее примерно в 2.35 раза. 34 новых теста; всего 1143 теста прошли. Общее превосходство и сравнение внешних сред обучения не установлены.

[Нагреваемый жидкостный HLLC](docs/HEATED_HLLC_RU.md): отдельный HeatedHEMNetwork с предписанными мощностями, линейной стенкой и энтропийным учётом. 18 нативных путей, шесть временных эталонов DOP853 с общим RHS; 6/6 обязательных сравнений на шаге 5 мкс. 28 новых тестов; всего 1109 тестов прошли. Выбранный путь точно повторён из wheel. Прежний адиабатический пакет и его подпись сохранены.

[Транзакционность ИИ-команд и restart](docs/HLLC_TRANSACTIONS_RU.md): шесть путей на двух сетках и трёх схемах, 18 точных восстановлений и 234 атомарных отказа. 24 новых теста; всего 1081 тест прошёл. Один полный путь с 39 отказами точно воспроизведён из установленного wheel вне checkout. Физическая область HLLC не расширялась.

Разветвлённая сеть: четыре узла, пять труб, два цикла; естественный разворот потока, балансы портовых реакций, смена ориентации и точный restart. Приняты семь нативных траекторий и шесть независимых временных решений; обязательные сравнения на меньшем шаге — 6/6. Добавлены 18 тестов; две выбранные траектории точно повторены из установленного wheel. [Протокол и границы применимости](docs/HLLC_BRANCHING_RU.md).
Открытый моделёр реакторных систем на Python для ИИ, RL и LLM-агентов. С версии 2.0 есть проверяемое ядро установки ВВЭР (статус `verified` по аналитическим эталонам и балансам) и прежняя учебная ветка. Это не цифровой двойник действующего блока и не `plant_referenced` модель.

Локальная проверка библиотеки одной командой: `python -m scripts.verify_local_library`.

Профиль численной проверки физики: `python -m scripts.verify_physics` —
EOS/фазы, балансы, сходимость, restart, связанный reduced-PWR и 1D-нейтроника.
[Область проверок и оставшиеся физические задачи](docs/PHYSICS_VERIFICATION_RU.md).
Для пространственных фазовых тестов добавьте `--include-phase-mesh`.
Полные длительные режимы и replay: `python -m scripts.verify_pwr_horizons`
([состав и ограничения](docs/PWR_FULL_HORIZONS_RU.md)).
[Тесты, сборка wheel/sdist, установка вне исходников и replay](docs/LOCAL_VERIFICATION_RU.md).
CI и GitHub-доступ не требуются.

Исследовательская multiphysics-связка и переносимые проекты: `atomic-multiphysics --output artifacts/research.json`; [OpenMC/MGXS, радиальный твэл, канал, depletion-методы и точные ограничения](docs/FINAL_17_IMPLEMENTATION_RU.md). Явная covariance и held-out coverage: `atomic-uncertainty examples/research/uncertainty.json --output artifacts/uncertainty.json`. Локальный порядок проверки описан выше; экспериментальные claims требуют отдельных данных.

Кампании обучения на основном PWR-ядре: `atomic-training examples/training/cases.json artifacts/my-training --workers 2`; [изолированные workers, точный EOS-кэш, cancel/resume и solver tiers](docs/AI_09_10_12_TRAINING_RUNTIME_RU.md). Ускорение 5,277× измерено только на объявленных коротких номинальных эпизодах; физическая валидация не заявлена.

Новый ограниченный HEM: [EOS до16 МПа, фазовые переходы, кипящая теплоотдача и численная квалификация](docs/AI_06_QUALIFIED_HEM_RU.md); `atomic-hem-qualified demo --output artifacts/qualified_demo.json`. Экспериментальная валидация двухфазной динамики остаётся отдельной задачей.

Первичные экспериментальные данные: [HIPER, неопределённости и проверка применимости](docs/AI_06_PRIMARY_DATA_ASSESSMENT_RU.md); `atomic-hem-assessment screen --output artifacts/hiper_screen.json`.

Проточный steady HEM: [уравнения, фазовые события и отрицательный HIPER benchmark](docs/AI_06_STEADY_THROUGHFLOW_RU.md); `atomic-hem-throughflow demo --output artifacts/throughflow_demo.json`. Это отдельная стационарная постановка, без заявления физической приёмки.

Model/data study: `python -m examples.eos_study_workflow --output artifacts/my-study`, затем `atomic-model-project-ui artifacts/my-study/study --study`. [Ревизии, эталоны, supervised comparisons и переносимость](docs/AI_11_MODEL_DATA_STUDY_RU.md).

Сравнение результатов: `python -m examples.eos_results_workflow --output artifacts/my-results`; [native replay, траектории и переносимые ZIP](docs/AI_11_RESULT_COMPARISON_RU.md).

Внешний локальный агент: `python -m examples.eos_external_agent_workflow --output artifacts/my-external-agents`; [JSONL-контракт, состояние, разрешения и лимиты](docs/AI_11_EXTERNAL_AGENT_CONTRACT_RU.md).

Открытый EOS benchmark: `python -m examples.eos_arena_workflow --output artifacts/my-arena`; [протокол, train/dev/test, метрики и переходы](docs/AI_13_EOS_OPEN_BENCHMARK_RU.md).

Проверка установки и recovery: `atomic-model-release doctor`, `atomic-model-release use-cases`, `atomic-model-release golden`; [исполняемый runbook](docs/release/RUNBOOK_RU.md).

Сохраняемые параллельные кампании: `python -m examples.eos_rollout_workflow --output artifacts/my-rollouts`; [лимиты, пауза, продолжение и границы гарантий](docs/AI_10_DURABLE_ROLLOUTS_RU.md).

Проверка по опубликованному эксперименту NIST: `atomic-model-validate run-pipe --output artifacts/my-nist-pipe`; [данные, критерии и ограничения](docs/AI_09_NIST_PIPE_EXPERIMENT_RU.md).

Совместные события гидравлической сети: `python -m examples.hybrid_thermal_workflow --output artifacts/my-hybrid`; [календарь, защиты, online-команды и транзакционный restart](docs/AI_05_HYBRID_ORCHESTRATION_RU.md).

Равновесная вода/пар IF97 и закрытый жёсткий объём: `atomic-hem demo --output artifacts/my-hem`; [EOS, фазовые границы и границы применимости](docs/AI_06_EQUILIBRIUM_WATER_FOUNDATION_RU.md).

Гомогенное двухфазное трение McAdams/Cicchitti: `atomic-hem-mixture run examples/hem_mixture.json --output artifacts/my-mixture`; [модель, проверка и границы применимости](docs/AI_06_HOMOGENEOUS_FRICTION_RU.md).

HEM-сеть с конечными узлами, ветвлениями, трением Дарси и ламинарной/Gnielinski теплоотдачей: `atomic-hem-network demo --output artifacts/my-hem-network`; [постановка и ограничения](docs/AI_06_HEM_NETWORK_RU.md), [замыкания и CHF](docs/AI_06_HEM_CLOSURES_RU.md), [экспериментальные свидетельства](docs/AI_06_HEM_EXPERIMENTAL_EVIDENCE_RU.md).

Конечный HEM-канал с торцевыми объёмами и теплоёмкими стенками: `atomic-hem-system demo --output artifacts/my-finite-system`; [уравнения, CLI и квалификация](docs/AI_06_FINITE_CHANNEL_WALL_SYSTEM_RU.md).

Периодический HEM-канал с массой, импульсом и полной энергией: `atomic-hem-flow demo --output artifacts/my-hem-flow`; [схема, проверки и ограничения](docs/AI_06_PERIODIC_HEM_FLOW_RU.md).

## Возможности ядра 2.0

**Локальные LLM tools:** [JSONL-протокол, наблюдения, действия и replay](docs/AI_11_LLM_TOOLS_RU.md).
Запуск: `python -m atomic_agent_tools` или `atomic-agent-tools` после установки.
LLM-провайдер, API-ключи и покупатели не требуются; пример клиента —
`python -m examples.llm_tools_episode`.

**Публичный EOS runtime:** [Python API, agent tools и эксперименты с resume](docs/AI_11_PUBLIC_EOS_RUNTIME_RU.md).
Пример: `python -m examples.eos_agent_workflow`; CLI: `atomic-model-experiment`.
Это отдельная экспериментальная синтетическая модель, не новый паспорт ВВЭР.

**Проекты и кампании агентов:** [переносимый EOS workspace, два baseline, отчёты и архив](docs/AI_11_EOS_WORKSPACE_RU.md).
CLI: `atomic-model-workspace`; полный пример: `python -m examples.eos_workspace_workflow`.
Кампания сохраняет принятые шаги и восстанавливается после паузы или timeout worker.

**Производительность EOS:** [фиксированный CPU-бюджет, профиль и RSS](docs/AI_10_EOS_PERFORMANCE_RU.md).
Запуск: `atomic-model-bench examples/eos_performance.json --output artifacts/eos_bench`.

**Повторное использование EOS worker:** [ограниченный пул и честное сравнение с отдельными процессами](docs/AI_10_EOS_WORKER_POOL_RU.md).
Запуск: `atomic-model-pool examples/eos_pool_comparison.json --output artifacts/eos_pool`.

**Оптимизация source checks:** [быстрый обход с полным свежим чтением файлов](docs/AI_10_SOURCE_SCAN_OPTIMIZATION_RU.md).
Частота проверок, численные методы и допуски сохранены; кеширования source bytes нет.

**Редактор проекта:** [preview, атомарные изменения, история и восстановление конфигурации](docs/AI_11_PROJECT_EDITOR_RU.md).
CLI: `atomic-model-project`; пример: `python -m examples.eos_project_edit_workflow`.

**Интерфейс редактора:** [локальные формы, паспорт, preview, сохранение и история](docs/AI_11_LOCAL_PROJECT_UI_RU.md).
Запуск: `python -m atomic_model_project_ui STORE --port 8765`.

**Эталонные данные:** [CSV import, единицы, split guards, независимые метрики и report replay](docs/AI_09_REFERENCE_COMPARISON_RU.md).
Пример: `python -m examples.eos_reference_workflow`; CLI: `atomic-model-reference`.

**Действующий план следующих работ:** [промышленная платформа для ИИ — 14 пакетов, 109 задач](docs/AI_INDUSTRIAL_EXECUTION_PLAN_RU.md).
AI-00/AI-01 выполнены: единые критерии приёмки и контракт среды. AI-02 добавляет [benchmark из 80 кейсов и трёх контроллеров](docs/AI_02_BENCHMARK_RU.md).
AI-03: [консервативный Euler-компонент с конвекцией импульса](docs/AI_03_EULER_LIQUID_RU.md), отдельный CLI и schema 3; пространственная/акустическая квалификация остаётся открытой.
AI-04: [сетка до 512 ячеек, точность интегрирования и матрица допуска](docs/AI_04_FRONT_REFINEMENT_RU.md), 7/7 задач в компонентном численном scope. Допуск относится к перечисленным компонентным задачам; акустическая и промышленная квалификация установки остаются открытыми.
AI-05: типизированная сеть, клапан и [связанная сборка насоса, вала и газовых накопителей](docs/AI_05_PUMP_STORAGE_RU.md); 4/8 задач приняты. Доступны экспериментальные EOS-сборки и замкнутый PI с отдельными паспортами; общая приёмка сети открыта. Контакт обратного клапана, общая разветвлённая сеть и паровой компенсатор остаются открытыми.
Статусы и доказательства закрытия: [машинный журнал](validation/ai_industrial_workplan_v1.json).

Первый реализованный пакет: [контракт ИИ-среды, episode snapshot и replay CLI](docs/AI_00_01_DELIVERY_RU.md).

Актуальные следующие этапы, критерии промышленной приёмки и сравнения с другими платформами: [план развития 3.0](docs/PRODUCT_DEVELOPMENT_PLAN_RU.md).

Проверка от 30 октября 2026, оставшиеся технические барьеры и исторический коммерческий контекст (заменён open-source scope 1.49.0): [старый roadmap](docs/INDUSTRIAL_STARTUP_ROADMAP_RU.md).

Исполняемый пакет расчётного эксперимента с checkpoint, CSV-траекторией, балансами и manifest: [запуск пилотного эксперимента](docs/PILOT_EXPERIMENTS_RU.md). После установки пакета доступна команда `atomic-experiment`.

Проверенные результаты поставки и оставшиеся ограничения: [отчёт реализации](docs/PILOT_DELIVERY_RU.md).

Прогресс численного ядра: [исправление точности EOS, номинал 60 с и сравнение допусков](docs/EOS_STABILITY_RU.md).

Продолжение: [локальный Newton IF97, стоимость отчёта и результаты длинных попыток](docs/THERMO_PERFORMANCE_RU.md).

Следующий этап: [история BDF, точные принятые состояния и результаты длинных прогонов](docs/BDF_CONTINUATION_RU.md).

Текущий результат: [часовой номинал на двух точностях и центральный якобиан](docs/CENTRAL_JACOBIAN_RU.md).

Продолжение: [120-секундный переход нагрузки на двух точностях и выбор Radau](docs/RADAU_TRANSIENT_RU.md).

Следующий результат: [точные упоры стержня, АЗ и выбег насоса](docs/SCRAM_COASTDOWN_RU.md).

Продолжение: [исправление EOS, восстановление checkpoint и предел выбега](docs/EOS_RESTART_RU.md).

Текущий этап: [контролируемый останов и уточнение границы паспорта](docs/MODEL_BOUNDARY_RU.md).

Добавлен [пакет из 10 задач по совместимому полному энергетическому балансу](docs/BATCH12_TOTAL_ENERGY_RU.md): явный новый режим, механический ledger и граничная полная энтальпия. Это баланс редуцированной модели, не квалификация полного Euler или установки.
Стационарная инициализация полной энергии расширена на [обратный поток](docs/REVERSE_TOTAL_INITIALIZATION_RU.md), включая отражение несимметричных сопротивлений и температур резервуаров.
Генератор обратного теплового импульса и JSON CLI проверены [двумя зеркальными расчётами](docs/TOTAL_ENERGY_REFLECTION_RU.md) с независимыми энергетическими бюджетами.
Добавлен [стационарный решатель MUSCL с полной энергией](docs/MUSCL_TOTAL_EQUILIBRIUM_RU.md): нагрев, оба направления потока, приёмка производственным RHS и три связанных перехода на 8 с.

Следующий [пакет из 10 задач: аналитический разреженный якобиан](docs/BATCH11_ANALYTIC_JACOBIAN_RU.md) реализован; конкретный тепловой MUSCL-сценарий прошёл сеточный и временной критерии 4/8/16. Общая промышленная валидация остаётся открытой.

Выполнен [пакет из 10 задач: MUSCL, допуски и диагностика](docs/BATCH10_TRANSPORT_RU.md). Второй порядок подтверждён для отдельного скалярного оператора; полная сеточная квалификация IF97 остаётся открытой.

Добавлены [согласованная инициализация и переходные сеточные проверки](docs/TRANSIENT_INITIALIZATION_RU.md). Переходная приёмка на 4/8/16 пока не пройдена.

Новая проверка: [сгущение сетки при неизменной геометрии трубы](docs/PIPE_MESH_RU.md), измеренный первый порядок стационарного нагрева.

Добавлен [канал с динамикой давления и расхода](docs/PRESSURE_DRIVEN_CHANNEL_RU.md): однофазная жидкость, импульс соединений, стенка и реверс.

Следующий компонент: [канал с тепловой инерцией стенки](docs/WALL_CHANNEL_RU.md).

Новый компонент: [консервативный изобарический канал кипения и его проверки](docs/ISOBARIC_CHANNEL_RU.md).

- Точечная кинетика с шестью группами, сверенная с матричной экспонентой.
- IAPWS-IF97 в однофазных областях 1, 2, 3 и 5. Смесь задаётся паросодержанием; проекция на купол только по явному запросу.
- Конечно-объёмная диффузия, радиальный твэл, сеть контрольных объёмов с обратным расходом.
- Сосредоточенная петля ВВЭР и отдельный канальный паспорт РБМК.
- Защита, датчики, непрерывная среда агента и повтор сценария.

Учебная ветка `reactor/` сохранена для прежних примеров. Её батч-симулятор — отдельная редуцированная модель, а не ускорение ядра.

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
- **Батч-симуляция**: отдельная редуцированная модель для черновой генерации, не эквивалент ядра 2.0.

## 🚀 Установка

### Базовая установка

```bash
# Клонировать репозиторий
git clone https://github.com/researchim-ai/atomic-sim.git
cd atomic-sim

# Установить библиотеку и зависимости
python -m pip install -e .
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
- Python 3.10+
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

## Структура проекта

Код библиотеки — в `src/`, приложения — в `apps/`, тесты — в `tests/`,
примеры — в `examples/`, документация — в `docs/`. Научные данные и свидетельства
сохранены в `data/`, `scenarios/`, `validation/`; локальные результаты — в `artifacts/`.
[Подробная структура и команды разработки](docs/PROJECT_STRUCTURE_RU.md).

## 📖 Быстрый старт

### Тестирование

```bash
# Запустить локальные тесты
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
python3 apps/gui/app.py
```
Запускает профессиональный пульт оператора в браузере (по умолчанию localhost:8081) с живыми графиками, управлением стержнями и мониторингом безопасности.

## 🚀 Расширенные возможности

### Генерация датасетов

```bash
# Генерация 100 сценариев
atomic-sim --n-scenarios 100 --duration 100 --format parquet

# Q&A для LLM
python -m dataset_builders.qa_from_traces --input datasets/*.parquet --output datasets/qa.jsonl
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

- **[Технический план открытого промышленного симулятора](docs/PRODUCT_DEVELOPMENT_PLAN_RU.md)** — физика, численные методы, архитектура и критерии валидации.
- **[Оценка текущей готовности](docs/READINESS_REVIEW_RU.md)** — результаты аудита и подтверждённые ограничения.
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
pip install -r requirements/docs.txt
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

Условная идентифицируемость Chen: [метод, ограничения и CLI](docs/AI_09_CLOSURE_IDENTIFIABILITY_RU.md). `python -m atomic_model_hem_identification analyse --output report.json --require-calibratable` сохраняет диагностику и возвращает код 3 при отказе в калибровке.

Пространственная многогрупповая диффузия: [контракт, примеры, квалификация и ограничения](docs/AI_08_SPATIAL_DIFFUSION_RU.md). `atomic-spatial` / `python -m atomic_model_spatial` предоставляет source/k/transient расчёты, локальные предшественники и restart с полной проверкой истории.

Согласованная инициализация нагреваемой сети и проверка полных горизонтов ВВЭР: [реализация, CLI и сохранённые отказы](docs/AI_05_06_07_SYSTEM_PROGRESS_RU.md).

Новая отдельная конфигурация `pwr-reduced-flow-controlled`, транзакционные сессии и независимые фазовые проверки описаны в [расширении AI-05/06/07](docs/AI_05_06_07_EXTENDED_MODEL_RU.md). Экспериментальная аттестация этой конфигурации не заявлена.

Единый интерфейс нового PWR-ядра для Python и Gymnasium: [сквозной пример агента, checkpoint и campaign replay](docs/PWR_AGENT_QUICKSTART_RU.md).

Управляемый PWR через локальный JSONL: `python -m examples.llm_controlled_episode`; [контракт LLM tools 1.2](docs/PWR_LLM_TOOLS_RU.md).

Последний численный блок: [ускорение EOS и полная PWR-матрица 7200 с](docs/EOS_OPTIMIZATION_RU.md),
[динамический benchmark OpenModelica со стенкой](docs/MODELICA_DYNAMIC_BENCHMARK_RU.md).
Пройдено 1196 тестов актуальной поставки; предыдущий физический профиль — семь групп.
[Граница насыщения и независимый расчёт стенки](docs/PHASE_BOUNDARY_COMPARISON_RU.md):
16/16 native траекторий, 103 одинаковых rho/T-состояния и внешний sweep шага/допуска.
Внешние фазовые
несовпадения сохранены; они не объявлены положительной приёмкой.
[Аудит коэффициентов MSL и корней перехода](docs/MSL_IF97_EVENT_AUDIT_RU.md):
native 16/16; исходная MSL 38/40, явно исправленная диагностическая копия 40/40.
Копия с поправкой только коэффициентов проходит строгую проверку времени на 6/8.
[Согласованные производные насыщения](docs/MSL_SATURATION_TANGENT_RU.md)
устранили остаточный сдвиг в отдельном диагностическом варианте: 40/40
динамических сравнений и 8/8 времён событий при прежних допусках.
[Неоднородная сеть с независимой Modelica-сборкой](docs/HETEROGENEOUS_NETWORK_BENCHMARK_RU.md):
32/32 внешних сравнений, 32/32 строгих сравнений с DOP853/Radau и измеренная
сходимость размытия контакта. Сравнивается одна схема Русанова в двух реализациях,
не готовая Modelica.Fluid-сеть и не экспериментальная установка.
[Новый опциональный HLLC для жидкости](docs/HLLC_CONTACT_GRAPH_RU.md)
сохранил неподвижный контакт на пяти сетках, где Русанов размывал его на 0,28–0,65 К.
Аналитические контакты, независимые временные решения и restart приняты.
Влияние на скорость измерено отдельно: универсального ускорения нет.

Независимое сравнение с другими проектами:
[CoolProp / OpenMC benchmark, результаты и оставшиеся этапы](docs/EXTERNAL_BENCHMARK_RU.md).
Первый прогон нашёл и позволил исправить коэффициент IF97; в измеренном EOS/API
профиле до ускорения CoolProp был быстрее примерно в 9,7 раза; после ускорения
измеренный разрыв составляет примерно 2,3 раза. Общего лидерства симулятора
по этому сравнению не заявляем. Старые source-bound отчёты относятся к
предыдущему физическому ядру и автоматически не перенесены.

Проверка физической точности: [шаг и допуски на полных PWR-горизонтах](docs/PWR_REFINEMENT_RU.md),
[сетевой сброс с flashing и уточнением сетки](docs/PHASE_RELEASE_RU.md),
[разветвлённые приёмники и эквивалентность топологий](docs/PHASE_BRANCHING_RU.md),
[тепловые фазовые циклы с точным запасом энергии](docs/PHASE_CYCLES_RU.md),
[машинные паспорта замыканий HEM](docs/HEM_CLOSURE_AUDIT_RU.md).
Эти численные проверки не заменяют [отбор экспериментальных данных](docs/EXPERIMENTAL_DATA_CANDIDATES_RU.md).

## Сайт проекта

Русская и английская версии сайта находятся в [`web/`](web/README.md).
Локальный запуск: `cd web` → `npm run dev`. Статическая сборка: `npm run build`.

[Внешняя HLLC-проверка](docs/EXTERNAL_HLLC_BENCHMARK_RU.md): 48/48 сравнений
с независимой Modelica-реализацией и неизменённой MSL, включая давления 2–14 МПа.
Полный стенд воспроизведён из установленного wheel.

[Нелинейный эталон ударных волн и разрежений](docs/LIQUID_RIEMANN_ORACLE_RU.md):
15 случаев на IF97, независимое от HLLC волновое решение, измеренные ошибки
приближённого потока и направление улучшения его диссипации.

[Локальный HLLC](docs/LOCAL_HLLC_RU.md): opt-in `scheme='hllc-local'`,
проверка звёздных состояний и сохранённый глобальный CFL. На широком
начальном разрыве ошибка массового потока относительно нелинейного эталона
снижена с 41,97% до 1,98%. Балансы и restart проверены отдельным стендом.

[Пространственная проверка HLLC](docs/HLLC_SPATIAL_CONVERGENCE_RU.md):
полный нелинейный волновой профиль и консервативные средние по ячейкам.
На сетках 16/32/64 ошибки давления и импульса уменьшаются; локальный HLLC
снижает ошибки профиля на самой мелкой сетке на 7–8% в двух проверенных разрывах.

[Внешняя проверка локального HLLC](docs/EXTERNAL_LOCAL_HLLC_RU.md):
48/48 сравнений с независимой нативной Modelica-сборкой и исходной MSL;
отдельно проверены отказы по звёздным сигналам и глобальной границе волн.
Весь стенд повторён из установленного пакета.

[HLLC до 40 мс](docs/HLLC_LONG_TRANSIENTS_RU.md): 12 native-траекторий,
независимые Radau/BDF, три шага и точный restart в 20 мс.
Новый стенд поддерживает неравномерную временную сетку и публикует ошибки
крупных шагов отдельно от обязательной приёмки самого мелкого.

[Ускорение HLLC без изменения решения](docs/HLLC_SPEED_OPTIMIZATION_RU.md):
точный кэш звуковых характеристик, побитное сравнение с архивным baseline,
пять парных замеров и повторная внешняя квалификация. На выбранных local-HLLC
профилях измерено 1,10–1,53×; универсальное преимущество не заявляется.

[Быстрый жидкостный EOS-путь для HLLC](docs/HLLC_LIQUID_EOS_FAST_PATH_RU.md):
одна Gibbs-оценка для состояния и якобиана, исходные residual/seeds и полный
адаптер на остальных ветвях. На выбранных волновых профилях — ещё 1,05–1,12×
при побитном совпадении наблюдений. Общий EOS других моделей не изменён.

[Прямой RHS HLLC](docs/HLLC_DIRECT_RHS_RU.md): устранён повторный транспорт
Русановым, сохранены ledger-поля и проверены все биты RHS/aux. В выбранных
профилях — 1,16× на контакте и 1,01–1,03× на волнах; signed-zero check усилен.
