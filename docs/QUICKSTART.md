# Быстрый старт

Нужен Python 3.10 или новее. Полная локальная проверка воспроизведена на
Linux x86_64 с Python 3.12.9. Из корня репозитория установите библиотеку:

```bash
python -m pip install -e '.[dev]'
```

Код находится в `src/`; editable-установка делает его доступным командам и
дочерним процессам. Подробности: [установка](INSTALLATION.md) и
[структура проекта](PROJECT_STRUCTURE_RU.md).

## Первый эпизод исследовательского PWR

```python
from atomic_pwr_agent import Episode

episode = Episode(duration_s=0.1, max_steps=10, seed=17)
observation = episode.observe()
result = episode.step({"throttle": 0.95},
                      observation_id=observation["observation_id"])
print(result["observation"]["time_s"])
print(result["observation"]["thermal_power_w"])
restored = Episode.restore(episode.checkpoint())
assert restored.observe() == episode.observe()
```

Это редуцированная исследовательская модель `pwr-reduced-flow-controlled`,
не цифровой двойник действующего блока. Наблюдение содержит полное состояние
и схему единиц; seed пока служит метаданными, не случайной физической инициализацией.

## Примеры и проверки

```bash
# Python: решения агента, checkpoint, повтор в training runtime
python -m examples.training.agent_episode artifacts/my-agent-episode.json

# Настоящий локальный JSONL-процесс для LLM-клиента
python -m examples.llm_controlled_episode

# Проверки основных интерфейсов
python -m pytest tests/test_pwr_agent.py tests/test_controlled_agent_tools.py -q
```

Файл результата первого примера должен быть новым. Примеры находятся в checkout;
CLI и Python-пакеты также работают после установки wheel.
Для полной проверки поставки дополнительно нужен `build`:

```bash
python -m pip install build
python -m scripts.verify_local_library
```

Команда проверяет тесты, сборку, установку вне checkout и воспроизводимость
расчётов. [Порядок и область локальной проверки](LOCAL_VERIFICATION_RU.md).
Сеть, CI и LLM-провайдер для выполнения расчётов не требуются.

## Следующие интерфейсы

- [Python и Gymnasium на основном PWR-ядре](PWR_AGENT_QUICKSTART_RU.md).
- [Локальные LLM tools и экспорт training case](PWR_LLM_TOOLS_RU.md).
- [Параллельные кампании обучения](AI_09_10_12_TRAINING_RUNTIME_RU.md).

Прежний `reactor.ReactorSimulator` и `envs.gym_reactor` сохранены как учебная
ветка. Её GUI запускается из checkout командой `python apps/gui/app.py` после
установки `python -m pip install -e '.[gui]'`. Рисунки демонстраций сохраняются
в `artifacts/demos/`.
