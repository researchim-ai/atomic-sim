# Единый интерфейс агента для исследовательского PWR

`atomic_pwr_agent.Episode` работает с `pwr-reduced-flow-controlled`.
Это редуцированная исследовательская модель, не экспериментально аттестованный
цифровой двойник. Прежний `envs.gym_reactor` остаётся учебной средой.

Из корня установленного для разработки проекта:

```bash
python -m examples.training.agent_episode artifacts/my-agent-episode.json
python -m pytest tests/test_pwr_agent.py -q
```

Пример выполняет три решения, восстанавливается после первого из checkpoint,
экспортирует траекторию и повторяет все команды существующим training runtime.
Файл назначения должен быть новым. Это демонстрация воспроизводимости,
а не обучение политики или независимая физическая валидация.

## Python и локальные LLM-клиенты

```python
from atomic_pwr_agent import Episode

episode = Episode(duration_s=0.1, max_steps=100, tier="training", seed=17)
observation = episode.observe()
result = episode.step({"throttle": 0.95},
                      observation_id=observation["observation_id"])
saved = episode.checkpoint()
episode = Episode.restore(saved)
case = episode.training_case(case_id="load_01", family="load_change")
```

Наблюдение — JSON-совместимый объект версии 1: полный вектор `state`, его
`state_schema` с именами, индексами и единицами, время в секундах, тепловая
мощность в ваттах, revision и observation_id. Это полное состояние модели,
не имитация измерительных каналов; sensor noise и скрытые состояния здесь не заявлены.
Клиент LLM может передавать проверенный объект команд этому же интерфейсу;
существующий JSONL tool server автоматически на новую модель не переключается.
Для явного выбора нового backend доступны [инструменты протокола 1.2](PWR_LLM_TOOLS_RU.md).

Команды проверяются существующим `sdk.api.validate_commands`:
`throttle` [0,1], `rod_command` [-1,1] 1/s,
`feed_bias_kg_s` ±100000 kg/s, `boron_ppm` [0,20000] ppm,
`sink_temperature_k` [273.15,373.15] K, булевы `scram`, `motor_off`, `motor_on`.
Это границы приёма команд, не гарантия физической допустимости любой комбинации.
Пропущенные настройки сохраняются. `scram` защёлкивается.

Каждый шаг требует актуальный observation_id; устаревший запрос отклоняется.
Длительность шага (0,5] s, бюджет 1–256 решений. SCRAM завершает эпизод
(`terminated`), достижение бюджета даёт `truncated`. После завершения нужен
новый эпизод. Ошибки интегратора и области модели передаются вызывающему коду,
не маскируются наградой. Checkpoint привязан к исходникам и runtime;
восстановление повторяет журнал с проверкой состояния.

`seed` — метаданные воспроизводимости; начальное физическое состояние
детерминировано. Экспортированный `training_case` принимается существующими
кампаниями. Обозначение split не делает повтор одной траектории независимым
экспериментом: для кампаний сохраняется проверка утечки между семействами.

## Gymnasium

```python
from atomic_pwr_agent.gymnasium import ControlledPWREnv

env = ControlledPWREnv(max_steps=100, target_power_w=3e9)
obs, info = env.reset(seed=17)
obs, reward, terminated, truncated, info = env.step(1)
env.close()
```

Действия 0/1/2 задают скорость стержней -0.001/0/+0.001 1/s по умолчанию.
Наблюдение — float64-вектор того же полного состояния; схема и единицы находятся
в `info["observation"]`. Награда — минус абсолютная ошибка тепловой мощности,
делённая на целевую мощность. Цель и скорость задаются конструктором.
Встроенной случайной физической инициализации и автоматического reset нет.
Для переноса checkpoint используйте Python Episode; Gym-обёртка не обещает
восстановление внутреннего состояния обучающего алгоритма.
