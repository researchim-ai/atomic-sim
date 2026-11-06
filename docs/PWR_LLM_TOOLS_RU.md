# Управляемый PWR через локальные LLM tools

Протокол `atomic_agent_tools` версии 1.2 добавляет три инструмента для
`pwr-reduced-flow-controlled`. Существующие `create_episode`/`step_episode`
и EOS-инструменты сохраняют свои модели и семантику.

```bash
python -m examples.llm_controlled_episode
python -m pytest tests/test_controlled_agent_tools.py -q
python -m atomic_agent_tools
```

Пример и pytest запускаются из checkout с установленными зависимостями;
`python -m atomic_agent_tools` доступен и из установленного wheel.

Первый пример запускает настоящий дочерний JSONL-процесс, выполняет действие,
сохраняет checkpoint, перематывает и повторяет его с точным сравнением состояния.
Затем завершает эпизод и получает case для существующего training runtime.
У каждого ответа в примере есть тайм-аут 60 s; это ограничение клиента,
а не прерывание решателя сервера. Сетевые сервисы, API-ключи и CI не нужны.

## Вызовы

Каждая строка stdin — запрос; stdout содержит только ответы JSONL.
Диагностика идёт в stderr. Это локальный JSONL, не MCP и не JSON-RPC.

```json
{"request_id":"r1","tool":"create_controlled_episode","arguments":{"session_id":"pwr","config":{"duration_s":0.1,"max_steps":100,"tier":"training","seed":17}}}
```

Ответ возвращает `session_id`, `revision`, `observation_id`, `status`,
`time_s`, `observation` и `action_channels`. В `observation` перечислены имена,
единицы и значения полного состояния. `observation_kind="full_model_state"`
явно отличает его от прежнего сенсорного интерфейса. `valid=true` здесь означает
значение состояния модели; измерительной неопределённости или датчика это не создаёт.
`thermal_power_w` — тепловая мощность в ваттах. `model.validated=false` и
`model.plant_referenced=false` сохраняются.

Следующий вызов использует revision и observation_id из последнего ответа:

```json
{"request_id":"r2","tool":"step_controlled_episode","arguments":{"session_id":"pwr","expected_revision":0,"observation_id":"ЗАМЕНИТЬ_ТОКЕНОМ_ИЗ_ОТВЕТА","action":{"throttle":0.95}}}
```

`action` — частичный объект стандартных команд SDK. Пустой объект сохраняет
текущие настройки. Булевы `scram`, `motor_off`, `motor_on` принимаются как
настоящие JSON boolean; числовые 0/1 вместо boolean отклоняются.
Числовые границы и единицы возвращаются в `action_channels`; границы приёма
команд не гарантируют допустимость любой физической комбинации.
Одновременные `motor_off=true` и `motor_on=true` запрещены.
Общие параметры Episode описаны в [Python/Gymnasium quickstart](PWR_AGENT_QUICKSTART_RU.md).

`last_transition.reward=null`: в LLM-интерфейсе нет неявно выбранной функции
награды. SCRAM даёт `terminated=true`; бюджет решений — `truncated=true`.
После завершения действие отклоняется. Ошибка решателя даёт `numerical_rejected`
без фиксации пробного состояния; неожиданные ошибки возвращаются отдельно.

## Checkpoint и экспорт кампании

Общие `inspect_episode`, `save_checkpoint`, `restore_checkpoint`,
`cancel_episode`, `close_episode`, `forget_checkpoint` работают с новым PWR.
Изменяющие операции и экспорт case требуют актуальные revision/observation_id.
Restore перематывает физику, но увеличивает revision инструмента. `core_revision`
показывает число шагов восстановленного физического журнала; оно может отличаться
от revision после restore/cancel. Старые токены не становятся действительными снова.
Сохранённый checkpoint другой модели нельзя применить к управляемому PWR.

После хотя бы одного принятого шага `export_controlled_case` возвращает
`{"model": ..., "case": ...}`. Аргументы: guard, `case_id`, `family` и
необязательный `split` (`validation` по умолчанию). Инструмент не пишет файлы.
Объект `case` можно передать `atomic_model_training.episode.run_episode` или
включить в JSON-массив кампании `atomic-training`. Экспорт доступен после
завершения эпизода и не изменяет его. Именование split/family не превращает
повтор одной траектории в независимую физическую проверку.

Полный перенос между процессами остаётся функцией доверенного хоста:
`AgentToolServer.export_session/import_session` или CLI `--checkpoint-out` /
`--resume` с `--session-id pwr`. Агент получает непрозрачные checkpoint handles.
Checkpoint привязан к реализации и runtime; после изменения исходников старые
checkpoint не объявляются совместимыми автоматически.

Лимиты числа сессий, запросов, model transactions, checkpoint bytes и памяти
handles общие с прежними backend. Для нового PWR максимум 256 решений в эпизоде.
Перед действием сервер восстанавливает изолированный пробный эпизод повтором
журнала: это обеспечивает отказ без фиксации и увеличивает стоимость длинных
эпизодов. Этот транспорт не заявляет ускорение training runtime.

## Локальная проверка поставки

`python -m scripts.verify_local_library` включает тесты нового и прежнего
tool server. После сборки wheel и установки вне checkout отдельная фаза
`installed-agent-interfaces` проверяет Python, Gymnasium, host restart и
восстановление через настоящий новый процесс JSONL. Результат сохраняется
в `agent.json`, а provenance сравнивается с исходной реализацией.
