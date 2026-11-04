# Открытый benchmark EOS-агентов и данные переходов

4 ноября 2026. План 1.61.0. Новый пакет `atomic_model_arena`, CLI
`atomic-model-arena`. Протокол фиксирует поддержанную модель, физические
разбиения, численные настройки, permissions, бюджеты и правило выбора политики.
Это **синтетический software benchmark**, а не сравнение точности с реальным
реактором, доказательство безопасности или промышленная физическая валидация.

## Реализованный большой блок

1. Data-only protocol schema1: model/benchmark ID, 3–12 случаев, baseline и
   1–7 кандидатов, 2–4 повтора, numerical config каждого случая, общие ограничения
   и бюджеты. Код/argv агента не загружается из protocol; host задаёт команды
   отдельно. Стандартный запуск использует полностью локальный JSONL baseline.
2. Обязательны train/dev/test. Hash канонического физического preset запрещает
   помещать один физический случай в разные split под другими ID/seed/objective
   или длительностью. Guard относится к принятым пресетам, а не доказывает
   семантическую независимость произвольных будущих моделей.
3. Все агенты имеют одинаковые разрешённые observations/actions, decision timeout
   и agent RSS budget. На каждом case они получают одинаковую точность,
   horizon/action interval, objective, seed, предел решений и wall budget.
4. Нормализованный protocol сохраняется до расчётов. Private worker — новый
   интерпретатор с affinity одного разрешённого CPU; настройки BLAS threads=1
   устанавливаются до импорта numerical packages. Эпизоды выполняются последовательно,
   порядок агентов циклически меняется между повторами. Фиксируются Python/ОС,
   CPU model/count, affinity, thread environment и launch identities.
5. Сначала выполняется полная train-матрица baseline+кандидаты. Кандидат допускается
   только при завершении **всех** требуемых эпизодов и соблюдении ограничений.
   Выбор — минимум среднего train MAE с равным весом case/repeat; точные ties
   разбираются лексикографически. Baseline не участвует в выборе кандидата.
6. Selection artifact с protocol SHA, hash выбранного descriptor и полным набором
   train task IDs записывается **до dev/test**. Затем на dev и test исполняются
   только baseline и уже выбранный кандидат. Параметры не корректируются по dev/test.
   Если подходящих кандидатов нет, статус `no_valid_train_candidate`, а holdout
   эпизоды и их результаты не выдумываются.
7. Точные piecewise-linear MAE/RMSE/bias/интегралы температурной ошибки относительно
   объявленного objective, с явными единицами каждой метрики. Отдельный аналитический
   тест пересечения нуля отличает точный интеграл |error| от трапеции модулей узлов.
8. По принятым decision samples измеряются max energy/entropy/mass/momentum/circuit
   residual и абсолютная мощность. Native admission продолжает действовать внутри
   интегратора. Неоконченный эпизод или одно нарушение исключает **весь agent/phase**
   из quality ranking: пропавшие неудобные случаи не удаляются из знаменателя.
9. На каждом case/repeat сохраняются actual process/native statuses, причины останова,
   целевой допуск, accepted decisions и hashes episode/траектории. Проверяется
   совпадение повторов без wall timings. Seeds здесь metadata: EOS и demo-политики
   детерминированы; повторы не объявляются независимыми статистическими опытами.
10. Измеряются create/run/publish wall, включая запуск агента, суммарный agent wall,
    peak agent RSS, captured output bytes. Последующий audit исключён из episode wall.
    CPU affinity и thread environment контролируются, но host load/частота CPU,
    внешние сервисы и отделившиеся subprocesses доверенного агента не изолируются.
    Временной рейтинг/глобальное преимущество по скорости не публикуются.
    token_usage/token_cost остаются **null**, поскольку реальные LLM не вызывались.
11. Dataset `transitions.jsonl`: только разрешённые observation/next_observation,
    action в W, explicit state/next_state, case/agent/split/repeat/decision и provenance.
    Reward отсутствует (`null`): benchmark не навязывает RL reward и не подменяет
    его скрытой суммой физических величин. Labels — synthetic policy responses,
    не экспертная диагностика и не causal annotation.
12. Time-limit сохраняет native Gym-семантику: `terminated=false`, последний
    переход `truncated=true`; `episode_completed` отдельно отмечает пройденный
    horizon, `completion_reason` отличает time_limit от отказа. При отсутствии
    принятого шага нет фиктивного перехода. Не сохраняются привилегированные
    model observations вместо разрешённых inputs агента.
13. Каждый episode экспортируется в проверяемом agent-job формате. Offline verify
    повторно интегрирует принятые команды и проверяет historical prefixes, заново
    считает метрики, train selection, точный schedule и итоговый report. Transitions
    генерируются заново и сравниваются побайтово. Агент при аудите не исполняется.
14. Общий worker wall/RSS/output budget ограничивает расчёт вместе с его аудитом.
    При превышении сохраняется `controller_failure.json` и неполный diagnostic
    каталог; он не выдаётся за валидный benchmark. Автоматического resume всей
    arena-кампании нет: новый run требует нового output. Отдельные agent jobs
    имеют прежние pause/resume/CAS возможности.
15. SHA-256 manifest, source binding, строгие schemas/paths, отсутствие symlinks
    и посторонних файлов; до 4096 объявленных файлов, 32 MiB на файл и 512 MiB
    суммарно. Code/dependency trust и physical validation не выводятся из хешей.

## Воспроизведение

```bash
python -m examples.eos_arena_workflow --output artifacts/my-arena
atomic-model-arena validate artifacts/my-arena/input.json
atomic-model-arena verify artifacts/my-arena/benchmark
```

Пример: uniform/train, mixed_ports/dev, gravity/test; baseline hold=0 и три
кандидата feedback с gain 0/500/1000 W/K. На train выполняются 8 эпизодов,
после выбора на dev/test — ещё по 4: **16 эпизодов, 48 принятых переходов**.
Каждый запуск использует короткий поддержанный horizon 0.03 s; длительные
траектории и реальная установка этим набором не квалифицируются.

```bash
atomic-model-arena run artifacts/my-arena/input.json --output artifacts/another-arena
```

`--cpu N` выбирает один CPU из исходной affinity. `--commands host-commands.json`
задаёт host map agent_id → argv; это явное разрешение запустить доверенные команды,
не инструкция выполнить код из открываемого protocol. Они должны соблюдать
заявленный процессный контракт. Параметры protocol замораживаются до исследования.

## Как читать результат

`report.json` содержит raw episodes, completion/constraint outcomes, per-phase
quality ranking, повторяемость и наблюдаемые затраты. Общего ранга train+dev+test
нет. В benchmark нет скрытой оптимизации на test и агрегирования K с J.
`hardware.json`, `launch.json`, `selection.json` и episode artifacts дают контекст
для проверки. Offline replay проверяет численную воспроизводимость и integrity,
а не подлинность измеренного wall или независимую hardware attestation.

Cases и test transitions открыты. Guard проверяет алгоритмический маршрут,
но не делает опубликованный test секретным: для честного сравнения нельзя
повторно подбирать параметры на его опубликованных результатах той же версии.
Значение native_repeat_exact — измеренный факт для конкретного case/agent,
а не гарантия детерминированности любой внешней LLM.

AI-13.03 и AI-13.04 принимаются **только в этом ограниченном EOS baseline scope**.
AI-13.06 не закрывается: новые реальные преимущества относительно сторонних
симуляторов/агентов не измерялись. Общая промышленная физика, uncertainty/calibration,
полный model registry и independent-machine/OS-matrix release acceptance остаются
отдельными критериями. Все validated/plant_referenced — false.

Доказательства и окончательные test counts:
`validation/ai13_eos_arena_delivery_v1.json` и
`validation/ai13_eos_arena_portable_replay_v1.json`.
