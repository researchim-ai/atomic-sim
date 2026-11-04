# Подключаемый агент: явное состояние и JSONL

4 ноября 2026. План 1.60.0. Новый пакет `atomic_model_agents`, Python API
и CLI `atomic-model-agent`. Область — поддержанный EOS runtime и **доверенная
команда, которую запускает хозяин процесса**, а не загружаемый код из проекта.
Существующие workspace/study/results source и журналы не изменены.

## Реализация

1. Data-only descriptor schema1: agent_id/version, разрешённые observations
   и actions, initial_state, decision_timeout_s и rss_limit_bytes. Неизвестные
   поля/порты, дубликаты, пустые permissions, bool вместо числа и NaN отклоняются.
2. В запрос передаются только разрешённые текущие наблюдения с именами,
   единицами, valid и sample_time_s. Action-порты объявляются в W. Ответ обязан
   заполнить ровно разрешённые порты; остальные порты получают явно нулевую
   мощность. Общий action повторно проверяется публичным EOS admission.
3. Версионированные request/response, decision index, request_id и SHA-256
   observation привязывают решение к конкретному шагу и явному состоянию.
   Устаревший ID/хеш, лишний порт, неверный тип или неограниченная мощность
   отклоняются до изменения симулятора.
4. **Новый процесс на каждое решение**. State приходит в запросе, возвращается
   в ответе и принимается только вместе с успешным native step. Это устраняет
   зависимость протокола от скрытой памяти долгоживущего процесса. Внешние
   сервисы/файлы агента могут иметь своё состояние и не объявляются обратимыми.
5. Linux transport без shell: argv задаётся только host API/CLI, не descriptor.
   Бюджет времени включает запуск, ввод/вывод и завершение процесса. Stdin,
   stdout и stderr обслуживаются неблокирующим selector, включая случай,
   когда агент перестал читать stdin или закрыл все pipes и остался жить.
6. Лимиты: state 32 KiB, request 256 KiB, stdout 64 KiB, stderr 8 KiB,
   argv до 64 элементов/4096 символов, решение 0.05–60 s. Наблюдаемый RSS
   главного процесса — 16 MiB–16 GiB. Опрос примерно 10 ms: возможен overshoot;
   это не cgroup quota и не sandbox для недоверенного кода/его subprocesses.
7. Timeout, cancellation и output/RSS limit завершают собственную process group.
   Parent-death signal устанавливается перед exec основного агента. Это не
   гарантия очистки произвольных отделившихся дочерних сервисов при аварии host.
8. Сохраняются хеш argv, исполняемого файла и явно объявленных identity files;
   скрипт во втором argv-элементе распознаётся автоматически. Изменившийся
   entry-файл после ответа обнаруживается **до** native step. Сам argv и
   содержимое stderr в журнал не записываются: сохраняются ограниченные
   счётчики/хеши захваченных байтов. Полный dependency lock внешнего агента
   и аутентичность автора не заявлены.
9. Immutable checkpoint journal и atomic HEAD: ответ агента, принятый шаг
   симулятора и новое agent_state фиксируются вместе. Ошибка процесса,
   протокола или native admission сохраняет прежние episode/state/rows;
   отдельная attempt-запись объясняет отказ. Typed telemetry хранит wall/RSS,
   exit code, размеры и хеши захваченного вывода. Это наблюдения, не независимая
   аттестация стоимости или поведение внешнего сервиса.
10. Per-call decision budget и общий wall deadline, остановка между шагами,
    отмена во время решения без ожидания writer lock, CAS resume после
    failure/cancel. После обычной pause можно продолжить run. После аварии
    host оставшийся writer lock требует проверки владельца вручную; автоматическое
    удаление неизвестного lock не реализовано.
11. До 512 ревизий/256 попыток. Load проверяет source, hash-chain, постоянство
    scenario/descriptor/launch, монотонный accepted prefix, state и соответствие
    attempt→request. Каждая принятая команда заново интегрируется; сверяются
    все model rows и checkpoints исторических префиксов. Replay **не запускает
    агент**, работает после удаления его исполняемого скрипта и не требует ключей.
12. Data-only export с manifest, относительными ссылками и exact replay.
    32 MiB на JSON, до 128 MiB объявленных файлов. Существующий output не
    перезаписывается. Это собственный agent-job формат, а не незаметное
    расширение старого workspace policy schema.

## Контракт решения

Request содержит schema_version, request_id, decision, observation_sha256,
observation, allowed_actions и state. Response содержит schema_version,
request_id, decision, observation_sha256, heat_power_w и state.
Ровно один JSONL-ответ, затем успешный exit=0. Строки логов в stdout,
несколько ответов, duplicate JSON keys, NaN/Infinity и поздние ответы запрещены.
Необязательные логи направляются в ограниченный stderr.

Разрешения контролируют **протокольные данные**, а не доступ процесса к ОС.
Доверенный агент работает с правами пользователя и может иметь файловые или
сетевые побочные эффекты. Откат симулятора не откатывает эти эффекты; state и
принятые ответы входят в экспортируемый артефакт. Произвольные агентские
решения могут быть недетерминированными: native replay воспроизводит записанные
действия, а не обещает повторить поведение внешней модели.

## Воспроизведение

```bash
python -m examples.eos_external_agent_workflow --output artifacts/my-external-agents
atomic-model-agent inspect artifacts/my-external-agents/hold
atomic-model-agent replay artifacts/my-external-agents/hold-export
```

В примере две полностью локальные реализации: hold и clipped temperature-P.
Результаты совпадают с существующими встроенными baseline; pause/resume
сохраняет native episode и явный счётчик решений. Внешние сервисы не вызываются.

```bash
atomic-model-agent create artifacts/my-agent-job \
  --scenario artifacts/my-external-agents/scenario.json \
  --descriptor artifacts/my-external-agents/hold-descriptor.json \
  --agent-command python -m atomic_model_agents.baseline
atomic-model-agent run artifacts/my-agent-job --max-decisions 1 \
  --agent-command python -m atomic_model_agents.baseline
```

`--agent-command` должен идти последним: оставшиеся аргументы передаются агенту.
Для своего скрипта используйте host argv `python path/to/agent.py`; дополнительные
файлы идентичности задаются через `--identity-file` перед `--agent-command`.
Resume требует checkpoint_sha256 из inspect и той же launch identity:

```bash
atomic-model-agent resume artifacts/my-agent-job --expected-sha256 SHA_FROM_INSPECT \
  --agent-command python -m atomic_model_agents.baseline
atomic-model-agent run artifacts/my-agent-job \
  --agent-command python -m atomic_model_agents.baseline
```

## Приёмка и границы

AI-11.05 принимается в этом headless EOS JSONL scope. Generic topology/model
registry, подключение внешних агентов к старым study/reference/results schemas,
GUI для их запуска, произвольные persistent Python objects, streaming callbacks,
OS sandbox и dependency lock остаются отдельной работой. Descriptor не выдаётся
за готовую интеграцию всех типов агентов во все форматы проекта.

Доказательства: `validation/ai11_external_agents_delivery_v1.json` и
`validation/ai11_external_agents_portable_replay_v1.json`. Новые проверки
охватывают permissions, свежесть, explicit-state resume, process lifecycle,
stdout/stderr/RSS/deadline, живую отмену, изменение entry-файла, typed telemetry,
порчу промежуточного состояния и agent-free native replay. Физические уравнения,
численные бюджеты и флаги validated/plant_referenced не изменены.

Общий deadline проверяется между шагами; его остаток ограничивает ожидание
агента. Native-интегратор и фиксация принятого шага не прерываются асинхронно,
поэтому общий wall может быть превышен во время интегрирования/записи. RSS budget
относится к главному процессу агента, а не к host-моделеру и всей группе процессов.
