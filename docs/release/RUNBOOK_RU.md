# Установка, контрольный маршрут и recovery

AI-12.08. Область этого runbook — Linux EOS open-agent profile. Другие engines
сохраняют собственные паспорта и evidence. OS/dependency matrix и production
sandbox для недоверенных агентов не добавляются инструкцией.

## Установка и контроль

Из checkout с установленными зависимостями:

```bash
python -m build
python -m atomic_model_release.install_check \
  --wheel dist/atomic_sim-2.0.0-py3-none-any.whl \
  --output artifacts/installed-release
```

Проверка создаёт fresh venv и рабочий каталог **вне checkout**, удаляет PYTHONPATH,
устанавливает доверенный wheel offline с --no-deps, проверяет module location,
packaged golden/typing resources и все три use cases. Scientific dependencies
наследуются из system environment: это не hermetic dependency installation и
не доказательство min-version/independent-machine совместимости.

Для обычной установки wheel используйте `python -m pip install PATH.whl`.
Сеть и разрешённый источник dependencies определяет пользователь; не используйте
--no-deps, если зависимости ещё не подготовлены и не проверены.

```bash
atomic-model-release doctor
atomic-model-release use-cases
atomic-model-release golden
atomic-model-release control-demo --output artifacts/control-case
atomic-model-release diagnose artifacts/control-case/job
atomic-model-release diagnostic-demo --output artifacts/negative-case
atomic-model-release data-demo --output artifacts/learning-case
atomic-model-release dataset-summary artifacts/learning-case/benchmark
```

Outputs должны быть новыми. control-demo: три принятых шага, native replay без
агента, data migration и два golden controls. diagnostic-demo намеренно получает
agent_exit и проверяет отсутствие принятого шага/изменения модели; demo_passed
означает успешную проверку ожидаемого отказа, а не успешное управление.
Data-demo выполняет 16 коротких synthetic episodes и выгружает 48 transitions.

## Если что-то не работает

- Missing numpy/scipy/gymnasium: `doctor` не импортирует numerical модули и
  показывает наблюдённые distribution versions. Установите зависимости выбранного
  окружения; значения doctor не являются dependency lock.
- Unknown schema/port/unit: исправьте вход по объявленной schema. Автоматического
  угадывания версии и clipping неверного действия нет.
- Source mismatch: используйте frozen runtime исходного snapshot. Data-only
  migrate-project начинает новое исследование, не маскирует смену численной модели.
- Agent timeout/exit/response-contract: diagnose показывает observed reason,
  last attempt и длину принятого префикса. Это evidence triage, не экспертная
  причинная диагностика. Запуск агента при diagnose не повторяется.
- Падение controller: `atomic-model-release doctor --folder PATH/job` смотрит
  writer lock read-only. Он не удаляет lock и не убивает PID. Один лишь факт
  PID present/absent не исключает reuse и не доказывает владение. Сначала подтвердите
  завершение **своего** writer по OS/process/session журналам. Не удаляйте неизвестный
  активный lock. После подтверждённого ручного recovery используйте native inspect,
  затем штатный CAS resume для подходящего job schema.
- OpenMC/MPI prerequisite отсутствует: external lane завершается ошибкой,
  а не зелёным skip. Установка/лицензия cross sections — отдельная provisioning-задача.
- Hardware threshold не пройден: это отдельный hardware результат; не ослабляйте
  physical tolerances и не переносите legacy throughput на основной EOS.

## Release notes 1.62.0

Добавлены release CLI, frozen 0/500 W projects, явная config migration, read-only
recovery/diagnosis, published use cases, публичные input stubs и разделённые CI lanes.
Существующие source-bound agent/study/results formats и их смысл сохранены.
Model card остаётся synthetic/implemented, validated=false, plant_referenced=false.
Remote CI и весь long/hardware lane не объявляются прошедшими на основании YAML.

Для `install_check` подготовьте dev dependencies, включая mypy: он проверяет
не только наличие .pyi/py.typed в wheel, но и реальный installed consumer.
Неверный пример должен завершиться exit=1 с arg-type; в receipt это
expected_exit=1/passed=true, а не скрытая ошибка. Обычное выполнение runtime CLI
не требует mypy.
