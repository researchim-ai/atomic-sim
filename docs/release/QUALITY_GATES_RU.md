# Gates актуального API и regression

AI-12.04. Portable lane выполняет весь выбранный набор tests/, включая современные
EOS/agent/study/results/arena API. Lint перечисляет актуальные atomic_model_*
пакеты и atomic_model_contracts; он больше не ограничивается legacy reactor/sdk.
Прежние дополнительные numerical lint/black/run/audit проверки сохранены в long lane.

Статический контракт включает canonical `Config`, `Objective`, `Scenario`,
`AgentDescriptor` TypedDict и типизированные публичные функции agent jobs.
В wheel поставляются jobs.pyi и py.typed. Проверяемый consumer использует реальный
jobs.create/run, положительный и отрицательный mypy-примеры предотвращают пустой gate.

```bash
python -m mypy --strict --follow-imports=silent \
  atomic_model_contracts atomic_model_release/api_contract.py
```

Scope типов — канонические входы публичного API и consumer, а не доказательство
типизации всех динамических внутренних модулей. Пределы чисел, finite values,
permissions и структура checkpoint по-прежнему проверяются runtime schemas.
Полная canonical config обязательна для TypedDict; runtime поддержка прежних
JSON-входов не сужена. `initial_state` остаётся открытым JSON payload с runtime
admission, а не статически доказанным доменом физики.

Golden gate выполняет два замороженных случая (0 W и 500 W) из ранее принятой
numerical evidence, проверяет порты/единицы, horizon, актуальность датчиков,
температуры, local/global ledgers. Oracle не вычисляется заново при тесте.
Ресурсы защищены manifest SHA, заданным кодом; их изменение требует явного review.
Допуски 1e-9 K и 1e-6 для соответствующих ledger-величин зафиксированы в oracle,
не подбираются при failure. Это regression controls синтетического EOS.

```bash
atomic-model-release golden
```

Новые gates не меняют численные уравнения/бюджеты и не заменяют исторические
полные numerical qualifications более коротким тестом. В delivery разделены
full portable receipt, новые release tests, installed-wheel checks и external receipts.
