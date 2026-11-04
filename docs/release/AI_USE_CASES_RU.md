# Публичный intended use для AI/RL/LLM

AI-13.01. Машиночитаемый registry: `atomic-model-release use-cases`.
Поддержанный здесь model_id — eos-thermal-assembly-v1: два finite-inventory
Euler circuit с caloric walls, synthetic stiffened EOS, first-order hydrostatic
Rusanov, ideal cached temperatures. Это исследовательская однофазная модель;
не water calibration, не validated plant и не полный динамический HEM/ВВЭР.

## Управление

`control-demo` подключает локальный JSONL hold agent без ключей/внешнего сервиса.
Агент получает только wall.w0 temperature в K и explicit state, возвращает
wall.w0 power в W; остальные action-порты явно нулевые. Host имеет полный native
checkpoint для аудита, но он не подменяет разрешённый observation агента.
Выходы: принятые действия/состояния, ledgers, replay и model card. Метрики:
полнота horizon, число решений, exact native replay и frozen semantic controls.

## Диагностика

`diagnostic-demo` — открытый воспроизводимый отрицательный контроль: реализация
локального агента отвергает намеренно неверную конфигурацию. `diagnose JOB`
читает source-bound checkpoint, проверяет native prefix и возвращает status,
observed reason, category, last attempt и accepted_time. Он не вызывает агента,
не меняет HEAD и не присваивает экспертную причинную аннотацию.

## Данные для обучения

`data-demo`/`dataset-summary` используют принятый EOS arena protocol: physical
train/dev/test, train-only выбор параметров, одинаковые permissions/budgets.
В 48 переходах сохраняются только разрешённые inputs, действия W, state/next_state,
provenance и корректные terminated/truncated. Reward=null: функция вознаграждения
не выдумывается. Labels — synthetic local policy response, не expert/causal truth.
Метрики: температурные MAE/RMSE/интегралы с units, sampled physical constraints,
completeness и observed repeat/cost. Token usage/cost=null; настоящая LLM не запускалась.
Публичный test нельзя повторно использовать для подбора той же версии.

## Критерии воспроизводимости

Все три маршрута реально выполняются из installed wheel вне checkout. Не нужно
наличие внешних клиентов, платного API или оплаты. Факт успешного software workflow
не повышает автоматически физическую fidelity. Общий model registry, реальные
экспериментальные labels и полная factory для всех engines остаются отдельными задачами.

Граница исполнения отражена в metadata: `data-demo` реально запускает локальных
агентов (`agent_executed=true`), а последующий `dataset-summary` только аудирует
сохранённый результат (`agent_executed=false`). Native audit может заново
интегрировать команды, но не повторяет поведение агента.
