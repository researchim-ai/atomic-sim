# Измеряемая производительность EOS-ядра

`atomic_model_bench` 1.0.0 — отдельный Linux-стенд для настоящего
`eos-thermal-assembly-v1`. Он не меняет уравнения, EOS, CFL, max_step или
балансовые допуски. Не использует surrogate и учебную batch-ветку.

```bash
atomic-model-bench examples/eos_performance.json --output artifacts/eos_bench
# либо
python -m atomic_model_bench examples/eos_performance.json --output artifacts/eos_bench
```

Выходной каталог должен быть новым. JSON schema1 содержит фиксированные cases
(id, config, полная карта heat_power_w, seed), workers из 1/2/4/8 с обязательным
baseline1, repeats, cpu_count, worker_timeout_s и worker_rss_limit_mib.
Все случаи проверяются до создания каталога. Seed — записанный идентификатор;
фиксированные EOS-команды не используют RNG. Восемь случаев примера отличаются
пресетом и/или мощностью. Это короткий workload инфраструктуры, не статистическое
доказательство качества управления или достоверности модели установки.

## Общий бюджет и изоляция

Supervisor и все его существующие потоки ограничиваются одним фиксированным
набором разрешённых CPU. При выходе прежняя affinity потоков восстанавливается.
Worker наследует этот набор и проверяет его до импорта NumPy/SciPy. Переменные
OMP/OpenBLAS/MKL/NumExpr задаются равными 1 до запуска интерпретатора worker.
Одинаковое общее число CPU сохраняется для всех режимов 1/2/4/8 workers;
oversubscription остаётся измеряемым результатом, а не дополнительным CPU.

Каждый case запускается в отдельном доверенном subprocess. Deadline включает
запуск интерпретатора; превышение завершает процесс и не превращается в быстрый
успешный episode. RSS читается из Linux `/proc/PID/status`; при превышении
supervisor останавливает worker и продолжает следующие случаи. После штатного
выхода также проверяется kernel high-water RSS (`getrusage`, Linux KiB → bytes).
Это **наблюдаемый RSS budget**, не hard OS memory quota: между samples возможен
выброс, память дочерних процессов не учитывается. Worker здесь не создаёт
дочерних процессов и не исполняет пользовательский код. CPU-time quotas,
cgroups, недоверенные политики и не-Linux backend не реализованы.

На исключении все принадлежащие стенду живые процессы завершаются и join/wait
выполняется с ограничением времени. Промежуточные case/progress JSON сохраняются.
Возобновление performance-run не заявляется; persistent job/campaign recovery
находится отдельно в `atomic_model_workspace`.

## Что именно измеряется

У всех trials одинаковый список случаев и точность. Wall time trial включает
startup/import, расчёт, запись результата и проверку supervisor. Throughput:
принятые завершённые episodes / wall second и принятые simulated seconds /
wall second. p50/p95 — линейная интерполяция наблюдённых end-to-end задержек
случаев; восемь samples не оценивают надёжно генеральный p95.

RSS: per-worker kernel high-water и sampled peak суммы текущих RSS живых workers.
Эти величины различны. Последняя не включает память supervisor; CPU allocation
включает supervisor. Полные результаты/ошибки, exitcode, source fingerprint,
настройки потоков, affinity, Python/NumPy/SciPy и CPU/ОС сохраняются.

Отдельный profiled case не входит в вычисление speedup. Он записывает:

- время импорта, инициализации, интегрирования с admission, checkpoint export
  и выходного snapshot;
- user/system CPU time worker перед записью результата;
- полную таблицу cProfile с self/cumulative time и количеством вызовов;
- сумму **exclusive self time** по явным группам функций: caloric state/entropy,
  SSPRK transaction, native equations/admission, прочие вызовы.

Cumulative times перекрываются: их нельзя суммировать. Self time группы тоже
не является полным inclusive временем феномена. Низкоуровневые NumPy/C вызовы
могут попасть в `other`; это не означает отсутствия затрат EOS.
Этот EOS backend использует явный SSPRK3, Jacobian **не применяется**. Стенд
не приписывает ему стоимость Jacobian другого Radau-backend. Выделение SciPy
Jacobian/integrator предусмотрено только для реально наблюдаемых вызовов.

Supervisor восстанавливает каждый native episode и сверяет config, horizon,
историю команд, source и digest. Итоговый episode каждого case должен побитово
совпасть между повторами, worker counts и профилированным запуском (через
канонический JSON hash). Это проверка одинакового результата в данном окружении,
не обещание кроссплатформенного bitwise replay или физической валидации.

## Отчёты и интерпретация

`input.json`, `report.json`, `manifest.json`, per-trial и per-case JSON/логи
содержат единицы и SHA-256. Частичный trial сохраняет отказы. Если хоть один
case baseline или сравниваемого режима не завершён, speedup этого режима —
`null`; успешные короткие префиксы не подменяют полный workload.

Speedup — отношение медианного throughput режима к медианному throughput1.
Efficiency — speedup / workers; при общем фиксированном числе CPU и
oversubscription она закономерно отличается от scaling при добавлении CPU.
Порог 2x фиксирован до измерений. Его недостижение оставляет AI-10 частичным.
Нет утверждения о превосходстве над другими симуляторами. Ни другие машины,
ни более длинные горизонты, ни другой backend автоматически не покрываются.

Первый exploratory-run `eos_performance_v1` сохраняется исторически: CPU set
ограничивал только workers. Основной следующий run ограничивает также все
существующие потоки supervisor; эти результаты не объединяются в одну выборку.
Установка и повтор проверяются отдельно на wheel вне checkout с наследованными
научными зависимостями; hermetic dependency/OS matrix не заявляются.

## Основной локальный результат v2

Восемь случаев, два повтора, 64 полных непрофилированных эпизода плюс один
отдельный profiled case. Все native episode hashes совпали. На общем бюджете
двух CPU лучший медианный throughput оказался у четырёх worker: около
3.93 episode/s против 2.24 у одного, speedup **1.756x**. Восемь worker дали
около 3.57 episode/s. Цель 2x не достигнута. Это результат данной короткой
задачи и машины; worker count не выбирается универсально по этому примеру.
Машинный отчёт: `validation/ai10_eos_performance_summary_v1.json`; исходные
per-case файлы и cProfile: `artifacts/eos_performance_v2/`.
