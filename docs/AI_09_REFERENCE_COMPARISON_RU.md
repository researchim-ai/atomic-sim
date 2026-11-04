# Эталонные ряды, независимые метрики и воспроизводимый отчёт

`atomic_model_reference` 1.0.0 — импорт и сравнение данных для объявленного
`eos-thermal-assembly-v1`. Инструмент сопоставляет выходные температуры портов,
глобальный ledger и локальные mass/momentum/energy residual. Он не калибрует
физику и не повышает статус модели по одному успешному сравнению.

```bash
python -m examples.eos_reference_workflow --output artifacts/reference_demo
atomic-model-reference inspect artifacts/reference_demo/energy-data
atomic-model-reference catalog artifacts/reference_demo/temperature-data artifacts/reference_demo/energy-data
atomic-model-reference replay artifacts/reference_demo/energy-report --output artifacts/reference_replayed
```

Эквивалент console command: `python -m atomic_model_reference`. Импорт:

```bash
atomic-model-reference import examples/eos_reference/energy_reference.json --output artifacts/energy_reference
atomic-model-reference compare artifacts/reference_demo/job artifacts/energy_reference --output artifacts/energy_comparison
```

Выходные каталоги должны быть новыми. Compare возвращает exit 2 при непройденных
критериях и сохраняет полный неуспешный отчёт. Replay возвращает успешное
воспроизведение, даже если воспроизведённый **результат сравнения** отрицательный:
эти две величины явно разделены.

## Вход и единицы

Schema1 содержит reference_id, physical_case_id, role, kind, provenance,
problem, csv, channels и criteria. Полный образец —
`examples/eos_reference/reference.json`; второй — `energy_reference.json`.

- `role`: calibration / validation / test — назначение данных, не статус модели.
- `kind`: experiment / intercode / numerical_reference. Категории не смешиваются
  в отчёте. Источник, лицензия, revision, title и notes обязательны и остаются
  **заявлением поставщика**; программная проверка не аутентифицирует эксперимент
  и не проверяет юридическую действительность лицензии/URI. URL не загружается.
- `problem`: точные model ID, config и поддержанная policy. Job должен совпасть
  с полной нормализованной постановкой. Другие модели и произвольный user code
  не загружаются.
- `csv`: переносимый относительный path, SHA-256 исходных байтов, time column
  и time unit (`s`/`ms`). Нужны относительные неотрицательные времена; скрытого
  time shift нет. Path traversal и symlink внутри reference root отклоняются.
- `channels`: явное соответствие column → model signal. Допустимы `K`, `degC`,
  `J`, `kJ`, `J/K`, `kJ/K`, `kg`, `kg m/s` по размерности сигнала. Температура
  переводится в K, энергия — J. Для uncertainty Celsius offset **не применяется**:
  это разность температур. Unknown uncertainty задаётся `null`, не нулём.

Максимум CSV 8 MiB, 20 000 rows и 32 channels; normalized JSON — 16 MiB.
Строго проверяются число/уникальность headers, полное соответствие колонок,
непустые числовые значения, NaN/Inf, возрастающее время, положительная absolute
температура и неотрицательная uncertainty. Пропуски/дубли не удаляются молча.
Некорректный вход отклоняется до создания bundle.

## Критерии и метрики

Во входе обязательны window_s, max_reference_gap_s, max_model_gap_s и ограничения
MAE/RMSE/peak absolute error/integral absolute error для каждого канала.
События объявляются отдельно: канал, порог в **канонической единице**, направление
up/down и максимальная ошибка времени. Декларация критериев сохраняется; сам
инструмент не доказывает, что пользователь зафиксировал их до просмотра данных.

Model/reference интерполируются линейно на объединении узлов внутри явного
окна; экстраполяция запрещена. Gap проверяется на исходном интервале, участвующем
в интерполяции, а не уменьшается искусственным crop. Незавершённый model episode,
недостаточное покрытие или model gap дают отрицательный результат без метрик
успешного префикса.

Вычисляются аналитические интегралы кусочно-линейного signed/absolute/squared
error, time-weighted bias/MAE/RMSE, maximum absolute error, model/reference
интегралы и пики. Интеграл квадрата использует h·(e0²+e0·e1+e1²)/3; при смене
знака absolute error сегмент разделяется по корню. Это интегралы объявленной
интерполяции в floating-point, не доказательство точности непрерывного solver.

Peak — первый глобальный максимум в окне; несколько сходных пиков не сопоставляются
по физическим признакам. Event — первое directional crossing со строгой стороны
к включённому порогу. Начальное равенство/плато не создаёт нового crossing.
Отсутствующий reference/model event записывается явно, time error остаётся null,
критерий не считается выполненным. Выходные samples не заменяют native root finding.

Fraction time within declared ±uncertainty вычисляется по пересечениям линейных
сегментов; это не sample fraction, не доверительный интервал и не сертификация
покрытия. Неопределённости параметров/замыканий/сетки/интегратора ещё не разделяются.
У каждого metric в отчёте записана единица, включая произведение на секунду.

## Разделение случаев и данных

Catalog проверяет cross-role совпадения physical_case_id, нормализованных
физических условий, исходного CSV и точных канонических кривых. Physical identity
включает preset geometry/material specification и policy; max_step, crop/horizon
и seed-label не превращают случай в новый. Для постоянного hold period решений
также не создаёт новый physical case; у feedback period является частью управления.

Изменение ID, numerical step, horizon, unit или uncertainty не должно превратить
ту же кривую в held-out evidence. Guard консервативен: одинаковые constant curves
в разных ролях отклоняются тоже. Он не обнаруживает любую приблизительную копию,
скрытое пересэмплирование или ложное описание реального эксперимента. Проверенные
источники и разделение реальных физических случаев остаются самостоятельной работой.

## Bundle, отчёт и replay

Import сохраняет исходный CSV побайтно, `source_input.json` — исходную
структурированную декларацию, `input.json` — переносимую копию с `data.csv`,
и нормализованный `reference.json`. Manifest содержит byte hashes, digest
канонической исходной декларации и importer source. При загрузке hashes,
согласованность original/portable metadata и повторный parse raw CSV проверяются.

Отчёт содержит raw CSV, reference declaration/snapshot, весь accepted job snapshot,
неизменённые criteria, результаты/отказы, units, source и hashes. Replay в
временном каталоге заново проверяет native checkpoint и вычисляет метрики;
JSON результата должен совпасть точно. Даже подмена metric с переписанным
внешним file hash обнаруживается recomputation. Другой comparator source требует
явного решения о новом сравнении; автоматической миграции report meaning нет.
SHA-256 подтверждает согласованность, не подпись автора или истинность измерений.

## Контрольные данные этого выпуска

Использованы только repository-authored synthetic данные с Apache-2.0 metadata.
Настоящие экспериментальные данные не использовались и не выдаются за собранные.
Положительный контроль — energy residual 0 J как conservation identity; это
numerical verification. Отрицательный — постоянные 300 K для wall.w0: в uniform
preset primary/secondary стартуют при 350/280 K, стенка при 300 K и нагревается
от теплообмена даже при нулевых командах. Такой thermal reference должен не пройти.

Workflow проверяет оба результата, exact replay pass/fail и отказ при переносе
того же physical case в другую роль. Его численный успех не доказывает точность
реальной установки. Все bundle/report flags остаются `validated=false`,
`plant_referenced=false`, promotion — none. AI-09.02/03/05/06/08 не закрываются
наличием CSV, metadata или количества тестов.
