# Реализация оставшихся 17 задач: проверенная область и внешние условия

Актуализация 1.80.0: обязательный CI и replay на другой физической машине отменены
пользователем для software-поставки. AI-11.06 принят по локальной проверке;
[текущий порядок](LOCAL_VERIFICATION_RU.md). Ниже сохранено описание поставки1.78;
его прежние внешние условия относятся к истории, а не к текущему запуску библиотеки.

## Что реализовано

Новые пакеты `atomic_model_multiphysics` и `atomic_model_research` дополняют прежние
принятые модели; их исходники и контрольные суммы не изменены.

- **AI-08.01/02:** реальный OpenMC 0.16.1.dev0 с установленной ENDF/B-VIII.0;
  три независимых seed, retained geometry/materials/settings/tallies XML,
  summary/statepoint HDF5, MGXS mean/std, лицензионный манифест и SHA-256 каждого
  использованного файла нуклида. Явные группы fast→thermal, scatter[in,out] →
  scatter[to,from], перевод cm→m и inverse-velocity. Офлайн-ресурс — результат
  настоящего transport, а не синтетические коэффициенты. Геометрия — объявленная
  однородная отражающая исследовательская пластина, не активная зона ВВЭР.
- **AI-08.04:** консервативное отображение интегральной мощности и средних температур
  между несовпадающими осевыми сетками; радиальный FV-твэл, стационарный канал с
  постоянным cp и совместная итерация с диффузией. Невязки температур ≤1e-5 K,
  мощности ≤1e-6, теплового баланса ≤1e-9. Нет кипения, переменного cp, механики
  зазора, аксиальной проводимости и динамического запаса канала. Синтетическая
  теплоформа использует постоянную энергию на произведённый нейтрон; для evaluated
  конфигурации обязательно передать явный heat-response, молча подставлять nu-fission нельзя.
- **AI-08.05:** таблица пяти осей Tf/Tc/плотность/бор/состав, происхождение и
  семантика неопределённости; строгий отказ вне области. Связанная квалификация
  использует явно синтетическую аффинную таблицу. Реальный MGXS относится к одной
  статической точке; он не выдан за измеренную многомерную таблицу отклика станции.
- **AI-08.06:** замкнутые capture/decay цепи с положительным оператором,
  CE/CM predictor-corrector, CRAM16 и интегральный энергетический счётчик. Три
  независимых состава/жёсткости сопоставлены с dense expm; дополнительное нелинейное
  выгорание проверено по рациональному решению, порядок CE/CM около 2.
  Это квалификация численных методов с явно заданными коэффициентами, без
  evaluated fuel-depletion chain, yields library или сертификации ANS decay heat.
- **AI-08.07:** точное пересечение поглотителя с ячейками вместо привязки к узлу;
  sweep положения между узлами, три сетки, независимый плотный k и интеграл
  поглощения. Для manufactured синусоидального пика измерен порядок около 2.
  Это не обоснование трёхмерных пиков реального реактора; 2D/3D не заявлены.
- **AI-08.08:** spatial-agent task с узким паспортом steady operating-point,
  ограниченными действиями мощности, revision, транзакционным отказом и полным
  source-bound replay. Решение агента не названо временным шагом реактора.
- **AI-11.01:** проекты с model/scenario/data/agent config, неизменяемыми ревизиями,
  проверкой ancestry, относительными content-addressed данными, атомарным HEAD,
  защитой от stale edit, bounded ZIP export/import и replay реального controlled-PWR
  после перемещения каталога. Поддержана только объявленная default-конфигурация.
- **AI-12.01:** поддерживается один точно проверенный headless-профиль:
  Linux x86_64, CPython 3.12.9, NumPy 2.2.6, SciPy 1.15.2. Минимальная и выбранная
  версии в этом узком профиле совпадают. Другие версии/ОС не повышаются в supported
  от одного факта импорта. Constraints фиксируют наблюдавшиеся зависимости;
  это не герметичный wheelhouse и не hash-locked поставка всех native libraries.
- **AI-12.02/06:** wheel/sdist, холодная установка вне checkout, проверка пакетных
  JSON/YAML ресурсов и CLI, numerical replay, source/version/dependency/license
  inventory, SPDX SBOM, checksum/evidence archive. NOASSERTION оставляется там,
  где не проведён юридический анализ. Подпись издателя не выдумана: реального
  канала публикации/ключа в этой поставке нет.
- **AI-13.05/06:** техническая приёмка открытого исследовательского release candidate
  по опубликованным критериям и ограничениям. Сохранено измеренное преимущество
  5,277× только на прежнем коротком nominal reset-heavy controlled-core workload;
  это не ускорение новой spatial-связки и не универсальное превосходство.
  Коммит, push и публичная публикация пользователем этой поставкой не выполняются.

## Четыре условия, которые нельзя закрыть числом тестов

- **AI-09.02:** есть лицензированные NIST pipe, HIPER и B-274, исходники, единицы,
  hashes и native clocks. Но нет принятого применимого системного измеренного ряда
  для связанной установки. B-274 pool boiling не становится forced-flow/plant validation.
- **AI-09.06:** реализован отчёт раздельных data/parameter/closure/mesh/integrator
  uncertainty; calibration/held-out делятся целыми NIST series, подгонка не проводилась.
  Дополнительно реализован `atomic_model_uncertainty`: полная совместная covariance
  пяти аддитивных компонент остаточной ошибки, PSD/блочные проверки, Gaussian marginal
  интервалы и held-out coverage без предположения независимости. 12 тестов и
  manufactured пример прошли. Неизвестная экспериментальная covariance не заменена
  digitization bounds; экспериментальное покрытие остаётся неподтверждённым.
- **AI-09.08:** validation report содержит положительные и отрицательные случаи,
  области применимости и hashes. Независимый reviewer не назначен и не выдуман;
  собственная проверка реализации не названа независимым техническим разбором.
- **AI-11.06:** перенос архива, экспорт со схемой/единицами и cold-wheel replay
  выполнены локально вне checkout. Это не запуск на другой физической машине.
  Для внешнего runner подготовлен `scripts.verify_research_install`.

Эти задачи остаются открытыми до получения соответствующих внешних свидетельств.
`validated=false`, `plant_referenced=false` сохраняются для всей связанной установки.

## Запуск

```bash
atomic-multiphysics --output artifacts/my-numerical-qualification.json
atomic-research create examples/research/project.json artifacts/my-project
atomic-research run artifacts/my-project artifacts/my-run
atomic-research export artifacts/my-project artifacts/my-project.zip
atomic-research import artifacts/my-project.zip artifacts/relocated-project
atomic-research release-manifest artifacts/my-release-manifest.json
```

Для настоящего transport нужен OpenMC с подходящей ENDF/B-VIII.0 и лицензионным
sidecar: `python -m scripts.run_research_transport artifacts/my-openmc --seed 41`.
MPI-сокеты должны быть разрешены окружением. Deadline и ограничение ресурсов
не превращают запуск в sandbox произвольного пользовательского кода.

## Свидетельства

Протокол: `validation/ai_final17_protocol_v1.json`.
Численная/transport квалификация: `artifacts/final17_multiphysics_qualification_v3.json`.
Transport: `artifacts/final17_openmc_v4/replica-{0,1,2}/`.
Матрица: `validation/final17_supported_runtime_v1.json`.
UQ: `artifacts/final17_uq_qualification_v2.json`; внешний порядок приёмки:
`docs/EXTERNAL_ACCEPTANCE_HANDOFF_RU.md`.
Проверки и решение по задачам: `validation/final17_delivery_v2.json`.
Финальные тесты: 73 на исходниках и 73 в установленном wheel; 760 JSON/YAML ресурсов,
11 фаз прежнего runbook. Побайтовый witness: `validation/final17_build_witness_v2.json`.
OpenMC binary/Python source witness: `validation/final17_transport_runtime_v1.json`.
Архив: `artifacts/final17_research_publication_v1.tar.gz`.
Известные отказы не удаляются: прежний HIPER physical rejection и нулевая применимость
B-274 к текущему forced-boiling ядру сохраняют исходный статус.

Источники методов: [OpenMC MGXS](https://docs.openmc.org/en/stable/usersguide/mgxs.html),
[CE/CM и CRAM](https://docs.openmc.org/en/latest/methods/depletion.html),
[открытые библиотеки OpenMC](https://openmc.org/data/).
