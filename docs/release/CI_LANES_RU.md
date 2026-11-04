# Раздельные проверочные профили

Версия плана 1.62.0, 4 ноября 2026. AI-12.03.

`.github/workflows/ci.yml` автоматически запускает portable regression на Linux
с запрошенной Python matrix 3.10/3.11/3.12. Это конфигурация CI, а не утверждение,
что все три версии уже испытаны локально. Qualification этого блока выполнена
на версии из delivery receipt; supported dependency/OS matrix AI-12.01 открыта.

Профили взаимно разделены функцией `atomic_model_release.lanes.classify`:

- portable: функциональные проверки без external_integration, long_running,
  hardware_specific; содержит новое ядро, агентские API, UI/HTTP, data guards.
- long: отдельный manual workflow `numerical.yml`, совмещённое numerical coverage
  с прежним порогом 80%, legacy smoke и сохранённые прежние run/audit CLI-проверки.
  Они перенесены из PR workflow, а не удалены. Общий численный прогон дорогой.
- OpenMC: `external.yml`, только workflow_dispatch + явный run_openmc,
  self-hosted label atomic-openmc, проверка binary/cross_sections/license sidecar,
  затем настоящий external integration. Наличие файлов — только preflight.
- MPI: отдельный explicit input и label atomic-mpi. Два rank проверяют gather
  и barrier; receipt **не квалифицирует MPI reactor solver**. API взят из
  [официального tutorial mpi4py](https://mpi4py.readthedocs.io/en/stable/tutorial.html).
- hardware: manual `hardware.yml`, self-hosted label atomic-performance и
  обязательный ATOMIC_HARDWARE_PROFILE. Прежние legacy throughput thresholds
  и реальные EOS process/pool workloads сохранены отдельно от portable gate.
  Достижение 2x не следует из запуска этого профиля.

`pytest_gate` создаёт receipt с collected/passed/failed/skipped/errors и exit code.
`--atomic-require-executed` не допускает пустого успеха; `--atomic-no-skips`
не допускает подмену external runtime пропуском теста. Ошибка collection,
fixture или теста сохраняет ненулевой exit. Xfail считается пропуском для strict lane.

```bash
python -m pytest tests/ -q \
  -m "not external_integration and not long_running and not hardware_specific" \
  -p atomic_model_release.pytest_gate --atomic-require-executed \
  --atomic-receipt artifacts/portable-receipt.json
```

Для provisioned внешнего runner:

```bash
atomic-model-release preflight openmc
python -m pytest tests/ -q -m external_integration \
  -p atomic_model_release.pytest_gate --atomic-require-executed --atomic-no-skips \
  --atomic-receipt artifacts/openmc-receipt.json
atomic-model-release preflight mpi
timeout 60s mpiexec -n 2 python -m atomic_model_release.mpi_probe \
  --output artifacts/mpi-transport.json
```

В текущем окружении отдельно выполнены реальный OpenMC test и MPI transport на
двух rank. Это локальные результаты из evidence, не запуск remote GitHub Actions
и не физическая валидация модели. Long/hardware workflow целиком не выполнялись
этим блоком; их separation проверена структурно и тестами классификации.
