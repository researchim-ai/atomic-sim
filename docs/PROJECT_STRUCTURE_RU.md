# Структура репозитория

Код библиотеки размещён в `src/`. Имена существующих Python-пакетов и CLI
сохранены: например, `atomic_pwr_agent`, `atomic_agent_tools`, `sdk`,
`reactor`, `atomic-training`. Перенос не объединяет разные физические модели
под одним названием и не меняет их область применимости.

Основные каталоги:

- `src/` — устанавливаемые Python-пакеты и модуль генерации датасетов.
- `apps/` — самостоятельные приложения; `apps/gui/app.py` — прежний GUI.
- `tests/` — проверки, не устанавливаемые как часть библиотеки.
- `examples/` — пользовательские примеры и входные проекты.
- `docs/` — документация; `assets/branding/` и `assets/demos/` — изображения.
- `configs/` — примеры пользовательских конфигураций.
- `requirements/` — зависимости документации и проверенный исследовательский профиль.
- `scripts/` — инструменты локальной проверки и внутренние численные drivers.
- `data/`, `scenarios/`, `validation/` — каталоги данных, сценариев и свидетельств.
- `artifacts/` — локальные результаты, сборки и отчёты; содержимое игнорируется Git,
  кроме README. Сборка документации записывается в `artifacts/site/`, рисунки
  демонстраций — в `artifacts/demos/`.

`scripts`, `data`, `scenarios`, `validation` пока остаются отдельными
устанавливаемыми пакетами в корне: на их логические имена ссылаются существующие
численные drivers, манифесты и контрольные суммы. Их расположение явно задано
в `pyproject.toml`. Научные свидетельства и первичные данные при переносе
не переписываются ради косметических изменений путей.

## Разработка и запуск

Из корня проекта:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_pwr_agent.py -q
python -m examples.llm_controlled_episode
python -m scripts.verify_local_library
```

Для сборки sdist/wheel нужен пакет `build`:

```bash
python -m pip install build
python -m build --outdir artifacts/dist
```

Установка из wheel поддерживает те же Python-импорты и CLI. Для запуска
примеров из `examples/` нужен checkout. Старые команды прямого запуска
`python generate_dataset.py` заменяются `atomic-sim` или
`python -m generate_dataset`; генератор Q&A запускается через
`python -m dataset_builders.qa_from_traces`. GUI: `python apps/gui/app.py`.

Конфигурация pytest добавляет `src/` и корень в пути импорта; pytest, Ruff
и mypy хранят кэши в `.cache/`. Подпроцессы разрешают пакеты через editable-
установку; тесты с изолированным PYTHONPATH явно учитывают обе части проекта.
Основная конфигурация сборки — `pyproject.toml`; дублирующий `setup.py` удалён.

Предыдущие локальные build/dist/site и кэши сохранены в
`artifacts/layout-migration-original/`. Старые пакеты checkpoint и отчёты
остаются историческими свидетельствами. Source-bound checkpoint проверяют
точную реализацию: после изменений поиска ресурсов старые checkpoint нужно
воспроизводить прежней версией проекта, а для новой версии создавать новые.
Сам перенос файлов не является повторной физической валидацией.
