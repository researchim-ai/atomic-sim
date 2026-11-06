# Установка

Пакет требует Python 3.10 или новее согласно `pyproject.toml`.
Проверенный локальный профиль — Linux x86_64, CPython 3.12.9,
NumPy 2.2.6 и SciPy 1.15.2. Успешная проверка этого профиля не означает,
что остальные платформы уже прошли такую же квалификацию.

## Из исходников

```bash
git clone https://github.com/researchim-ai/atomic-sim.git
cd atomic-sim
python -m venv .venv
```

Активируйте окружение:

```bash
source .venv/bin/activate
```

В Windows PowerShell вместо этой команды используйте
`.venv\Scripts\Activate.ps1`.

Установите библиотеку; зависимости перечислены в `pyproject.toml`:

```bash
python -m pip install -e .
atomic-sim --help
python -c "from atomic_pwr_agent import Episode; print(Episode().observe()['model_id'])"
```

`src/` — корень исходников Python. Одна установка `requirements.txt` не
заменяет установку библиотеки для CLI и дочерних процессов.
Имена Python-импортов после переноса в `src/` сохранены.

## Разработка и локальные тесты

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_pwr_agent.py -q
```

Полная локальная проверка поставки использует также `build`:

```bash
python -m pip install build
python -m scripts.verify_local_library
```

Она собирает sdist/wheel, устанавливает wheel вне checkout и проверяет replay.
Офлайн-установка в этой проверке переиспользует уже установленные научные
зависимости; это не автономный набор всех зависимостей.
[Подробности и расширенные наборы тестов](LOCAL_VERIFICATION_RU.md).
CI, GitHub-аккаунт и удалённый runner не нужны.

## Дополнительные компоненты

Для прежнего учебного GUI:

```bash
python -m pip install -e '.[gui]'
python apps/gui/app.py
```

Для документации:

```bash
python -m pip install -r requirements/docs.txt
python -m mkdocs build
```

Сайт создаётся в `artifacts/site/`. Дополнительные файлы исследовательского
профиля находятся в `requirements/research-qualified.txt` и
`requirements/research-environment.constraints.txt`. Это описание проверенной
среды, не герметичный lockfile и не требование устанавливать все версии
на любую платформу.

Для сборки дистрибутива:

```bash
python -m build --outdir artifacts/dist
```

Готовый wheel устанавливается обычной командой `python -m pip install PATH.whl`.
Импорты и CLI работают без checkout; примеры `examples/` и приложение `apps/`
запускаются из исходного репозитория. OpenMC и ядерные данные для специальных
transport-проверок устанавливаются отдельно; обычный пример PWR их не требует.

[Быстрый старт](QUICKSTART.md) · [Структура проекта](PROJECT_STRUCTURE_RU.md)
