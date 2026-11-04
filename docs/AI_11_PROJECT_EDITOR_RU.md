# Версионированный редактор параметров EOS-проекта

`atomic_model_projects` 1.0.0 добавляет API и CLI для безопасного редактирования
поддержанного `eos-thermal-assembly-v1`. Редактор показывает паспорт модели,
текущие конфигурации, имена/единицы наблюдений и действий и историю ревизий.
Проект проверяется тем же admission-кодом, который использует campaign runner.

```bash
atomic-model-project create examples/eos_workspace/project.json --output artifacts/project_store
atomic-model-project inspect artifacts/project_store
atomic-model-project preview artifacts/project_store request.json
atomic-model-project edit artifacts/project_store request.json
atomic-model-project export artifacts/project_store --output artifacts/edited_project
atomic-model-workspace initialize artifacts/edited_project/project.json --output artifacts/edited_campaign
atomic-model-workspace run artifacts/edited_campaign
```

Эквивалент: `python -m atomic_model_projects`. Выходные каталоги должны быть
новыми. `inspect` возвращает актуальные revision и project_sha256. Пример запроса
с подстановкой **фактического** digest из inspect:

```json
{
  "schema_version": 1,
  "expected_revision": 0,
  "expected_project_sha256": "DIGEST_FROM_INSPECT",
  "operations": [
    {
      "op": "set_scenario_config",
      "id": "mixed_dev",
      "value": {"episode_horizon_s": 0.04}
    }
  ],
  "note": "Увеличен горизонт контрольного случая"
}
```

Preview показывает изменённые поля, прежние/новые значения, предложенный digest
и model card. Preview не пишет ревизию. Edit заново проверяет ожидаемую версию
под writer lock; изменения другого writer между preview и edit приводят к
отказу stale revision. Ревизия и digest обязательны: молчаливого last-write-wins нет.

## Поддержанные операции

`set_scenario_config` меняет именованные config-поля; `set_objective` заменяет
полную цель. `replace_scenario`/`replace_policy` заменяют объект целиком.
`add_scenario`/`add_policy` добавляют новый ID; `remove_scenario`/`remove_policy`
удаляют существующий. Для одного ID в одной группе разрешена одна операция за
транзакцию. Для одновременного изменения config/objective используют полный
`replace_scenario`. Неизвестные поля/версии/операции и пустые изменения отклоняются.

Весь пакет из 1..64 операций сначала применяется к копии и проходит полную
проверку всех сценариев и политик. Только после этого публикуется одна ревизия.
Нельзя сохранить пустой проект, неизвестный preset, неполную карту heat ports,
несуществующий sensor или выход за численные ограничения. Смена preset также
проверяет совместимость **всех** политик со всеми сценариями. Нет скрытой
автоматической достройки отсутствующих портов или изменения модели.

Это редактор поддержанных параметров/пресетов и политик. Произвольный граф,
геометрия/оборудование установки и внешние экспериментальные datasets не входят
в это API. [Локальный визуальный интерфейс](AI_11_LOCAL_PROJECT_UI_RU.md) поставлен
отдельным пакетом; AI-11.02 принят в ограниченном EOS-preset scope. AI-11.01 и
общая topology/data integration остаются открытыми.

## История, публикация и восстановление

Store содержит `HEAD.json` и `revisions/<sha256>.json`. Каждая revision включает
полный нормализованный проект, parent hash, номер, source identity, operation и
заметку до 512 символов. Число ревизий ограничено 64. История проверяется целиком:
схемы, hash, последовательность parent, project admission и source identity.
Чужой source или неизвестная schema требуют явного решения о миграции; тихого
изменения смысла старого store нет.

Новая revision создаётся отдельным файлом, выполняется fsync; на POSIX также
синхронизируется каталог revisions. Затем публикуется атомарный HEAD с fsync
родительского каталога. Сбой публикации оставляет прежний HEAD действующим;
недостижимая revision считается orphan. Повтор той же операции может использовать
тот же проверенный orphan без перезаписи его содержимого. Это протокол локальной
файловой системы; гарантии неисправного оборудования или distributed filesystem
в этот scope не входят.

Возврат конфигурации создаёт **новую** ревизию, ссылаясь на текущую историю:

```bash
atomic-model-project restore artifacts/project_store --expected-revision 1 --expected-project-sha256 CURRENT_DIGEST --target-revision 0
```

Восстанавливается конфигурация проекта. Уже созданная кампания хранит собственный
нормализованный input и продолжает свой расчёт. Состояния физических эпизодов
восстанавливаются отдельным job checkpoint API.

Export создаёт обычные относительные project/scenario/policy JSON и файл
`editor_origin.json`. Такой проект принимается прежним runner без специальной
семантики редактора. Workspace data archive переносит текущую конфигурацию и
campaign checkpoints; полная editor history не является частью этого data ZIP.
Для сохранения истории нужен сам store с его revision-файлами. Контрольные суммы
проверяют целостность, но не аутентифицируют автора.

## Сквозная проверка

```bash
python -m examples.eos_project_edit_workflow --output artifacts/project_editor_demo
```

Пример: import → preview → edit → stale-edit rejection → export → шесть jobs →
точный повтор edited native episode → archive/import → restore конфигурации
новой ревизией. Все derived projects/episode checks связаны digest и source.

Проверены невалидный последний элемент batch, конфликтующие операции, несовместимый
preset/policy mask, stale writer, неизвестная schema, порча history, path escape,
сбой записи HEAD, ограничение числа ревизий и CLI вне исходного каталога.
Linux/Python 3.12 и wheel вне checkout проверяются отдельно; поддержка других
матриц не выводится из syntax parsing. Модель остаётся экспериментальной:
`validated=false`, `plant_referenced=false`.
