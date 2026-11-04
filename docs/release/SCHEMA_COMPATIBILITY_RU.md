# Schema compatibility и явная data-only миграция

AI-12.05. Поддержан текущий workspace project schema1 и study snapshot schema1.
Golden project поставляется внутри wheel с относительными scenario/policy refs.
Новая версия схемы не угадывается, единицы не переименовываются автоматически.
Schema=2, лишние поля, сломанные refs, symlinks и выход за project root отклоняются.

Единственная новая миграция — **workspace-v1 → study-v1** для конфигурации:

```bash
atomic-model-release migrate-project PATH/project.json --output artifacts/migrated
```

Создаются current-runtime study, переносимый snapshot и повторно импортированная
копия. Сравнивается вся нормализованная модель/scenarios/policies/limits и её SHA,
а не только названия. Receipt `migration.json` имеет semantic_project_equal=true
и source_checkpoints_migrated=false. Существующий output не перезаписывается.

Native/agent checkpoints сохраняют строгую source binding. Миграция **не**
переписывает source/version старого checkpoint, не выдумывает historical schema0
и не продолжает старую траекторию на новых уравнениях. При source drift используйте
соответствующий frozen runtime либо перенесите только исходную конфигурацию и
начните новый расчёт с новым provenance. Не исправляйте JSON source/hash вручную.

Тесты проверяют schema mismatch и source drift, порчу units/oracle, roundtrip
полной конфигурации и отказ до создания output для неизвестных/небезопасных refs.
Совместимость future schema2 или автоматический перенос всех исторических engines
не заявляются. Scientific runtime fingerprint этим блоком не изменён.
