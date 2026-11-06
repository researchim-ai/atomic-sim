# Актуальный порядок проверки

Порядок заменён по указанию пользователя: библиотека проверяется локально,
без CI, GitHub-интеграции, внешнего runner или аттестации машины.

Команда, охват тестов, установка wheel и replay описаны в
[локальной проверке библиотеки](LOCAL_VERIFICATION_RU.md).

Сравнение готовых пакетов также остаётся локальным:

```bash
atomic-acceptance prepare examples/research/project.json artifacts/my-request
atomic-acceptance verify-packet artifacts/my-request
atomic-acceptance compare request.json project-receipt.json --output comparison.json
```

GitHub API collector, отдельный workflow и пакуемая копия workflow удалены.
Старые `external_acceptance_*` receipts относятся к предыдущему порядку и сохранены
как исторические свидетельства, а не как обязательные требования.

Проверка публикаций MOTEL/NSTF/NIST и различение лицензии статьи и прав на исходные
измерения сохранены. В частности, MOTEL-публикация CC BY4.0 находится в
`atomic_model_acceptance/data/motel_2025/`; metadata admission не превращает её
в отсутствующие raw measurements. Экспериментальная валидация и независимый
технический разбор остаются отдельными научными направлениями.
