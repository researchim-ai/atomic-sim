# Локальный интерфейс редактора EOS-проекта

`atomic_model_project_ui` 1.0.0 — браузерное рабочее место поверх того же
`atomic_model_projects.store`, который обслуживает CLI. Сохранение вызывает
тот же schema admission, CAS, immutable revision и atomic HEAD. Поддерживаются
параметры готовых EOS-пресетов и две объявленные baseline policies.

```bash
python -m atomic_model_projects create examples/eos_workspace/project.json --output artifacts/my_project
python -m atomic_model_project_ui artifacts/my_project --port 8765
```

После установки доступны `atomic-model-project` и `atomic-model-project-ui`.
Адрес — `http://127.0.0.1:8765`. `--port 0` выбирает свободный порт и печатает
фактический адрес. Завершение — Ctrl+C. `--exports DIR` задаёт доверенный корень
выходных проектов; браузер не может передать произвольный путь записи. По
умолчанию экспорт расположен рядом со store, в каталоге `<store>-exports`.

## Пользовательский маршрут

1. При открытии видны model card, fidelity, validated/plant_referenced,
   ограничения и project revision. Раздел «Версии и происхождение расчёта»
   раскрывает фактический source fingerprint и версии научных библиотек.
2. Выберите сценарий, пресет, action period, horizon, max_step, датчик цели,
   температуру и конечный tolerance. Имена/единицы наблюдений и действий показаны
   рядом. Политика редактируется полной JSON-конфигурацией.
3. Нажмите проверку сценария/политики: сервер проверит **весь** проект и покажет
   diff. Ревизия при preview не создаётся. Правка формы после preview делает
   прежнее предложение недействительным.
4. «Сохранить проверенные изменения» создаёт одну ревизию. Конфликт с изменением
   из другого окна/CLI отклоняется CAS-проверкой. Пока запрос выполняется,
   элементы формы блокируются: ранее проверенная конфигурация не выдаётся за
   позднее отредактированные поля.
5. История позволяет вернуть конфигурацию прежней ревизии новым commit.
   «Экспортировать» создаёт обычный portable project для campaign runner.

При неподдержанном preset, sensor или неполной карте action ports появляется
ошибка; HEAD остаётся прежним. Пример: переход mixed → uniform при старой полной
hold-карте портов требует согласованной политики. Незаявленные связи не
исправляются молча. Произвольный graph/topology editor, внешние experimental
таблицы, learner API и управление реальной установкой в этот UI не входят.

## Локальный HTTP-контракт

Сервер слушает только IPv4 loopback, не имеет `--host 0.0.0.0`. Это доверенное
**однопользовательское** рабочее место, без пользователей/ролей/TLS и не сервис
для публикации в интернете или недоверенной общей машине. Loopback/CSRF checks
не аутентифицируют другого локального пользователя или локальную программу.

API требует token из загруженной страницы; write также требует совпадающий
Origin. Проверяются Host и browser fetch-site, CORS не включён. Размер JSON body
ограничен 1 MiB, chunked transport не принимается, socket timeout — 10 секунд.
Произвольного file browser/upload/command execution endpoint нет.

CSP разрешает только локальные JS/CSS, запрещает eval/inline scripts, frames,
сторонние загрузки и формы. Ответы no-store/nosniff. Project data, notes, diff и
history выводятся через textContent; HTML из проекта не исполняется. После
перезапуска сервера страницу нужно перезагрузить, чтобы получить новый token.

GET `/api/inspect`; POST `/api/preview`, `/api/edit`, `/api/restore`, `/api/export`.
Edit/restore используют поля CAS backend API. Export принимает только `{}`;
конечный каталог создаёт сервер внутри заранее настроенного export root.

Изменение SDK source scan в этом выпуске изменило source identity. Старые
source-bound job/checkpoint/store требуют прежнего замороженного runtime;
автоматической миграции нет. Изменения editor UI сами по себе не меняют
термогидравлическую семантику: её хранит schema-admitted portable project.

## Проверка

17 HTTP integration tests: preview/commit/stale/restore/export, schema refusal,
Host/Origin/token, CSP/static assets, размер body, traversal и destination bounds.
Для них среда должна разрешать создание loopback sockets.

Опциональная Chromium-проверка (Playwright и установленный Chromium):

```bash
python -m scripts.verify_project_ui_browser --output artifacts/ui_browser_check
```

Проверяются реальная форма, read-only preview, инвалидирование preview после
правки, save/undo/export, exact native episode replay экспортированного проекта,
неисполняемая пользовательская заметка, отказ неправильной binding и mobile
layout без горизонтального overflow. CSP остаётся включённой. Скриншоты desktop,
mobile и ошибки сохраняются вместе с browser proof. Browser QA не является
физической валидацией; `validated=false`, `plant_referenced=false`.

Поставочный wheel проверяется вне checkout с теми же локальными сценариями,
HTTP/CLI и Chromium. Научные/dev-зависимости наследованы из окружения; hermetic
и полная OS/browser matrix не заявляются. Remote CI в этом сеансе не запускался.
