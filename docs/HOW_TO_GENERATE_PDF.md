# 📄 Генерация PDF документации

## Быстрая инструкция

```bash
# 1. Установить зависимости для документации
pip install -r requirements-docs.txt

# 2. Установить Chromium для Playwright (только первый раз)
playwright install chromium

# 3. Собрать HTML документацию
mkdocs build

# 4. Экспортировать в PDF
python scripts/export_pdf.py

# Результат: site/pdf/AtomicSimDocs.pdf
```

## Подробная инструкция

### Шаг 1: Установка зависимостей

```bash
pip install -r requirements-docs.txt
```

**Что устанавливается:**
- `mkdocs` - генератор статических сайтов
- `mkdocs-material` - красивая тема
- `mkdocs-print-site-plugin` - плагин для создания print-версии
- `pymdown-extensions` - расширения Markdown (формулы, диаграммы)
- `playwright` - браузерный движок для PDF

### Шаг 2: Установка браузера

```bash
playwright install chromium
```

Скачивает Chromium (~100 МБ). **Нужно только один раз!**

### Шаг 3: Сборка HTML

```bash
mkdocs build
```

**Что происходит:**
1. Читает `mkdocs.yml` (конфигурация)
2. Обрабатывает все `.md` файлы из `docs/`
3. Генерирует HTML в `site/`
4. Плагин `print-site` создает объединенную страницу в `site/print_page/`

**Результат:**
- `site/` - обычная версия сайта
- `site/print_page/index.html` - версия для печати

### Шаг 4: Экспорт в PDF

```bash
python scripts/export_pdf.py
```

**Что происходит:**
1. Запускает Chromium в headless режиме
2. Открывает `site/print_page/index.html`
3. Ждет загрузки MathJax (формулы) и Mermaid (диаграммы)
4. Экспортирует в PDF с правильным форматированием

**Результат:** `site/pdf/AtomicSimDocs.pdf`

## Просмотр результата

### Linux
```bash
xdg-open site/pdf/AtomicSimDocs.pdf
```

### macOS
```bash
open site/pdf/AtomicSimDocs.pdf
```

### Windows
```bash
start site/pdf/AtomicSimDocs.pdf
```

## Предварительный просмотр (опционально)

Перед генерацией PDF можно посмотреть HTML версию:

```bash
# Запустить локальный сервер
mkdocs serve

# Открыть в браузере
# http://127.0.0.1:8000

# Страница для печати доступна по:
# http://127.0.0.1:8000/print_page/
```

## Настройка PDF

### Изменить margins

Отредактируйте `scripts/export_pdf.py`:

```python
await page.pdf(
    path=str(output_pdf),
    format='A4',
    margin={
        'top': '25mm',    # Увеличить отступ сверху
        'right': '20mm',
        'bottom': '25mm',
        'left': '20mm'
    },
    # ...
)
```

### Изменить header/footer

```python
header_template='<div style="...">Ваш текст заголовка</div>',
footer_template='<div style="...">Страница <span class="pageNumber"></span></div>',
```

### Изменить формат страницы

```python
format='A4',  # или 'Letter', 'Legal', 'A3'
```

## Решение проблем

### Проблема: playwright не найден

```bash
pip install playwright
playwright install chromium
```

### Проблема: MathJax не отображается

В `scripts/export_pdf.py` увеличьте время ожидания:

```python
await page.wait_for_timeout(5000)  # 5 секунд вместо 3
```

### Проблема: файл слишком большой

**Оптимизация:**

1. Уменьшите изображения в `docs/`
2. Используйте сжатие:
   ```python
   await page.pdf(..., prefer_css_page_size=True)
   ```

3. Разбейте на несколько PDF (по разделам)

### Проблема: диаграммы Mermaid не видны

Проверьте что в `mkdocs.yml` включен superfences:

```yaml
markdown_extensions:
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
```

## Альтернативные методы

### Метод 1: Pandoc (без JavaScript)

```bash
# Установить pandoc
sudo apt install pandoc texlive-xetex

# Объединить все MD файлы и конвертировать
cd docs
pandoc *.md -o documentation.pdf --pdf-engine=xelatex --toc -V lang=ru
```

**Минусы:** Нет Mermaid диаграмм, MathJax может работать не идеально.

### Метод 2: Браузер вручную

1. `mkdocs serve`
2. Открыть http://127.0.0.1:8000/print_page/
3. Ctrl+P (Печать)
4. "Сохранить как PDF"

**Минусы:** Ручной процесс, может быть неконсистентно.

### Метод 3: WeasyPrint (CSS-based)

```bash
pip install weasyprint
weasyprint site/print_page/index.html output.pdf
```

**Минусы:** Может не поддерживать все CSS3 фичи.

## Качество PDF

Наш метод (Playwright) обеспечивает:

✅ **MathJax** - все формулы рендерятся корректно  
✅ **Mermaid** - диаграммы экспортируются как SVG  
✅ **CSS** - полная поддержка стилей  
✅ **Fonts** - правильные шрифты для кода и текста  
✅ **Colors** - сохранение цветов (print_background=True)  
✅ **TOC** - оглавление из mkdocs  
✅ **Navigation** - правильная нумерация страниц  

## Автоматизация

### Makefile

```makefile
.PHONY: docs pdf

docs:
    mkdocs build

pdf: docs
    python scripts/export_pdf.py

clean:
    rm -rf site/
```

Использование:
```bash
make pdf
```

### GitHub Actions (CI)

```yaml
# .github/workflows/docs.yml
name: Generate Docs

on:
  push:
    branches: [main]

jobs:
  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements-docs.txt
          playwright install chromium
      - name: Build docs
        run: mkdocs build
      - name: Export PDF
        run: python scripts/export_pdf.py
      - name: Upload PDF
        uses: actions/upload-artifact@v3
        with:
          name: documentation-pdf
          path: site/pdf/*.pdf
```

## Структура сгенерированного PDF

1. **Титульная страница** (cover page)
2. **Оглавление** (TOC) с номерами страниц
3. **Главная страница** (index)
4. **Быстрый старт**
5. **Установка**
6. **Архитектура** с диаграммами
7. **Математика** с формулами
8. **Физика** подробно
9. **Расширенные возможности**
10. **Производительность**
11. **Contributing**
12. **Changelog**
13. **Summary**

Всего: **~100-150 страниц** в зависимости от контента.

## Советы

1. **Проверьте HTML** перед PDF: `mkdocs serve`
2. **Формулы**: используйте `$...$` для inline, `\[...\]` для display
3. **Диаграммы**: Mermaid поддерживается автоматически
4. **Изображения**: кладите в `docs/images/` или используйте относительные пути
5. **Таблицы**: Markdown tables работают отлично

---

**После генерации у вас будет профессиональная PDF документация со всеми формулами и диаграммами!**

