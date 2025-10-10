# Makefile для генерации документации

.PHONY: help pdf html clean

help:
	@echo "Доступные команды:"
	@echo "  make pdf     - Сгенерировать PDF документацию"
	@echo "  make html    - Сгенерировать HTML документацию"
	@echo "  make clean   - Очистить сгенерированные файлы"

pdf:
	@echo "Генерация PDF документации..."
	@echo "Требуется: pandoc, texlive"
	@command -v pandoc >/dev/null 2>&1 || { echo "Ошибка: pandoc не установлен. Установите: sudo apt install pandoc"; exit 1; }
	@mkdir -p build
	pandoc README.md QUICKSTART.md INSTALLATION.md ADVANCED_FEATURES.md PERFORMANCE.md \
		-o build/atomic-sim-documentation.pdf \
		--pdf-engine=pdflatex \
		--toc \
		--number-sections \
		-V geometry:margin=1in \
		-V lang=ru \
		--metadata title="Симулятор Атомного Реактора - Документация" \
		--metadata author="Atomic Sim Team" \
		--metadata date="$(shell date +%Y-%m-%d)"
	@echo "✓ PDF создан: build/atomic-sim-documentation.pdf"

html:
	@echo "Генерация HTML документации..."
	@mkdir -p build/html
	@for file in *.md; do \
		pandoc "$$file" -o "build/html/$${file%.md}.html" --standalone --toc; \
	done
	@echo "✓ HTML создан в: build/html/"

clean:
	@rm -rf build/
	@echo "✓ Очищено"

