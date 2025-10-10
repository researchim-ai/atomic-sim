#!/usr/bin/env python3
"""
Экспорт документации в PDF через Playwright
Обеспечивает корректный рендеринг MathJax и Mermaid диаграмм
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright


async def export_pdf():
    """Экспорт HTML документации в PDF"""
    
    # Пути
    project_root = Path(__file__).parent.parent
    print_page = project_root / "site" / "print_page" / "index.html"
    output_pdf = project_root / "site" / "pdf" / "AtomicSimDocs.pdf"
    
    # Проверка входного файла
    if not print_page.exists():
        print(f"❌ Ошибка: {print_page} не найден!")
        print("   Сначала запустите: mkdocs build")
        return False
    
    # Создание выходной директории
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("ЭКСПОРТ ДОКУМЕНТАЦИИ В PDF")
    print("=" * 70)
    print(f"\nВходной файл: {print_page}")
    print(f"Выходной PDF: {output_pdf}")
    
    async with async_playwright() as p:
        # Запуск браузера
        print("\n1. Запуск браузера Chromium...")
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        # Загрузка страницы
        print(f"2. Загрузка документации...")
        await page.goto(f"file://{print_page.absolute()}", wait_until="networkidle")
        
        # Ожидание рендеринга MathJax
        print("3. Ожидание рендеринга формул (MathJax)...")
        
        # Ждем загрузки MathJax
        await page.wait_for_timeout(3000)
        
        # Проверяем что MathJax готов и запускаем рендеринг всех формул
        await page.evaluate("""
            async () => {
                if (window.MathJax) {
                    await window.MathJax.typesetPromise();
                    console.log('MathJax typeset complete');
                }
            }
        """)
        
        # Даем дополнительное время на рендеринг
        await page.wait_for_timeout(5000)
        
        # Проверяем количество отрендеренных формул
        math_count = await page.evaluate("""
            () => {
                const arithmatex = document.querySelectorAll('.arithmatex').length;
                const mjx = document.querySelectorAll('mjx-container').length;
                return { arithmatex, mjx };
            }
        """)
        print(f"   ✓ Найдено: {math_count['arithmatex']} arithmatex блоков, {math_count['mjx']} MathJax элементов")
        
        # Ожидание Mermaid диаграмм
        print("4. Проверка диаграмм (Mermaid)...")
        
        # Проверяем различные селекторы Mermaid
        selectors_to_try = [
            '.mermaid svg',
            'svg[id^="mermaid"]',
            'svg[class*="mermaid"]',
            '.language-mermaid',
            'pre.mermaid'
        ]
        
        mermaid_found = False
        for selector in selectors_to_try:
            count = await page.locator(selector).count()
            if count > 0:
                print(f"   ✓ Найдено элементов по селектору '{selector}': {count}")
                mermaid_found = True
                break
        
        if not mermaid_found:
            print("   ℹ Mermaid диаграммы в виде code blocks (нормально для некоторых тем)")
        
        # Дополнительная задержка для полного рендеринга
        print("   Финальное ожидание...")
        await page.wait_for_timeout(5000)
        
        # Экспорт в PDF
        print("5. Генерация PDF...")
        await page.pdf(
            path=str(output_pdf),
            format='A4',
            print_background=True,
            margin={
                'top': '20mm',
                'right': '15mm',
                'bottom': '20mm',
                'left': '15mm'
            },
            display_header_footer=True,
            header_template='<div style="font-size:9pt; width:100%; text-align:center; color:#666;">Симулятор Атомного Реактора v1.1.0</div>',
            footer_template='<div style="font-size:9pt; width:100%; text-align:center; color:#666;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
        )
        
        await browser.close()
    
    print(f"\n✅ PDF успешно создан: {output_pdf}")
    print(f"   Размер: {output_pdf.stat().st_size / 1024:.1f} КБ")
    
    return True


def main():
    """Основная функция"""
    import sys
    
    # Проверка что site/ существует
    site_dir = Path(__file__).parent.parent / "site"
    if not site_dir.exists():
        print("❌ Директория site/ не найдена!")
        print("\nСначала соберите документацию:")
        print("  mkdocs build")
        sys.exit(1)
    
    # Запуск экспорта
    success = asyncio.run(export_pdf())
    
    if not success:
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("ГОТОВО! PDF документация создана.")
    print("=" * 70)
    print("\nДля просмотра:")
    print("  xdg-open site/pdf/AtomicSimDocs.pdf  # Linux")
    print("  open site/pdf/AtomicSimDocs.pdf      # macOS")
    print("  start site/pdf/AtomicSimDocs.pdf     # Windows")


if __name__ == '__main__':
    main()

