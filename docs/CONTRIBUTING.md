# 🤝 Руководство по внесению вклада

Спасибо за интерес к проекту! Мы приветствуем любые улучшения.

## 🚀 Быстрый старт для разработчиков

```bash
# Клонировать репозиторий
git clone https://github.com/yourusername/atomic-sim.git
cd atomic-sim

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или venv\Scripts\activate  # Windows

# Установить в режиме разработки
pip install -e ".[dev]"

# Запустить тесты
pytest tests/ -v
```

## 🧪 Разработка

### Структура проекта

```
atomic-sim/
├── reactor/              # Основной код симулятора
│   ├── neutronics.py    # Нейтронная кинетика
│   ├── thermal.py       # Тепловая модель
│   ├── control.py       # Система управления
│   ├── simulator.py     # Базовый симулятор
│   ├── simulator_advanced.py  # Расширенный симулятор
│   ├── poisoning.py     # Xe/I отравление
│   ├── events.py        # События
│   └── batch_simulator.py  # Батч-симуляция
├── tests/               # Тесты
├── examples/            # Примеры
├── envs/                # Gymnasium обёртки
├── dataset_builders/    # Генераторы датасетов
├── configs/             # Конфигурации
└── scenarios/           # Сценарии событий
```

### Добавление новых фич

1. **Создайте ветку**
   ```bash
   git checkout -b feature/my-feature
   ```

2. **Пишите код**
   - Следуйте существующему стилю
   - Добавьте docstrings
   - Используйте type hints где возможно

3. **Добавьте тесты**
   ```bash
   # Создайте тест в tests/
   pytest tests/test_my_feature.py -v
   ```

4. **Проверьте код**
   ```bash
   # Форматирование
   black reactor/ tests/ --line-length 100
   
   # Линтер
   ruff check reactor/ tests/
   ```

5. **Запустите все тесты**
   ```bash
   pytest tests/ -v
   ```

6. **Создайте Pull Request**

## 📝 Стиль кода

- **PEP 8** для Python
- **Длина строки:** 100 символов
- **Docstrings:** Google style
- **Type hints:** желательны для публичных API

Пример:
```python
def my_function(param1: float, param2: Optional[str] = None) -> Dict[str, Any]:
    """
    Краткое описание функции
    
    Args:
        param1: описание параметра 1
        param2: описание параметра 2
    
    Returns:
        описание возвращаемого значения
    """
    # Код
    return result
```

## 🧪 Тестирование

### Написание тестов

```python
import pytest
from reactor import ReactorSimulator

class TestMyFeature:
    """Тесты моей функции"""
    
    def test_basic_behavior(self):
        """Описание теста"""
        sim = ReactorSimulator()
        sim.reset()
        
        # Действия
        result = sim.my_new_method()
        
        # Проверки
        assert result is not None
        assert result > 0
```

### Запуск конкретного теста

```bash
pytest tests/test_my_feature.py::TestMyFeature::test_basic_behavior -v
```

## 📖 Документация

### Обновление документации

- **README.md** - основная документация
- **ADVANCED_FEATURES.md** - продвинутые функции
- **CHANGELOG.md** - история изменений
- **Docstrings** - в коде

### Генерация документации (TODO)

```bash
# Sphinx documentation (будущая фича)
cd docs
make html
```

## 🐛 Сообщения об ошибках

При создании issue включите:
- Версию Python и PyTorch
- Операционную систему
- Минимальный код для воспроизведения
- Ожидаемое поведение
- Фактическое поведение
- Traceback если есть

## ✨ Идеи для вклада

### Легко (good first issue)

- Добавить новые тесты
- Улучшить документацию
- Исправить опечатки
- Добавить примеры использования

### Средне

- Новые типы событий
- Дополнительные сценарии
- Улучшения визуализации
- Оптимизация производительности

### Сложно

- Многомерная нейтронная диффузия
- Детальная термогидравлика
- MPC/LQR контроллеры
- Батч-симуляция на GPU с torch.compile

## 📬 Связь

- **Issues:** GitHub Issues
- **Discussions:** GitHub Discussions
- **Pull Requests:** приветствуются!

## 📜 Лицензия

Проект распространяется под MIT лицензией. См. LICENSE в корне репозитория.

---

**Спасибо за вклад в проект! 🙏**

