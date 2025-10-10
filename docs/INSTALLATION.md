# 📦 Руководство по установке

## Требования

- **Python**: 3.8 или выше
- **Операционная система**: Linux, macOS, Windows
- **Память**: Минимум 4 ГБ RAM (8 ГБ рекомендуется)
- **Диск**: 500 МБ для кода и зависимостей

## Методы установки

### 1. Быстрая установка (только использование)

```bash
git clone https://github.com/yourusername/atomic-sim.git
cd atomic-sim
pip install -r requirements.txt
python tests/test_simulator.py  # Проверка
```

### 2. Установка пакетом (рекомендуется)

```bash
git clone https://github.com/yourusername/atomic-sim.git
cd atomic-sim

# Установка в editable mode
pip install -e .

# Проверка установки
atomic-sim --help
python -c "import reactor; print(reactor.__version__)"
```

### 3. Установка для разработки

```bash
git clone https://github.com/yourusername/atomic-sim.git
cd atomic-sim

# Установка с dev-зависимостями
pip install -e ".[dev]"

# Запуск тестов
pytest tests/ -v

# Проверка качества кода
black --check reactor/ tests/
ruff check reactor/
```

### 4. Установка в виртуальном окружении (рекомендуется)

```bash
# Создание виртуального окружения
python -m venv venv

# Активация
source venv/bin/activate  # Linux/macOS
# или
venv\Scripts\activate  # Windows

# Установка
pip install -e .

# Для выхода из виртуального окружения
deactivate
```

### 5. Установка с conda

```bash
# Создание conda окружения
conda create -n atomic-sim python=3.10
conda activate atomic-sim

# Установка PyTorch (выберите версию под вашу систему)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Установка остальных зависимостей
pip install -e .

# Проверка
python tests/test_simulator.py
```

## Проверка установки

### Базовая проверка

```python
python -c "
from reactor import ReactorSimulator
sim = ReactorSimulator()
history = sim.run(duration=10.0)
print(f'✓ Симуляция работает! {len(history[\"time\"])} шагов')
"
```

### Полная проверка

```bash
# Все тесты
pytest tests/ -v

# Или быстрый smoke-test
python tests/test_simulator.py
```

Должны пройти 34 теста за ~70 секунд.

## Решение проблем

### Проблема: ModuleNotFoundError

```bash
# Убедитесь что вы в правильной директории
cd /path/to/atomic-sim

# Проверьте PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Или установите как пакет
pip install -e .
```

### Проблема: torch not found

```bash
# Установите PyTorch
pip install torch>=2.0.0

# Или с conda
conda install pytorch -c pytorch
```

### Проблема: тесты падают

```bash
# Переустановите зависимости
pip install -r requirements.txt --upgrade

# Проверьте версии
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

### Проблема: медленная работа

```bash
# Используйте float32 вместо float64
simulator = ReactorSimulator(dtype=torch.float32)

# Или батч-симуляцию
from reactor.batch_simulator import BatchReactorSimulator
sim = BatchReactorSimulator(batch_size=16, dtype=torch.float32)
```

## Обновление

```bash
cd atomic-sim
git pull origin main
pip install -e . --upgrade
```

## Удаление

```bash
# Если установлено как пакет
pip uninstall atomic-sim

# Удалить файлы
rm -rf atomic-sim/

# Удалить виртуальное окружение
rm -rf venv/  # или conda remove -n atomic-sim --all
```

## Дополнительные компоненты

### GPU поддержка (опционально)

```bash
# Для NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Проверка
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Jupyter notebooks (опционально)

```bash
pip install jupyter
jupyter notebook

# Создайте notebook с:
from reactor import ReactorSimulator
# ...
```

## Версии

- **Стабильная**: v1.1.0
- **Разработка**: main branch
- **Python**: 3.8, 3.9, 3.10, 3.11 (тестируется)

## Поддержка

Если возникли проблемы:
1. Проверьте [FAQ](https://github.com/yourusername/atomic-sim/wiki/FAQ)
2. Создайте [Issue](https://github.com/yourusername/atomic-sim/issues)
3. См. [CONTRIBUTING.md](CONTRIBUTING.md)

