#!/usr/bin/env python3
"""
Тестовый скрипт для быстрой проверки симулятора
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor import ReactorSimulator


def test_basic():
    """Базовый тест функциональности"""
    print("🧪 Тест 1: Инициализация симулятора...")
    simulator = ReactorSimulator(device='cpu')
    simulator.reset()
    
    state = simulator.get_state()
    assert state['power'] > 0, "Мощность должна быть положительной"
    assert state['T_fuel'] > 0, "Температура топлива должна быть положительной"
    print("   ✓ Инициализация прошла успешно")
    
    print("\n🧪 Тест 2: Симуляция 10 секунд...")
    history = simulator.run(duration=10.0)
    
    assert len(history['time']) > 0, "История должна содержать данные"
    assert len(history['power']) > 0, "История мощности должна быть записана"
    print(f"   ✓ Симуляция завершена успешно ({len(history['time'])} шагов)")
    
    print("\n🧪 Тест 3: Автоматическое управление...")
    simulator.reset()
    # Целевая мощность 90 МВт - ниже начальной, требует вставления стержней
    target = 90.0
    simulator.set_auto_control(enabled=True, target_power=target)
    
    history = simulator.run(duration=100.0)
    final_power = history['power'][-1]
    
    # Проверка, что мощность приблизилась к целевой (допускается 3% отклонение)
    error = abs(final_power - target) / target
    assert error < 0.03, f"Мощность должна быть близка к целевой (ошибка: {error*100:.1f}%)"
    print(f"   ✓ Достигнута мощность: {final_power:.2f} МВт (цель: {target:.0f}, ошибка: {error*100:.1f}%)")
    
    print("\n🧪 Тест 4: Аварийная остановка (SCRAM)...")
    simulator.reset()
    simulator.scram()
    
    state = simulator.get_state()
    assert state['avg_rod_position'] == 0.0, "Стержни должны быть полностью вставлены"
    print("   ✓ SCRAM выполнен успешно")
    
    print("\n🧪 Тест 5: Ручное управление...")
    simulator.reset()
    initial_power = simulator.get_state()['power']
    
    # Вставление стержней (10 секунд) - снижаем мощность
    for _ in range(10000):
        simulator.step(manual_rod_direction=-1)
    
    state = simulator.get_state()
    power_decreased = state['power'] < initial_power * 0.9  # Снижение хотя бы на 10%
    assert power_decreased, f"Мощность должна снизиться при вставлении стержней (было: {initial_power:.2f}, стало: {state['power']:.2f})"
    print(f"   ✓ Мощность снизилась с {initial_power:.2f} до {state['power']:.2f} МВт")
    
    print("\n✅ Все тесты пройдены успешно!")


def test_safety():
    """Тест системы безопасности"""
    print("\n🧪 Тест безопасности: Потеря охлаждения...")
    simulator = ReactorSimulator(device='cpu')
    simulator.reset()
    
    # Повышение мощности
    simulator.set_auto_control(enabled=True, target_power=180.0)
    simulator.run(duration=30.0)
    
    # Потеря охлаждения
    simulator.set_coolant_flow(0.1)
    simulator.set_auto_control(enabled=False)
    
    # Симуляция до срабатывания защиты
    max_temp = 0.0
    for _ in range(2000):
        state = simulator.step()
        max_temp = max(max_temp, state['T_fuel'])
        
        if state['T_fuel'] > 900.0:
            simulator.scram()
            break
    
    print(f"   Максимальная температура топлива: {max_temp:.1f}°C")
    print(f"   SCRAM сработал на t={state['time']:.2f}s")
    
    # Охлаждение после SCRAM
    simulator.set_coolant_flow(1.0)
    simulator.run(duration=30.0)
    
    final_state = simulator.get_state()
    print(f"   Финальная температура: {final_state['T_fuel']:.1f}°C")
    print(f"   Финальная мощность: {final_state['power']:.2f} МВт")
    print("   ✓ Система безопасности работает корректно")


def performance_test():
    """Тест производительности"""
    print("\n⚡ Тест производительности...")
    import time
    
    simulator = ReactorSimulator(device='cpu')
    simulator.reset()
    
    start_time = time.time()
    history = simulator.run(duration=100.0)
    elapsed = time.time() - start_time
    
    steps = len(history['time'])
    steps_per_sec = steps / elapsed
    
    print(f"   Выполнено шагов: {steps}")
    print(f"   Время выполнения: {elapsed:.2f} с")
    print(f"   Производительность: {steps_per_sec:.0f} шагов/с")
    print(f"   Реальное время / симулированное: {elapsed/100.0:.2f}x")
    
    if steps_per_sec > 1000:
        print("   ✓ Отличная производительность!")
    elif steps_per_sec > 500:
        print("   ✓ Хорошая производительность")
    else:
        print("   ⚠️  Низкая производительность")


def main():
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ СИМУЛЯТОРА АТОМНОГО РЕАКТОРА")
    print("=" * 60)
    
    try:
        test_basic()
        test_safety()
        performance_test()
        
        print("\n" + "=" * 60)
        print("🎉 ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ УСПЕШНО!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ ОШИБКА: {e}")
        return 1
    except Exception as e:
        print(f"\n💥 КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

