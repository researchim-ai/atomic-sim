"""
Пример использования автоматического управления
Демонстрирует работу ПИД-регулятора для поддержания заданной мощности
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor import ReactorSimulator
from reactor.visualization import plot_simulation_results, create_summary_report
import numpy as np


def main():
    print("=" * 60)
    print("СИМУЛЯТОР ЯДЕРНОГО РЕАКТОРА")
    print("Автоматическое управление мощностью")
    print("=" * 60)
    
    # Создание симулятора
    simulator = ReactorSimulator(device='cpu')
    simulator.reset()
    
    # Включение автоматического управления
    print("\n1. Включение автоматического управления...")
    print("   Целевая мощность: 150 МВт")
    simulator.set_auto_control(enabled=True, target_power=150.0)
    
    simulator.print_state()
    
    # Симуляция с автоматическим управлением
    print("\n2. Симуляция увеличения мощности до 150 МВт...")
    
    history = simulator.run(
        duration=100.0,
        callback=lambda state: print(f"   t={state['time']:.1f}s, "
                                    f"P={state['power']:.2f} МВт (цель: 150.00), "
                                    f"Стержни: {state['avg_rod_position']*100:.1f}%"),
        callback_interval=10.0
    )
    
    # Изменение целевой мощности
    print("\n3. Изменение целевой мощности на 80 МВт...")
    simulator.set_auto_control(enabled=True, target_power=80.0)
    
    history2 = simulator.run(
        duration=100.0,
        callback=lambda state: print(f"   t={state['time']:.1f}s, "
                                    f"P={state['power']:.2f} МВт (цель: 80.00), "
                                    f"Стержни: {state['avg_rod_position']*100:.1f}%"),
        callback_interval=10.0
    )
    
    # Объединение историй
    combined_history = {
        'time': np.concatenate([history['time'], history2['time']]),
        'power': np.concatenate([history['power'], history2['power']]),
        'reactivity': np.concatenate([history['reactivity'], history2['reactivity']]),
        'T_fuel': np.concatenate([history['T_fuel'], history2['T_fuel']]),
        'T_coolant': np.concatenate([history['T_coolant'], history2['T_coolant']]),
        'rod_position': np.concatenate([history['rod_position'], history2['rod_position']]),
    }
    
    # Визуализация
    print("\n4. Построение графиков...")
    plot_simulation_results(combined_history,
                           title="Автоматическое управление: Изменение мощности",
                           save_path="auto_control.png")
    
    # Финальное состояние
    print("\n5. Финальное состояние:")
    simulator.print_state()
    
    print("\nГотово! График сохранен в 'auto_control.png'")


if __name__ == "__main__":
    main()

