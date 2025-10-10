"""
Базовый пример использования симулятора атомного реактора
Демонстрирует основные функции и возможности
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor import ReactorSimulator
from reactor.visualization import plot_simulation_results, create_summary_report


def main():
    print("=" * 60)
    print("СИМУЛЯТОР АТОМНОГО РЕАКТОРА")
    print("Базовый пример")
    print("=" * 60)
    
    # Создание симулятора
    simulator = ReactorSimulator(device='cpu')
    
    print("\n1. Инициализация реактора...")
    simulator.reset()
    simulator.print_state()
    
    # Пример 1: Свободный дрейф (без управления)
    print("\n2. Симуляция свободного дрейфа (30 секунд)...")
    print("   (без активного управления)")
    
    history = simulator.run(
        duration=30.0,
        callback=lambda state: print(f"   t={state['time']:.1f}s, P={state['power']:.2f} МВт, "
                                    f"T_fuel={state['T_fuel']:.1f}°C"),
        callback_interval=5.0
    )
    
    # Визуализация
    print("\n3. Построение графиков...")
    plot_simulation_results(history, 
                           title="Базовая симуляция: Свободный дрейф",
                           save_path="basic_simulation.png")
    
    # Отчет
    print("\n4. Отчет о симуляции:")
    report = create_summary_report(history)
    print(report)
    
    print("\nГотово! График сохранен в 'basic_simulation.png'")


if __name__ == "__main__":
    main()

