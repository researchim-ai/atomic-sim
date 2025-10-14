"""
Пример ручного управления реактором
Демонстрирует ручное управление управляющими стержнями
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor import ReactorSimulator
from reactor.visualization import plot_simulation_results
import numpy as np


def main():
    print("=" * 60)
    print("СИМУЛЯТОР ЯДЕРНОГО РЕАКТОРА")
    print("Ручное управление")
    print("=" * 60)
    
    # Создание симулятора
    simulator = ReactorSimulator(device='cpu')
    simulator.reset()
    
    print("\n1. Начальное состояние:")
    simulator.print_state()
    
    # Фаза 1: Извлечение стержней (увеличение мощности)
    print("\n2. Фаза 1: Извлечение стержней (30 секунд)...")
    
    simulator.start_recording()
    
    duration = 30.0
    steps = int(duration / simulator.dt)
    
    for i in range(steps):
        simulator.step(manual_rod_direction=1)  # Извлекать стержни
        
        if i % int(5.0 / simulator.dt) == 0:  # Каждые 5 секунд
            state = simulator.get_state()
            print(f"   t={state['time']:.1f}s, "
                  f"P={state['power']:.2f} МВт, "
                  f"Стержни: {state['avg_rod_position']*100:.1f}%")
    
    # Фаза 2: Удержание стержней
    print("\n3. Фаза 2: Удержание стержней (20 секунд)...")
    
    duration = 20.0
    steps = int(duration / simulator.dt)
    
    for i in range(steps):
        simulator.step(manual_rod_direction=0)  # Удерживать
        
        if i % int(5.0 / simulator.dt) == 0:
            state = simulator.get_state()
            print(f"   t={state['time']:.1f}s, "
                  f"P={state['power']:.2f} МВт, "
                  f"T_fuel={state['T_fuel']:.1f}°C")
    
    # Фаза 3: Вставление стержней (снижение мощности)
    print("\n4. Фаза 3: Вставление стержней (30 секунд)...")
    
    duration = 30.0
    steps = int(duration / simulator.dt)
    
    for i in range(steps):
        simulator.step(manual_rod_direction=-1)  # Вставлять стержни
        
        if i % int(5.0 / simulator.dt) == 0:
            state = simulator.get_state()
            print(f"   t={state['time']:.1f}s, "
                  f"P={state['power']:.2f} МВт, "
                  f"Стержни: {state['avg_rod_position']*100:.1f}%")
    
    # Фаза 4: Стабилизация
    print("\n5. Фаза 4: Стабилизация (20 секунд)...")
    
    duration = 20.0
    steps = int(duration / simulator.dt)
    
    for i in range(steps):
        simulator.step(manual_rod_direction=0)
        
        if i % int(5.0 / simulator.dt) == 0:
            state = simulator.get_state()
            print(f"   t={state['time']:.1f}s, "
                  f"P={state['power']:.2f} МВт")
    
    simulator.stop_recording()
    
    # Получение истории
    history = simulator.get_history()
    
    # Визуализация
    print("\n6. Построение графиков...")
    plot_simulation_results(history,
                           title="Ручное управление реактором",
                           save_path="manual_control.png")
    
    # Финальное состояние
    print("\n7. Финальное состояние:")
    simulator.print_state()
    
    print("\nГотово! График сохранен в 'manual_control.png'")


if __name__ == "__main__":
    main()

