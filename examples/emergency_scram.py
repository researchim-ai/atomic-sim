"""
Пример аварийной ситуации и SCRAM
Демонстрирует поведение реактора при потере охлаждения и аварийной остановке
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor import ReactorSimulator
from reactor.visualization import plot_simulation_results


def main():
    print("=" * 60)
    print("СИМУЛЯТОР АТОМНОГО РЕАКТОРА")
    print("Аварийная ситуация: Потеря охлаждения и SCRAM")
    print("=" * 60)
    
    # Создание симулятора
    simulator = ReactorSimulator(device='cpu')
    simulator.reset()
    
    # Установка повышенной мощности
    print("\n1. Увеличение мощности реактора...")
    simulator.set_auto_control(enabled=True, target_power=200.0)
    
    history1 = simulator.run(
        duration=50.0,
        callback=lambda state: print(f"   t={state['time']:.1f}s, "
                                    f"P={state['power']:.2f} МВт, "
                                    f"T_fuel={state['T_fuel']:.1f}°C"),
        callback_interval=10.0
    )
    
    print(f"\n   Достигнута мощность: {simulator.get_state()['power']:.2f} МВт")
    
    # Симуляция потери охлаждения
    print("\n2. АВАРИЯ: Потеря охлаждения!")
    print("   Расход теплоносителя снижен до 20%...")
    
    simulator.set_coolant_flow(0.2)  # Снижение до 20% от номинального
    simulator.set_auto_control(enabled=False)  # Отключение авторегулирования
    
    # Мониторинг температуры
    def monitor_callback(state):
        print(f"   t={state['time']:.1f}s, "
              f"T_fuel={state['T_fuel']:.1f}°C, "
              f"T_coolant={state['T_coolant']:.1f}°C")
        
        # Ручной SCRAM при превышении температуры топлива
        if state['T_fuel'] > 900.0 and not hasattr(monitor_callback, 'scrammed'):
            print("\n   >>> КРИТИЧЕСКАЯ ТЕМПЕРАТУРА! <<<")
            print("   >>> ВЫПОЛНЕНИЕ АВАРИЙНОГО SCRAM <<<")
            simulator.scram()
            monitor_callback.scrammed = True
    
    history2 = simulator.run(
        duration=50.0,
        callback=monitor_callback,
        callback_interval=5.0
    )
    
    # Восстановление охлаждения после SCRAM
    print("\n3. Восстановление охлаждения...")
    simulator.set_coolant_flow(1.0)  # Восстановление номинального расхода
    
    history3 = simulator.run(
        duration=100.0,
        callback=lambda state: print(f"   t={state['time']:.1f}s, "
                                    f"P={state['power']:.2f} МВт, "
                                    f"T_fuel={state['T_fuel']:.1f}°C"),
        callback_interval=20.0
    )
    
    # Объединение историй
    import numpy as np
    combined_history = {
        'time': np.concatenate([history1['time'], history2['time'], history3['time']]),
        'power': np.concatenate([history1['power'], history2['power'], history3['power']]),
        'reactivity': np.concatenate([history1['reactivity'], history2['reactivity'], history3['reactivity']]),
        'T_fuel': np.concatenate([history1['T_fuel'], history2['T_fuel'], history3['T_fuel']]),
        'T_coolant': np.concatenate([history1['T_coolant'], history2['T_coolant'], history3['T_coolant']]),
        'rod_position': np.concatenate([history1['rod_position'], history2['rod_position'], history3['rod_position']]),
    }
    
    # Визуализация
    print("\n4. Построение графиков...")
    plot_simulation_results(combined_history,
                           title="Аварийная ситуация: Потеря охлаждения и SCRAM",
                           save_path="emergency_scram.png")
    
    # Финальное состояние
    print("\n5. Финальное состояние после аварии:")
    simulator.print_state()
    
    print("\nГотово! График сохранен в 'emergency_scram.png'")


if __name__ == "__main__":
    main()

