"""
Пример симуляции отравления ксеноном
Демонстрирует эффект Xe-135 на реактивность после изменения мощности
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor.simulator_advanced import ReactorSimulatorAdvanced
from reactor.visualization import plot_simulation_results
import matplotlib.pyplot as plt
import numpy as np


def main():
    print("=" * 60)
    print("СИМУЛЯТОР ЯДЕРНОГО РЕАКТОРА")
    print("Эффект отравления ксеноном")
    print("=" * 60)
    
    # Создание расширенного симулятора с ксеноном
    simulator = ReactorSimulatorAdvanced(enable_xenon=True)
    simulator.reset(equilibrium_xenon=False)  # Начинаем без ксенона
    
    print("\n1. Работа на полной мощности (накопление ксенона)...")
    simulator.set_auto_control(enabled=True, target_power=100.0)
    
    history1 = simulator.run(
        duration=50000.0,  # ~14 часов для установления равновесия Xe
        callback_interval=5000.0,
        show_progress=True
    )
    
    print(f"\n   Равновесная концентрация Xe: {simulator.xenon.Xe.item():.6f}")
    print(f"   Реактивность от Xe: {simulator.xenon.get_state()['Xe_reactivity']:.6f} β")
    
    # Теперь снижаем мощность
    print("\n2. Снижение мощности до 50% (ксеноновая яма)...")
    simulator.set_auto_control(enabled=True, target_power=50.0)
    
    history2 = simulator.run(
        duration=50000.0,  # Еще ~14 часов
        callback_interval=5000.0,
        show_progress=True
    )
    
    print(f"\n   Пиковая концентрация Xe: {max(history2['Xe_concentration']):.6f}")
    print(f"   Минимальная реактивность: {min(history2['reactivity_xenon']):.6f} β")
    
    # Объединение историй
    combined = {}
    for key in history1.keys():
        combined[key] = np.concatenate([history1[key], history2[key]])
    
    # Визуализация
    print("\n3. Построение графиков...")
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle('Эффект отравления ксеноном', fontsize=16, fontweight='bold')
    
    time_hours = combined['time'] / 3600.0  # В часах
    
    # Мощность
    axes[0].plot(time_hours, combined['power'], 'b-', linewidth=2)
    axes[0].set_ylabel('Мощность (МВт)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=history1['time'][-1]/3600.0, color='r', linestyle='--', alpha=0.5, label='Снижение мощности')
    axes[0].legend()
    
    # Концентрации
    axes[1].plot(time_hours, combined['I_concentration'], 'g-', linewidth=2, label='I-135')
    axes[1].plot(time_hours, combined['Xe_concentration'], 'r-', linewidth=2, label='Xe-135')
    axes[1].set_ylabel('Концентрация', fontsize=11)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Реактивности
    axes[2].plot(time_hours, combined['reactivity_rods'], label='Стержни', linewidth=2)
    axes[2].plot(time_hours, combined['reactivity_temp'], label='Температура', linewidth=2)
    axes[2].plot(time_hours, combined['reactivity_xenon'], label='Ксенон', linewidth=2)
    axes[2].plot(time_hours, combined['reactivity'], 'k--', label='Полная', linewidth=2)
    axes[2].set_ylabel('Реактивность (β)', fontsize=11)
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.2)
    
    # Позиция стержней
    axes[3].plot(time_hours, np.array(combined['rod_position']) * 100, 'm-', linewidth=2)
    axes[3].set_ylabel('Позиция стержней (%)', fontsize=11)
    axes[3].set_xlabel('Время (часы)', fontsize=11)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xenon_poisoning.png', dpi=150, bbox_inches='tight')
    print("   График сохранен: xenon_poisoning.png")
    
    # Анализ ксеноновой ямы
    xe_before_drop = history1['Xe_concentration'][-1]
    xe_peak = max(history2['Xe_concentration'][:20000])  # Пик в первые ~5 часов
    xe_final = history2['Xe_concentration'][-1]
    
    print("\n4. Анализ ксеноновой ямы:")
    print(f"   Концентрация Xe при 100 МВт (равновесие): {xe_before_drop:.6f}")
    print(f"   Пиковая концентрация Xe после снижения: {xe_peak:.6f}")
    print(f"   Увеличение: {(xe_peak/xe_before_drop - 1)*100:.1f}%")
    print(f"   Конечная концентрация при 50 МВт: {xe_final:.6f}")
    
    print("\n✓ Готово!")


if __name__ == "__main__":
    main()

