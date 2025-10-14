"""
Утилиты для визуализации результатов симуляции
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_simulation_results(history, title="Симуляция ядерного реактора", save_path=None):
    """
    Построить графики результатов симуляции
    
    Args:
        history: история симуляции из simulator.get_history()
        title: заголовок графика
        save_path: путь для сохранения (если None, показать на экране)
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    time = history['time']
    
    # График 1: Мощность реактора
    ax = axes[0]
    ax.plot(time, history['power'], 'b-', linewidth=2)
    ax.set_ylabel('Мощность (МВт)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time[0], time[-1])
    
    # График 2: Температуры
    ax = axes[1]
    ax.plot(time, history['T_fuel'], 'r-', linewidth=2, label='Топливо')
    ax.plot(time, history['T_coolant'], 'b-', linewidth=2, label='Теплоноситель')
    ax.axhline(y=1200, color='r', linestyle='--', alpha=0.5, label='Предел топлива')
    ax.axhline(y=320, color='b', linestyle='--', alpha=0.5, label='Предел теплоносителя')
    ax.set_ylabel('Температура (°C)', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time[0], time[-1])
    
    # График 3: Реактивность
    ax = axes[2]
    ax.plot(time, history['reactivity'], 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_ylabel('Реактивность (β)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time[0], time[-1])
    
    # График 4: Позиция стержней
    ax = axes[3]
    ax.plot(time, np.array(history['rod_position']) * 100, 'm-', linewidth=2)
    ax.set_ylabel('Позиция стержней (%)', fontsize=11)
    ax.set_xlabel('Время (с)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_phase_space(history, save_path=None):
    """
    Построить фазовое пространство (мощность vs температура)
    
    Args:
        history: история симуляции
        save_path: путь для сохранения
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    power = history['power']
    T_fuel = history['T_fuel']
    time = history['time']
    
    # Цветовая карта по времени
    scatter = ax.scatter(T_fuel, power, c=time, cmap='viridis', 
                        s=20, alpha=0.6, edgecolors='none')
    
    ax.set_xlabel('Температура топлива (°C)', fontsize=12)
    ax.set_ylabel('Мощность (МВт)', fontsize=12)
    ax.set_title('Фазовое пространство: Мощность vs Температура', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Цветовая шкала
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Время (с)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_realtime_dashboard(state):
    """
    Создать панель мониторинга в реальном времени (для использования с анимацией)
    
    Args:
        state: текущее состояние реактора
    """
    # Эта функция предназначена для интеграции с matplotlib.animation
    # или другими инструментами реального времени
    pass


def create_summary_report(history):
    """
    Создать текстовый отчет о симуляции
    
    Args:
        history: история симуляции
    
    Returns:
        str: отчет
    """
    power = np.array(history['power'])
    T_fuel = np.array(history['T_fuel'])
    T_coolant = np.array(history['T_coolant'])
    reactivity = np.array(history['reactivity'])
    
    report = []
    report.append("=" * 60)
    report.append("ОТЧЕТ О СИМУЛЯЦИИ ЯДЕРНОГО РЕАКТОРА")
    report.append("=" * 60)
    report.append("")
    
    report.append(f"Длительность симуляции: {history['time'][-1]:.2f} с")
    report.append("")
    
    report.append("МОЩНОСТЬ:")
    report.append(f"  Начальная:  {power[0]:.2f} МВт")
    report.append(f"  Конечная:   {power[-1]:.2f} МВт")
    report.append(f"  Максимальная: {np.max(power):.2f} МВт")
    report.append(f"  Минимальная:  {np.min(power):.2f} МВт")
    report.append(f"  Средняя:    {np.mean(power):.2f} МВт")
    report.append("")
    
    report.append("ТЕМПЕРАТУРА ТОПЛИВА:")
    report.append(f"  Начальная:  {T_fuel[0]:.1f} °C")
    report.append(f"  Конечная:   {T_fuel[-1]:.1f} °C")
    report.append(f"  Максимальная: {np.max(T_fuel):.1f} °C")
    report.append(f"  Минимальная:  {np.min(T_fuel):.1f} °C")
    report.append("")
    
    report.append("ТЕМПЕРАТУРА ТЕПЛОНОСИТЕЛЯ:")
    report.append(f"  Начальная:  {T_coolant[0]:.1f} °C")
    report.append(f"  Конечная:   {T_coolant[-1]:.1f} °C")
    report.append(f"  Максимальная: {np.max(T_coolant):.1f} °C")
    report.append(f"  Минимальная:  {np.min(T_coolant):.1f} °C")
    report.append("")
    
    report.append("РЕАКТИВНОСТЬ:")
    report.append(f"  Начальная:  {reactivity[0]:.4f} β")
    report.append(f"  Конечная:   {reactivity[-1]:.4f} β")
    report.append(f"  Максимальная: {np.max(reactivity):.4f} β")
    report.append(f"  Минимальная:  {np.min(reactivity):.4f} β")
    report.append("")
    
    # Проверка безопасности
    safety_ok = True
    report.append("БЕЗОПАСНОСТЬ:")
    if np.any(T_fuel > 1200):
        report.append("  ⚠️  ПРЕДУПРЕЖДЕНИЕ: Превышен предел температуры топлива!")
        safety_ok = False
    if np.any(T_coolant > 320):
        report.append("  ⚠️  ПРЕДУПРЕЖДЕНИЕ: Превышен предел температуры теплоносителя!")
        safety_ok = False
    
    if safety_ok:
        report.append("  ✓ Все параметры в пределах нормы")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)

