import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor import ReactorSimulator1D

def run_demo():
    print("=== Запуск 1D Симулятора (Пространственная кинетика) ===")
    
    # Создаем симулятор (50 узлов)
    sim = ReactorSimulator1D(num_nodes=50)
    print("Инициализация выполнена.")
    
    times = []
    powers = []
    axial_offsets = []
    
    # 1. Стабилизация (2 секунды)
    print("Этап 1: Стабилизация...")
    for _ in range(200):
        state = sim.step(dt=0.01)
        times.append(state['time'])
        powers.append(state['total_power'])
        axial_offsets.append(state['axial_offset'])
    
    profile_initial = sim.neutronics.phi.cpu().numpy().copy()
    temp_initial = sim.thermal.T_fuel.cpu().numpy().copy()
    
    # 2. Ввод стержней на 50% (за 5 секунд)
    print("Этап 2: Ввод стержней на 50%...")
    sim.set_rod_speed(0.1) # 10% в секунду
    for _ in range(500):
        state = sim.step(dt=0.01)
        times.append(state['time'])
        powers.append(state['total_power'])
        axial_offsets.append(state['axial_offset'])
        
    sim.set_rod_speed(0.0)
    
    # 3. Удержание (3 секунды)
    print("Этап 3: Удержание...")
    for _ in range(300):
        state = sim.step(dt=0.01)
        times.append(state['time'])
        powers.append(state['total_power'])
        axial_offsets.append(state['axial_offset'])
        
    profile_final = sim.neutronics.phi.cpu().numpy().copy()
    temp_final = sim.thermal.T_fuel.cpu().numpy().copy()
    
    print("Симуляция завершена.")
    
    # === Визуализация ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # График 1: Общая мощность
    axes[0, 0].plot(times, powers, 'b-', label='Total Power (MW)')
    axes[0, 0].set_title('Мощность реактора')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Power (MW)')
    axes[0, 0].grid(True)
    
    # График 2: Аксиальный оффсет (перекос поля)
    axes[0, 1].plot(times, axial_offsets, 'r-', label='Axial Offset')
    axes[0, 1].set_title('Аксиальный Оффсет (Ao)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Ao')
    axes[0, 1].axhline(0, color='k', linestyle='--')
    axes[0, 1].grid(True)
    
    # График 3: Профиль нейтронного потока
    z = np.linspace(0, 1, 50) # 0 = Top, 1 = Bottom
    axes[1, 0].plot(profile_initial, z, 'b--', label='Initial (Rods Out)')
    axes[1, 0].plot(profile_final, z, 'r-', label='Final (Rods 50%)')
    axes[1, 0].invert_yaxis() # Top is 0
    axes[1, 0].set_title('Профиль нейтронного потока')
    axes[1, 0].set_xlabel('Flux (n/cm^2/s)')
    axes[1, 0].set_ylabel('Position (0=Top)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # График 4: Профиль температуры топлива
    axes[1, 1].plot(temp_initial, z, 'b--', label='Initial')
    axes[1, 1].plot(temp_final, z, 'r-', label='Final')
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_title('Температура топлива')
    axes[1, 1].set_xlabel('Temperature (C)')
    axes[1, 1].set_ylabel('Position (0=Top)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('spatial_kinetics_demo.png')
    print("Результат сохранен в spatial_kinetics_demo.png")

if __name__ == "__main__":
    run_demo()

