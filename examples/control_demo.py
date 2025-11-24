import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor import ReactorSimulator1D

def run_control_demo():
    print("=== Запуск Демонстрации Системы Управления (Rods + Boron) ===")
    
    sim = ReactorSimulator1D(num_nodes=50)
    
    # Настройка контроллера
    # Включаем контроль мощности (стержни)
    sim.control.auto_power = True
    sim.control.set_target_power(3000.0) # Target 3000 MW
    
    # Включаем контроль бора (для удержания стержней в оптимальной зоне)
    # Оптимальная зона: rod_ref_pos = 0.2 (20% inserted)
    sim.control.auto_boron = True
    sim.control.rod_ref_pos = 0.2
    
    # Начальное состояние: 
    # Rods at 0.2 (Ideal)
    # Power starts at equilibrium (~3000 MW)
    
    print("Система инициализирована.")
    print(f"Target Power: {sim.control.target_power} MW")
    print(f"Target Rod Pos: {sim.control.rod_ref_pos}")
    
    times = []
    powers = []
    rod_pos = []
    boron = []
    aos = []
    
    # 1. Load Follow Maneuver (Снижение мощности до 50%)
    print("Этап 1: Плавное снижение мощности до 50% (1500 MW)...")
    
    # Плавное изменение уставки (Ramp)
    target_power = 3000.0
    
    # Симулируем 200 секунд (шаг 0.01 сек, итого 20000 шагов)
    # Важно: малый шаг нужен для стабильности нейтроники
    steps = 20000
    dt = 0.01
    
    for i in range(steps):
        # Каждые 0.01 сек
        if i < 5000: # 50 sec ramp
            target_power -= (1500.0 / 5000.0) 
        else:
            target_power = 1500.0
            
        sim.control.set_target_power(target_power)
            
        state = sim.step(dt=dt)
        
        # Пишем в лог реже
        if i % 100 == 0:
            times.append(state['time'])
            powers.append(state['total_power'])
            rod_pos.append(state['rod_position'])
            boron.append(state['boron_ppm'])
            aos.append(state['axial_offset'])
        
    print(f"Power: {state['total_power']:.1f} MW (Target: 1500)")
    print(f"Rods: {state['rod_position']:.3f} (Ref: 0.2)")
    print(f"Boron: {state['boron_ppm']:.1f} ppm")
    
    # Что должно произойти:
    # 1. Стержни упадут внутрь, чтобы снизить мощность.
    # 2. Они окажутся глубоко (например, 0.6).
    # 3. Система бора увидит, что rods > 0.2.
    # 4. Она начнет добавлять бор (borate), чтобы скомпенсировать реактивность.
    # 5. По мере добавления бора, стержни начнут выходить обратно к 0.2, чтобы удержать мощность 1500.
    
    # 2. Возврат к 100% мощности
    print("\nЭтап 2: Плавный возврат к 100% (3000 MW)...")
    
    # Плавный подъем (Ramp)
    for i in range(steps):
        if i < 5000:
             target_power += (1500.0 / 5000.0)
        else:
             target_power = 3000.0
             
        sim.control.set_target_power(target_power)
    
        state = sim.step(dt=dt)
        
        if i % 100 == 0:
            times.append(state['time'])
            powers.append(state['total_power'])
            rod_pos.append(state['rod_position'])
            boron.append(state['boron_ppm'])
            aos.append(state['axial_offset'])
        times.append(state['time'])
        powers.append(state['total_power'])
        rod_pos.append(state['rod_position'])
        boron.append(state['boron_ppm'])
        aos.append(state['axial_offset'])
        
    print(f"Power: {state['total_power']:.1f} MW")
    print(f"Rods: {state['rod_position']:.3f}")
    print(f"Boron: {state['boron_ppm']:.1f} ppm")

    # === Визуализация ===
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Power
    axes[0, 0].plot(times, powers, 'b-', label='Actual')
    axes[0, 0].axhline(3000, color='g', linestyle=':', label='100%')
    axes[0, 0].axhline(1500, color='r', linestyle=':', label='50%')
    axes[0, 0].set_title('Reactor Power (MW)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Rod Position
    axes[0, 1].plot(times, rod_pos, 'k-', label='Rod Position')
    axes[0, 1].axhline(0.2, color='g', linestyle='--', label='Target (0.2)')
    axes[0, 1].invert_yaxis() # 0 is top
    axes[0, 1].set_title('Control Rod Position (0=Out, 1=In)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Boron
    axes[1, 0].plot(times, boron, 'm-', label='Boron (ppm)')
    axes[1, 0].set_title('Boron Concentration')
    axes[1, 0].grid(True)
    
    # Axial Offset
    axes[1, 1].plot(times, aos, 'r-', label='Axial Offset')
    axes[1, 1].axhline(0, color='k', linestyle='--')
    axes[1, 1].set_title('Axial Offset (Ao)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('control_demo_result.png')
    print("Результат сохранен в control_demo_result.png")

if __name__ == "__main__":
    run_control_demo()

