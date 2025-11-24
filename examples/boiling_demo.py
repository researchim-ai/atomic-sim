import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor import ReactorSimulator1D

def run_boiling_demo():
    print("=== Запуск Демонстрации Кипения (Void Fraction Physics) ===")
    
    sim = ReactorSimulator1D(num_nodes=50)
    
    # Начальное состояние
    times = []
    voids = [] 
    powers = []
    max_temps = []
    
    # 1. Нормальная работа (2 сек)
    print("Этап 1: Нормальная работа...")
    for _ in range(200):
        state = sim.step(dt=0.01)
        times.append(state['time'])
        voids.append(state['max_void_fraction'])
        powers.append(state['total_power'])
        max_temps.append(state['max_fuel_temp'])
        
    # 2. Снижение расхода теплоносителя (Loss of Flow)
    # Это должно вызвать кипение
    print("Этап 2: Снижение расхода до 20% (Авария насоса)...")
    # В Simulator1D нужно пробросить flow_factor
    # Сейчас мы это сделаем хаком или добавим метод, но пока 
    # Simulator1D hardcodes thermal.step parameters inside.
    # Let's modify thermal directly for this demo
    
    target_flow = 0.2
    current_flow = 1.0
    
    for _ in range(500): # 5 seconds
        # Smooth ramp down
        current_flow = 0.99 * current_flow + 0.01 * target_flow
        
        # Manual step to inject flow
        # Note: Simulator1D.step doesn't take flow yet. 
        # We rely on the fact we can touch sim.thermal
        sim.thermal.flow_factor = current_flow # This will be overwritten in step usually?
        # Wait, simulator_1d.py calls thermal.step(dt, power). thermal.step resets flow_factor if passed?
        # Looking at thermal_1d.py: step(dt, power, flow_factor=1.0).
        # Simulator1D step calls self.thermal.step(dt, power_profile_MW). Default flow=1.0.
        # So we need to PATCH Simulator1D or add method.
        # Let's monkey-patch for demo speed or update simulator class.
        
        # Update simulator class is better but for demo script:
        # We can call internal steps manually or subclass.
        # Actually, let's just modify the flow inside step call by overriding the method? No.
        
        # Let's use the updated Simulator1D feature (we need to add set_flow method there).
        # Assuming we will add it, let's pretend it exists or patch.
        # For now: manually call components.
        
        # Manual Step Loop:
        # 1. Rods
        sim.rod_position = 0.0 # Keep rods out
        
        # 2. Feedback
        reactivity_feedback = sim.thermal.compute_reactivity_feedback()
        xenon_absorption = sim.poisoning.get_absorption_cross_section()
        
        # 3. Neutronics
        sim.neutronics.update_cross_sections(sim.rod_position, reactivity_feedback, xenon_absorption)
        avg_flux = sim.neutronics.step(dt=0.01)
        
        # 4. Power conversion
        flux_profile = sim.neutronics.phi
        conversion = 60.0 / 1e13 
        power_profile_MW = flux_profile * conversion
        
        # 5. Thermal (WITH LOW FLOW)
        sim.thermal.step(0.01, power_profile_MW, flow_factor=current_flow)
        sim.time += 0.01
        
        # Record
        state = sim.get_state()
        times.append(state['time'])
        voids.append(state['max_void_fraction'])
        powers.append(state['total_power'])
        max_temps.append(state['max_fuel_temp'])

    print(f"Максимальное паросодержание: {max(voids)*100:.1f}%")
    print("Этап 3: Аварийная защита (SCRAM)...")
    
    # SCRAM
    sim.rod_position = 1.0
    
    for _ in range(300):
        # Manual loop again
        reactivity_feedback = sim.thermal.compute_reactivity_feedback()
        xenon_absorption = sim.poisoning.get_absorption_cross_section()
        
        sim.neutronics.update_cross_sections(1.0, reactivity_feedback, xenon_absorption) # Rods IN
        sim.neutronics.step(dt=0.01)
        
        flux_profile = sim.neutronics.phi
        power_profile_MW = flux_profile * conversion
        
        sim.thermal.step(0.01, power_profile_MW, flow_factor=current_flow)
        sim.time += 0.01
        
        state = sim.get_state()
        times.append(state['time'])
        voids.append(state['max_void_fraction'])
        powers.append(state['total_power'])
        max_temps.append(state['max_fuel_temp'])
        
    # === Plotting ===
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Void Fraction', color=color)
    ax1.plot(times, voids, color=color, label='Max Void Fraction')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Power (MW)', color=color)  
    ax2.plot(times, powers, color=color, linestyle='--', label='Power')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('Демонстрация Кипения: Loss of Flow Accident')
    plt.savefig('boiling_demo_result.png')
    print("График сохранен в boiling_demo_result.png")

if __name__ == "__main__":
    run_boiling_demo()

