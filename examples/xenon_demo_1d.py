import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor import ReactorSimulator1D

def run_xenon_oscillation():
    print("=== Запуск Демонстрации Ксеноновых Колебаний ===")
    print("Это длительный процесс (часы в реальном времени).")
    print("Мы ускорим его, симулируя длинные промежутки.")
    
    # 1. Инициализация
    sim = ReactorSimulator1D(num_nodes=50)
    print("Симулятор инициализирован.")
    
    times = []
    aos = [] # Axial Offsets
    powers = []
    
    # 2. Создаем возмущение
    # Вводим стержни немного, чтобы сместить поток вниз
    # Это вызовет выгорание ксенона вверху (где поток упал) и накопление внизу
    print("Этап 1: Создание возмущения (Ввод стержней на 20%)...")
    sim.rod_position = 0.2
    
    # Симулируем 1 час реального времени (3600 сек)
    # Шаг 0.1 сек -> 36000 шагов. Это долго для демо.
    # Для демо возьмем большой шаг dt (физика позволяет, если не слишком быстро)
    dt = 0.5 
    duration = 3600.0 * 2 # 2 hours
    steps = int(duration / dt)
    
    print(f"Симуляция {duration/3600:.1f} часов процесса...")
    
    for i in range(steps):
        state = sim.step(dt=dt)
        if i % 100 == 0:
            times.append(state['time'] / 3600.0) # Hours
            aos.append(state['axial_offset'])
            powers.append(state['total_power'])
            
    print(f"Ao после возмущения: {state['axial_offset']:.4f}")
    
    # 3. Убираем возмущение (возвращаем стержни)
    print("Этап 2: Снятие возмущения (Стержни OUT)...")
    sim.rod_position = 0.0
    
    # Теперь должны начаться свободные колебания
    # Симулируем 24 часа
    duration = 3600.0 * 24
    steps = int(duration / dt)
    
    print(f"Симуляция {duration/3600:.1f} часов свободных колебаний...")
    
    for i in range(steps):
        state = sim.step(dt=dt)
        if i % 100 == 0:
            times.append(state['time'] / 3600.0)
            aos.append(state['axial_offset'])
            powers.append(state['total_power'])
            
        # Простейший контроль мощности (чтобы не ушла в бесконечность)
        # В реальности оператор держит мощность постоянной
        # Мы здесь просто сбросим излишек реактивности, если мощность сильно растет
        # Это "идеальный регулятор мощности", который не влияет на форму поля
        # (симуляция "criticality search")
        current_k = sim.neutronics.nu_Sigma_f / sim.neutronics.Sigma_a
        # Если мощность ушла, мы чуть корректируем Sigma_a равномерно
        if state['total_power'] > 3100 or state['total_power'] < 2900:
             # Не будем усложнять демо, просто посмотрим свободную динамику
             pass

    # === Визуализация ===
    plt.figure(figsize=(10, 6))
    plt.plot(times, aos, 'b-', label='Axial Offset')
    plt.title('Ксеноновые Колебания (Xenon Oscillations)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Axial Offset (Ao)')
    plt.grid(True)
    plt.axhline(0, color='k', linestyle='--')
    plt.legend()
    
    plt.savefig('xenon_oscillation_demo.png')
    print("График сохранен в xenon_oscillation_demo.png")

if __name__ == "__main__":
    run_xenon_oscillation()

