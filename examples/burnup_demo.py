import torch
import matplotlib.pyplot as plt
import numpy as np
from reactor import ReactorSimulator1D

def run_burnup_demo():
    print("=== Запуск Демонстрации Топливного Цикла (Burnup) ===")
    
    # Инициализация
    sim = ReactorSimulator1D(num_nodes=20, device='cpu')
    
    # Начальный разгон до 100% мощности
    print("Выход на мощность...")
    # Убираем бор, чтобы запуститься
    sim.boron_concentration = 1000.0 # Начальное высокое значение (BOC - Beginning of Cycle)
    sim.rod_position = 0.2 # Почти извлечены
    
    # Сделаем так: сначала дадим системе прийти в равновесие
    # Нам нужно, чтобы реактор был критичен при высоком боре.
    # nu_Sigma_f=0.022, Sigma_a=0.015. Запас 0.007.
    # Бор должен компенсировать это. 1400 ppm * 5e-6 = 0.007.
    sim.boron_concentration = 1300.0 
    
    # Прогон кинетики для стабилизации
    for _ in range(500):
        sim.step(dt=0.01)
        
    print(f"Начальная мощность: {sim.get_state()['total_power']:.2f} MW")
    print(f"Начальный бор: {sim.boron_concentration:.1f} ppm")
    
    # История
    history = {
        'days': [],
        'u235': [],
        'pu239': [],
        'boron': [],
        'power': []
    }
    
    # Цикл кампании (например, 18 месяцев = 540 дней)
    # Мы ускорим процесс, делая шаги по 10 дней
    total_days = 540
    step_days = 10
    current_day = 0
    
    print(f"Начало кампании ({total_days} дней)...")
    
    while current_day < total_days:
        # 1. Шаг выгорания (Fast Forward)
        # 24 часа * step_days
        sim.burnup_step(time_hours=24 * step_days)
        current_day += step_days
        
        # 2. Компенсация реактивности (Criticality Search)
        # Выгорание уменьшает k_eff. Нам нужно уменьшить бор, чтобы вернуть k_eff=1.
        # Мы можем сделать это, запустив кинетику и позволив контроллеру бора сделать работу?
        # Но контроллер бора медленный (0.1 ppm/s).
        # Давайте сделаем "быструю подстройку".
        
        # Оценка потери реактивности:
        state = sim.get_state()
        fuel_rho_change = state['fuel_reactivity_change'] # Это дельта сечений
        # Если сечение генерации падает, а поглощения растет -> реактивность падает.
        # delta_rho ~ (d_nu_f - d_a)
        # Нам нужно уменьшить бор: d_boron_abs = - d_fuel_rho
        # sigma_boron * d_ppm = - d_fuel_rho
        # d_ppm = - d_fuel_rho / sigma_boron
        
        # Однако, точный расчет сложен. Проще позволить авто-контроллеру поработать чуть дольше.
        # Ускорим контроллер бора для демо
        sim.control.boron_rate = 10.0 # Очень быстро меняем бор
        
        # Запускаем кинетику на 10 секунд, чтобы контроллер выровнял мощность
        target_power = 3000.0
        sim.control.target_power = target_power
        sim.control.auto_power = True
        sim.control.auto_boron = True
        
        for _ in range(100): # 100 * 0.1s = 10s (увеличим dt для скорости)
            sim.step(dt=0.1) # Больший шаг для скорости
            
        # Запись данных
        state = sim.get_state()
        history['days'].append(current_day)
        history['u235'].append(state['N_U235_avg'])
        history['pu239'].append(state['N_Pu239_avg'])
        history['boron'].append(state['boron_ppm'])
        history['power'].append(state['total_power'])
        
        if current_day % 60 == 0:
            print(f"Day {current_day}: Boron {state['boron_ppm']:.1f} ppm, U-235 {state['N_U235_avg']:.2e}")

    # Визуализация
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    ax1.plot(history['days'], history['u235'], label='U-235', color='blue')
    ax1.plot(history['days'], history['pu239'], label='Pu-239', color='red')
    ax1.set_ylabel('Концентрация (ядер/см³)')
    ax1.set_title('Изотопный состав топлива')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['days'], history['boron'], color='green', label='Boron Concentration')
    ax2.set_ylabel('Бор (ppm)')
    ax2.set_title('Кривая выгорания (Boron Letdown Curve)')
    ax2.legend()
    ax2.grid(True)
    
    ax3.plot(history['days'], history['power'], color='orange', label='Power')
    ax3.set_ylabel('Мощность (МВт)')
    ax3.set_xlabel('Время кампании (дни)')
    ax3.set_title('История мощности')
    ax3.grid(True)
    ax3.set_ylim(0, 3500)
    
    plt.tight_layout()
    plt.savefig('burnup_demo_result.png')
    print("Результат сохранен в burnup_demo_result.png")

if __name__ == "__main__":
    run_burnup_demo()

