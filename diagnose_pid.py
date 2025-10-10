"""Детальная диагностика ПИД-регулятора"""
from reactor import ReactorSimulator
import numpy as np

simulator = ReactorSimulator()
simulator.reset()
simulator.set_auto_control(enabled=True, target_power=90.0)

print("=" * 90)
print("ДЕТАЛЬНАЯ ДИАГНОСТИКА ПИД-РЕГУЛЯТОРА")
print("=" * 90)

print("\nПараметры ПИД:")
print(f"  Kp = {simulator.control.Kp.item():.6f}")
print(f"  Ki = {simulator.control.Ki.item():.6f}")
print(f"  Kd = {simulator.control.Kd.item():.6f}")
print(f"  Порог = {0.5}")
print(f"  Скорость стержней = {simulator.control.rod_speed.item():.3f}")

print("\nМониторинг работы (первые 50 секунд):")
print("Время | Мощн | Цель | Ошибка | P-член | I-член | D-член | Сигнал | Напр | Стержни")
print("-" * 90)

simulator.start_recording()

last_print_time = 0
for i in range(50000):  # 50 секунд
    state = simulator.step()
    
    if state['time'] - last_print_time >= 2.0:  # Каждые 2 секунды
        last_print_time = state['time']
        
        error = simulator.control.target_power.item() - state['power']
        p_term = simulator.control.Kp.item() * error
        i_term = simulator.control.Ki.item() * simulator.control.integral_error.item()
        d_term = simulator.control.Kd.item() * (error - simulator.control.previous_error.item()) / simulator.dt
        
        control_signal = p_term + i_term + d_term
        
        direction = "→" if control_signal > 0.5 else "←" if control_signal < -0.5 else "•"
        
        print(f"{state['time']:5.1f} | {state['power']:5.1f} | {simulator.control.target_power.item():4.0f} | "
              f"{error:6.2f} | {p_term:6.3f} | {i_term:6.3f} | {d_term:6.3f} | "
              f"{control_signal:6.3f} | {direction} | {state['avg_rod_position']*100:5.1f}%")

simulator.stop_recording()
history = simulator.get_history()

print("\n" + "=" * 90)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 90)

final_power = history['power'][-1]
target = 90.0
error_pct = abs(final_power - target) / target * 100

print(f"\nКонечная мощность: {final_power:.2f} МВт")
print(f"Целевая мощность: {target:.2f} МВт")
print(f"Ошибка: {error_pct:.1f}%")

# Анализ движения стержней
rod_changes = 0
for i in range(1, len(history['rod_position'])):
    if abs(history['rod_position'][i] - history['rod_position'][i-1]) > 1e-6:
        rod_changes += 1

print(f"\nСтержни двигались {rod_changes} раз из {len(history['time'])} шагов ({rod_changes/len(history['time'])*100:.1f}%)")

# Анализ установившейся ошибки
steady_state_start = int(len(history['power']) * 0.8)  # Последние 20%
steady_errors = [abs(history['power'][i] - target) for i in range(steady_state_start, len(history['power']))]
avg_steady_error = np.mean(steady_errors)

print(f"Средняя ошибка в установившемся режиме: {avg_steady_error:.2f} МВт ({avg_steady_error/target*100:.1f}%)")

print("\n" + "=" * 90)
print("ДИАГНОЗ ПРОБЛЕМЫ:")
print("=" * 90)

if error_pct > 3:
    print("\n1. ОШИБКА СЛИШКОМ БОЛЬШАЯ")
    
    if abs(history['rod_position'][-1] - history['rod_position'][-100]) < 0.001:
        print("   → Стержни НЕ ДВИГАЮТСЯ в конце!")
        print("   → Возможно, управляющий сигнал меньше порога")
        
        final_error = target - final_power
        final_signal = (simulator.control.Kp.item() * final_error + 
                       simulator.control.Ki.item() * simulator.control.integral_error.item())
        
        print(f"\n   Финальная ошибка: {final_error:.2f} МВт")
        print(f"   Финальный сигнал: {final_signal:.3f}")
        print(f"   Порог: ±0.5")
        
        if abs(final_signal) < 0.5:
            print("\n   ✗ ПРОБЛЕМА: Сигнал меньше порога!")
            print("   РЕШЕНИЕ: Уменьшить порог ИЛИ увеличить коэффициенты ПИД")
        else:
            print("\n   ? Сигнал больше порога, но стержни не двигаются - странно!")
    else:
        print("   → Стержни все еще двигаются")
        print("   → Возможно, нужно больше времени для установления")

