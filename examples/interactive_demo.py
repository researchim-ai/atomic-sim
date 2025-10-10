"""
Интерактивная демонстрация симулятора
Позволяет управлять реактором в реальном времени
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reactor import ReactorSimulator
import time


def print_dashboard(state):
    """Вывести панель состояния реактора"""
    # Очистка экрана (кросс-платформенно)
    print("\033[2J\033[H", end='')
    
    print("=" * 70)
    print(" " * 20 + "🔬 СИМУЛЯТОР АТОМНОГО РЕАКТОРА 🔬")
    print("=" * 70)
    print()
    
    # Мощность
    power = state['power']
    power_bar = "█" * int(power / 5) + "░" * (40 - int(power / 5))
    print(f"МОЩНОСТЬ:        [{power_bar}] {power:6.2f} МВт")
    print()
    
    # Температуры
    T_fuel = state['T_fuel']
    T_coolant = state['T_coolant']
    
    fuel_danger = "🔴" if T_fuel > 900 else "🟡" if T_fuel > 700 else "🟢"
    coolant_danger = "🔴" if T_coolant > 310 else "🟡" if T_coolant > 300 else "🟢"
    
    print(f"ТЕМПЕРАТУРЫ:")
    print(f"  Топливо:       {fuel_danger} {T_fuel:6.1f} °C  (предел: 1200°C)")
    print(f"  Теплоноситель: {coolant_danger} {T_coolant:6.1f} °C  (предел: 320°C)")
    print()
    
    # Управляющие стержни
    rod_pos = state['avg_rod_position'] * 100
    rod_bar = "▓" * int(rod_pos / 2.5) + "░" * (40 - int(rod_pos / 2.5))
    print(f"СТЕРЖНИ:         [{rod_bar}] {rod_pos:5.1f} %")
    print()
    
    # Реактивность
    reactivity = state['rod_reactivity']
    react_status = "+" if reactivity > 0 else " "
    print(f"РЕАКТИВНОСТЬ:    {react_status}{reactivity:+.4f} β")
    print()
    
    # Безопасность
    safety = "✅ НОРМА" if state['safe'] else "⚠️  АВАРИЯ"
    print(f"БЕЗОПАСНОСТЬ:    {safety}")
    print()
    
    # Время
    print(f"ВРЕМЯ:           {state['time']:.1f} с")
    print()
    
    print("-" * 70)
    print("УПРАВЛЕНИЕ:")
    print("  [W] - Извлечь стержни (увеличить мощность)")
    print("  [S] - Вставить стержни (снизить мощность)")
    print("  [SPACE] - Удерживать текущую позицию")
    print("  [A] - Включить автоматическое управление (150 МВт)")
    print("  [E] - АВАРИЙНАЯ ОСТАНОВКА (SCRAM)")
    print("  [Q] - Выход")
    print("=" * 70)


def interactive_mode():
    """Интерактивный режим управления"""
    print("Запуск интерактивного режима...")
    print("(Для корректной работы требуется терминал с поддержкой ANSI)")
    time.sleep(1)
    
    simulator = ReactorSimulator(device='cpu')
    simulator.reset()
    
    # Настройка для неблокирующего ввода
    try:
        import select
        import sys
        import tty
        import termios
        
        # Сохранить оригинальные настройки терминала
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            running = True
            manual_direction = 0
            
            while running:
                # Проверка ввода (неблокирующая)
                if select.select([sys.stdin], [], [], 0)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    if key == 'q':
                        running = False
                    elif key == 'w':
                        manual_direction = 1  # Извлечь
                        simulator.set_auto_control(enabled=False)
                    elif key == 's':
                        manual_direction = -1  # Вставить
                        simulator.set_auto_control(enabled=False)
                    elif key == ' ':
                        manual_direction = 0  # Удерживать
                        simulator.set_auto_control(enabled=False)
                    elif key == 'a':
                        simulator.set_auto_control(enabled=True, target_power=150.0)
                        manual_direction = None
                    elif key == 'e':
                        simulator.scram()
                        print("\n🚨 ВЫПОЛНЕН АВАРИЙНЫЙ SCRAM! 🚨")
                        time.sleep(2)
                
                # Шаг симуляции
                state = simulator.step(manual_rod_direction=manual_direction)
                
                # Отображение состояния
                print_dashboard(state)
                
                # Задержка (примерно реальное время)
                time.sleep(0.05)
            
        finally:
            # Восстановить настройки терминала
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    except ImportError:
        # Упрощенный режим для Windows или систем без select
        print("\n⚠️  Интерактивный режим недоступен на этой платформе")
        print("Запуск демонстрационного режима...\n")
        
        # Автоматическая демонстрация
        demonstration_mode(simulator)


def demonstration_mode(simulator):
    """Автоматический демонстрационный режим"""
    simulator.reset()
    
    print("Фаза 1: Увеличение мощности...")
    for i in range(500):
        state = simulator.step(manual_rod_direction=1)
        if i % 100 == 0:
            print(f"  t={state['time']:.1f}s, P={state['power']:.2f} МВт")
    
    print("\nФаза 2: Автоматическое управление (150 МВт)...")
    simulator.set_auto_control(enabled=True, target_power=150.0)
    
    for i in range(1000):
        state = simulator.step()
        if i % 200 == 0:
            print(f"  t={state['time']:.1f}s, P={state['power']:.2f} МВт")
    
    print("\nФаза 3: Снижение мощности...")
    simulator.set_auto_control(enabled=False)
    
    for i in range(500):
        state = simulator.step(manual_rod_direction=-1)
        if i % 100 == 0:
            print(f"  t={state['time']:.1f}s, P={state['power']:.2f} МВт")
    
    print("\nДемонстрация завершена!")
    simulator.print_state()


def main():
    print("=" * 70)
    print(" " * 15 + "ИНТЕРАКТИВНАЯ ДЕМОНСТРАЦИЯ СИМУЛЯТОРА")
    print("=" * 70)
    print()
    print("Выберите режим:")
    print("  1 - Интерактивное управление (требуется Unix-терминал)")
    print("  2 - Автоматическая демонстрация")
    print()
    
    try:
        choice = input("Ваш выбор (1/2): ").strip()
    except KeyboardInterrupt:
        print("\nОтменено пользователем")
        return
    
    if choice == '1':
        interactive_mode()
    elif choice == '2':
        simulator = ReactorSimulator(device='cpu')
        demonstration_mode(simulator)
    else:
        print("Неверный выбор")


if __name__ == "__main__":
    main()

