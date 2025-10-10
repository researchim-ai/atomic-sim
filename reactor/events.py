"""
Модуль событий и возмущений для симулятора
Реализует различные аварийные и штатные события
"""

import torch


class EventManager:
    """Менеджер событий для симулятора"""
    
    def __init__(self):
        self.events = []
        self.active_events = []
    
    def add_event(self, event):
        """Добавить событие в очередь"""
        self.events.append(event)
        self.events.sort(key=lambda e: e.trigger_time)
    
    def check_events(self, current_time, simulator):
        """
        Проверить и выполнить события
        
        Args:
            current_time: текущее время симуляции
            simulator: экземпляр ReactorSimulator
        """
        # Проверка запланированных событий
        while self.events and self.events[0].trigger_time <= current_time:
            event = self.events.pop(0)
            event.execute(simulator)
            self.active_events.append(event)
        
        # Обновление активных событий
        for event in self.active_events[:]:
            if hasattr(event, 'update'):
                event.update(current_time, simulator)
            
            if hasattr(event, 'duration'):
                if current_time - event.trigger_time >= event.duration:
                    if hasattr(event, 'cleanup'):
                        event.cleanup(simulator)
                    self.active_events.remove(event)


class Event:
    """Базовый класс события"""
    
    def __init__(self, trigger_time):
        """
        Args:
            trigger_time: время срабатывания события (секунды)
        """
        self.trigger_time = trigger_time
    
    def execute(self, simulator):
        """Выполнить событие"""
        raise NotImplementedError


class ReactivityInsertionEvent(Event):
    """Событие: ввод внешней реактивности"""
    
    def __init__(self, trigger_time, reactivity, duration=None):
        """
        Args:
            trigger_time: время срабатывания
            reactivity: вводимая реактивность (в долях β)
            duration: длительность (None = постоянно)
        """
        super().__init__(trigger_time)
        self.reactivity = reactivity
        self.duration = duration
    
    def execute(self, simulator):
        """Ввести реактивность"""
        simulator.control.external_reactivity = torch.tensor(
            self.reactivity, device=simulator.control.device, dtype=simulator.control.dtype
        )
        print(f"⚡ [t={self.trigger_time:.1f}s] Ввод реактивности: {self.reactivity:+.3f} β")
    
    def cleanup(self, simulator):
        """Убрать реактивность после окончания duration"""
        simulator.control.external_reactivity = torch.tensor(
            0.0, device=simulator.control.device, dtype=simulator.control.dtype
        )


class PumpCoastdownEvent(Event):
    """Событие: отказ насоса с экспоненциальным снижением расхода"""
    
    def __init__(self, trigger_time, initial_flow=1.0, final_flow=0.1, time_constant=5.0):
        """
        Args:
            trigger_time: время начала отказа
            initial_flow: начальный расход
            final_flow: финальный расход
            time_constant: постоянная времени (секунды)
        """
        super().__init__(trigger_time)
        self.initial_flow = initial_flow
        self.final_flow = final_flow
        self.time_constant = time_constant
        self.start_time = None
    
    def execute(self, simulator):
        """Начать отказ насоса"""
        self.start_time = self.trigger_time
        print(f"🔴 [t={self.trigger_time:.1f}s] Отказ насоса! Расход падает...")
    
    def update(self, current_time, simulator):
        """Обновить расход"""
        if self.start_time is None:
            return
        
        elapsed = current_time - self.start_time
        
        # Экспоненциальное снижение
        flow = self.final_flow + (self.initial_flow - self.final_flow) * torch.exp(
            torch.tensor(-elapsed / self.time_constant)
        )
        
        simulator.set_coolant_flow(flow.item())


class StuckRodEvent(Event):
    """Событие: заклинивание управляющего стержня"""
    
    def __init__(self, trigger_time, rod_index=0):
        """
        Args:
            trigger_time: время заклинивания
            rod_index: индекс застрявшего стержня
        """
        super().__init__(trigger_time)
        self.rod_index = rod_index
        self.original_position = None
    
    def execute(self, simulator):
        """Зафиксировать стержень"""
        self.original_position = simulator.control.rod_positions[self.rod_index].item()
        print(f"⚠️  [t={self.trigger_time:.1f}s] Стержень #{self.rod_index} заклинило на позиции {self.original_position*100:.1f}%")
    
    def update(self, current_time, simulator):
        """Удерживать стержень на месте"""
        if self.original_position is not None:
            simulator.control.rod_positions[self.rod_index] = self.original_position


class RodEjectionEvent(Event):
    """Событие: выброс управляющего стержня (тяжелая авария)"""
    
    def __init__(self, trigger_time, rod_index=0, ejection_speed=10.0):
        """
        Args:
            trigger_time: время выброса
            rod_index: индекс выбрасываемого стержня
            ejection_speed: скорость выброса (1/с)
        """
        super().__init__(trigger_time)
        self.rod_index = rod_index
        self.ejection_speed = ejection_speed
        self.duration = 0.1  # Выброс занимает 0.1 секунды
    
    def execute(self, simulator):
        """Начать выброс стержня"""
        print(f"🚨 [t={self.trigger_time:.1f}s] АВАРИЯ: Выброс стержня #{self.rod_index}!")
    
    def update(self, current_time, simulator):
        """Извлечь стержень с высокой скоростью"""
        # Мгновенный выброс до максимума
        simulator.control.rod_positions[self.rod_index] = 1.0


class LossOfHeatSinkEvent(Event):
    """Событие: потеря теплосъёма (loss of heat sink)"""
    
    def __init__(self, trigger_time):
        super().__init__(trigger_time)
    
    def execute(self, simulator):
        """Отключить охлаждение"""
        simulator.set_coolant_flow(0.0)
        print(f"🔴 [t={self.trigger_time:.1f}s] Потеря теплосъёма!")


# Примеры предопределенных сценариев

def create_loss_of_cooling_scenario():
    """Создать сценарий потери охлаждения"""
    manager = EventManager()
    manager.add_event(PumpCoastdownEvent(trigger_time=30.0, time_constant=3.0))
    return manager


def create_rod_ejection_scenario():
    """Создать сценарий выброса стержня"""
    manager = EventManager()
    manager.add_event(RodEjectionEvent(trigger_time=20.0, rod_index=0))
    return manager


def create_stuck_rod_scenario():
    """Создать сценарий застрявшего стержня"""
    manager = EventManager()
    manager.add_event(StuckRodEvent(trigger_time=10.0, rod_index=5))
    return manager

