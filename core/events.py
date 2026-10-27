"""События с локализацией корня. Дискретный скачок не размазывается большим шагом."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

EventFn = Callable[[float, np.ndarray], float]


@dataclass
class EventSpec:
    name: str
    function: EventFn
    direction: float = 0.0
    terminal: bool = True
    priority: int = 0
    hysteresis: float = 0.0
    armed: bool = True


@dataclass
class EventRecord:
    name: str
    time_s: float
    state: np.ndarray
    priority: int = 0


def bracket_root(
    func: EventFn,
    y_left: np.ndarray,
    y_right: np.ndarray,
    t_left: float,
    t_right: float,
    tol: float = 1e-10,
) -> float:
    """Локализация корня по времени при линейной интерполяции состояния между концами шага."""
    f_left = float(func(t_left, y_left))
    f_right = float(func(t_right, y_right))
    if f_left == 0.0:
        return t_left
    if f_left * f_right > 0.0:
        raise ValueError("Корень события не заключён в интервале")
    a, b = t_left, t_right
    ya, yb = y_left, y_right
    fa = f_left
    for _ in range(60):
        mid = 0.5 * (a + b)
        weight = (mid - a) / (b - a) if b != a else 0.0
        y_mid = (1.0 - weight) * ya + weight * yb
        fm = float(func(mid, y_mid))
        if abs(fm) < tol or abs(b - a) < tol:
            return mid
        if fa * fm <= 0.0:
            b, yb = mid, y_mid
        else:
            a, ya, fa = mid, y_mid, fm
    return 0.5 * (a + b)


def collect_sign_changes(
    events: List[EventSpec], t0: float, y0: np.ndarray, t1: float, y1: np.ndarray
) -> List[EventRecord]:
    found: List[EventRecord] = []
    for event in events:
        f0 = float(event.function(t0, y0))
        f1 = float(event.function(t1, y1))
        crossed = f0 == 0.0 or f0 * f1 < 0.0
        if event.direction > 0.0:
            crossed = f0 < 0.0 <= f1 or (f0 == 0.0 and f1 > 0.0)
        elif event.direction < 0.0:
            crossed = f0 > 0.0 >= f1 or (f0 == 0.0 and f1 < 0.0)
        if crossed and f0 * f1 <= 0.0:
            t_event = bracket_root(event.function, y0, y1, t0, t1)
            weight = 0.0 if t1 == t0 else (t_event - t0) / (t1 - t0)
            y_event = (1.0 - weight) * y0 + weight * y1
            found.append(EventRecord(event.name, t_event, y_event, event.priority))
    found.sort(key=lambda item: (item.time_s, -item.priority))
    return found


def resolve_events(
    events: List[EventSpec], t0: float, y0: np.ndarray, t1: float, y1: np.ndarray
) -> List[EventRecord]:
    """Корни внутри шага. При равном времени выше приоритет. Гистерезис снимает дребезг."""
    armed = []
    for event in events:
        value = float(event.function(t1, y1))
        if not event.armed:
            if value < -event.hysteresis:
                event.armed = True
            continue
        armed.append(event)
    found = collect_sign_changes(armed, t0, y0, t1, y1)
    fired = {record.name for record in found}
    for event in armed:
        if event.name in fired:
            event.armed = False
    return found
