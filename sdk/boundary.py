"""Уточнение допустимого префикса при строгих ограничениях на каждой пробе RHS.

Полученный интервал относится к численной допустимости интегрирования,
не является доказанной скобкой корня физического события.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.status import ModelLimitError, SolverError


@dataclass
class BoundaryAdvance:
    observation: dict
    boundary: dict | None
    events: list
    accepted_prefix_steps: int


def advance_until_limit(plant, dt_s, *, time_tolerance_s=1e-4, **solver_options):
    """Сохранить принятый префикс; при отмене/другом отказе откатить всё."""
    if (
        isinstance(dt_s, bool)
        or not math.isfinite(dt_s)
        or not 0 < dt_s <= 5
        or isinstance(time_tolerance_s, bool)
        or not math.isfinite(time_tolerance_s)
        or not 1e-8 <= time_tolerance_s <= 0.1
    ):
        raise ValueError("Шаг должен быть в (0,5] с, допуск границы в [1e-8,0.1] с")
    saved = plant.snapshot()
    ledger = plant.horizon_ledger
    ledger_mark = None if ledger is None else ledger.checkpoint()
    metadata = (plant.last_energy_ledger, plant.last_motion_events, plant.last_protection_event_s)
    target = plant.time_s + dt_s
    upper = target
    events = []
    ledgers = []
    accepted = 0
    failure = None
    failure_start = plant.time_s

    def attempt(end):
        nonlocal accepted
        observed = plant.advance(end - plant.time_s, **solver_options)
        ledgers.append(plant.last_energy_ledger)
        accepted += 1
        events.extend(plant.last_motion_events)
        if plant.last_protection_event_s is not None:
            events.append({"kind": "protection", "t": plant.last_protection_event_s})
        return observed

    try:
        try:
            observed = attempt(target)
            return BoundaryAdvance(observed, None, events, accepted)
        except ModelLimitError as exc:
            failure = exc
        attempts = 1
        while upper - plant.time_s > time_tolerance_s:
            if attempts >= 64:
                raise SolverError("Исчерпан бюджет уточнения границы применимости")
            end = plant.time_s + 0.5 * (upper - plant.time_s)
            if not plant.time_s < end < upper:
                raise SolverError("Допуск границы меньше разрешения времени")
            attempts += 1
            try:
                attempt(end)
            except ModelLimitError as exc:
                upper, failure = end, exc
                failure_start = plant.time_s
        boundary = {
            "kind": "model_limit",
            "scope": "integration_admissibility",
            "t": plant.time_s,
            "accepted_time_s": plant.time_s,
            "failed_endpoint_s": upper,
            "failed_attempt_start_s": failure_start,
            "interval_width_s": upper - plant.time_s,
            "requested_time_tolerance_s": time_tolerance_s,
            "requested_endpoint_s": target,
            "attempts": attempts,
            "accepted_prefix_steps": accepted,
            "probe": failure.details(),
            "message": str(failure),
            "physical_crossing_localized": False,
        }
        # restore после неудачной пробы возвращает запасы и свойства;
        # потоки/уставки алгебры также должны соответствовать принятому состоянию.
        plant.refresh_algebraic(plant.pack())
        if ledgers:
            # Внешний шаг может состоять из нескольких принятых префиксов.
            # CSV/SDK должны показывать их сумму, а horizon ledger уже учёл
            # каждый из них: повторно observe здесь вызывать нельзя.
            combined = {}
            for account in ("delta_j", "supplied_j"):
                names = {key for item in ledgers for key in item[account]}
                combined[account] = {
                    key: sum(item[account].get(key, 0.0) for item in ledgers) for key in names
                }
            combined["residual_j"] = sum(item["residual_j"] for item in ledgers)
            combined["interval_s"] = plant.time_s - saved.time_s
            plant.last_energy_ledger = combined
        boundary["last_accepted_state"] = plant.model_domain_diagnostics()
        return BoundaryAdvance(plant.observation(), boundary, events, accepted)
    except BaseException:
        if ledger is not None:
            ledger.rollback(ledger_mark)
        plant.restore(saved)
        plant.refresh_algebraic(plant.pack())
        plant.last_energy_ledger, plant.last_motion_events, plant.last_protection_event_s = metadata
        raise
