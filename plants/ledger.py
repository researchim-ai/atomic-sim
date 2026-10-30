"""Накопительный баланс горизонта.

Локальная невязка — ошибка одного принятого шага. Интеграл — их сумма.
Глобальный дрейф считается заново от начального запаса и суммы внешних интегралов.
Короткий горизонт не помечается принятым.
"""

from __future__ import annotations

from core.status import DomainError


class HorizonLedger:
    def __init__(self) -> None:
        self.origin = None
        self.previous = None
        self.energy_residuals_j: list = []
        self.supplied_j: list = []
        self.throughput_j: list = []

    def attach(self, plant) -> None:
        sample = plant.closed_inventories()
        self.origin = dict(sample)
        self.previous = dict(sample)
        self.energy_residuals_j = []
        self.supplied_j = []
        self.throughput_j = []
        plant.horizon_ledger = self

    def observe(self, plant) -> None:
        if self.origin is None or plant.last_energy_ledger is None:
            raise DomainError("Баланс горизонта ещё не открыт принятым шагом")
        now = plant.closed_inventories()
        supplied = float(sum(plant.last_energy_ledger["supplied_j"].values()))
        self.energy_residuals_j.append(float(plant.last_energy_ledger["residual_j"]))
        self.supplied_j.append(supplied)
        self.throughput_j.append(
            sum(abs(float(value)) for value in plant.last_energy_ledger["supplied_j"].values())
        )
        self.previous = now

    def report(self) -> dict:
        if self.origin is None or self.previous is None or not self.energy_residuals_j:
            raise DomainError("Отчёт горизонта пуст")
        energy_integral = float(sum(self.energy_residuals_j))
        supplied = float(sum(self.supplied_j))
        energy_global = float(self.previous["stored_j"] - self.origin["stored_j"] - supplied)
        net_scale = max(1.0, sum(abs(item) for item in self.supplied_j))
        # В номинале источник и сток взаимно сокращаются. Их малая разность
        # не задаёт масштаб точности энергобаланса; сохраняем её как диагностику.
        energy_scale = max(1.0, abs(self.origin["stored_j"]), sum(self.throughput_j))
        mass_drift = float(self.previous["closed_mass_kg"] - self.origin["closed_mass_kg"])
        boron_drift = float(self.previous["boron_kg"] - self.origin["boron_kg"])
        mass_scale = max(1.0, abs(self.origin["closed_mass_kg"]))
        boron_scale = max(1.0e-9, abs(self.origin["boron_kg"]))
        return {
            "steps": len(self.energy_residuals_j),
            "energy_max_abs_j": max(abs(item) for item in self.energy_residuals_j),
            "energy_integral_j": energy_integral,
            "energy_global_j": energy_global,
            "energy_relative": abs(energy_global) / energy_scale,
            "energy_scale_j": energy_scale,
            "energy_net_relative": abs(energy_global) / net_scale,
            "energy_throughput_j": sum(self.throughput_j),
            "energy_normalization": "max(1 J, abs(initial stored energy), gross boundary throughput)",
            "mass_drift_kg": mass_drift,
            "mass_relative": abs(mass_drift) / mass_scale,
            "boron_drift_kg": boron_drift,
            "boron_relative": abs(boron_drift) / boron_scale,
            "full_horizon_accepted": False,
        }

    def checkpoint(self) -> dict:
        """Метка отката для составного шага; накопленные списки не копируются."""
        return {
            "origin": None if self.origin is None else dict(self.origin),
            "previous": None if self.previous is None else dict(self.previous),
            "lengths": tuple(
                len(items)
                for items in (self.energy_residuals_j, self.supplied_j, self.throughput_j)
            ),
        }

    def rollback(self, checkpoint: dict) -> None:
        self.origin, self.previous = checkpoint["origin"], checkpoint["previous"]
        for items, length in zip(
            (self.energy_residuals_j, self.supplied_j, self.throughput_j), checkpoint["lengths"]
        ):
            del items[length:]
