"""Регуляторы и защиты. Не путать с reactor.control — это отдельный пакет ядра."""

from control.regulation import PID, PID_PASSPORT, PROTECTION_PASSPORT, ProtectionLogic

__all__ = ["PID", "PID_PASSPORT", "PROTECTION_PASSPORT", "ProtectionLogic"]
