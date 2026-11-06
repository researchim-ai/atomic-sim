"""Сборка типов. Неподдержанный тип возвращает unsupported, а не чужую схему."""

from __future__ import annotations

from core.status import Applicability, DomainError
from plants.rbmk import RBMKChannel
from plants.reactor_types import BWRPlant, HTGRBrayton, MSRPrecursors, SFRLoop
from plants.vver import VVERPlant

SUPPORTED = ("vver", "pwr", "rbmk", "bwr", "sfr", "htgr", "msr")


def build_plant(kind: str):
    key = kind.lower()
    if key in ("vver", "pwr"):
        return VVERPlant()
    if key == "rbmk":
        return RBMKChannel()
    if key == "bwr":
        return BWRPlant()
    if key == "sfr":
        return SFRLoop()
    if key == "htgr":
        return HTGRBrayton()
    if key == "msr":
        return MSRPrecursors()
    raise DomainError("Неизвестный тип %s" % kind, Applicability.UNSUPPORTED)
