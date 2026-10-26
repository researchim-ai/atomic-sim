"""Связка с OpenMC, если пакет установлен. Иначе режим unsupported."""

from __future__ import annotations

from core.status import Applicability


def openmc_status() -> dict:
    try:
        import openmc  # type: ignore
    except ImportError as exc:
        return {
            "applicability": Applicability.UNSUPPORTED.value,
            "message": "OpenMC не установлен: %s" % exc,
            "version": None,
        }
    return {
        "applicability": Applicability.IN_DOMAIN.value,
        "message": "OpenMC доступен как внешний адаптер",
        "version": getattr(openmc, "__version__", "unknown"),
    }


class OpenMCAdapter:
    """Не подменяет отсутствующий OpenMC внутренним спектром."""

    def __init__(self) -> None:
        self.status = openmc_status()

    def run_mgxs(self, specification: dict) -> dict:
        if self.status["applicability"] != Applicability.IN_DOMAIN.value:
            return dict(self.status)
        import openmc  # type: ignore

        return {
            "applicability": Applicability.IN_DOMAIN.value,
            "version": self.status["version"],
            "specification_keys": sorted(specification.keys()),
            "openmc_module": openmc.__name__,
        }
