"""Какие данные можно класть в репозиторий и подключать к OpenMC.

Внешний нейтронный расчёт принимается только для ENDF/B с NNDC и только если
рядом с файлом сечений лежит подтверждение лицензии. Чужая библиотека не
подменяется тестовыми макросечениями проекта.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from core.status import DomainError

_CATALOG = Path(__file__).with_name("sources.json")
_ALLOWED_EXTERNAL = {"endf-b-nndc"}


def load_sources() -> list:
    payload = json.loads(_CATALOG.read_text(encoding="utf-8"))
    return list(payload["sources"])


def bundled_redistributable_ids() -> list:
    """Источники, которые и лежат в репозитории, и разрешены к распространению."""
    return [
        item["id"]
        for item in load_sources()
        if item["bundled"] and item["redistribution_allowed"]
    ]


def source_by_id(source_id: str) -> dict:
    for item in load_sources():
        if item["id"] == source_id:
            return item
    raise DomainError("Источник данных %s не описан" % source_id)


def assert_bundled_checksum(source_id: str) -> dict:
    item = source_by_id(source_id)
    if not item["bundled"]:
        raise DomainError("Источник %s в репозиторий не входит" % source_id)
    if not item["redistribution_allowed"]:
        raise DomainError("Источник %s нельзя распространять с этим кодом" % source_id)
    path = Path(item["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        raise DomainError("Контрольная сумма %s не совпала" % source_id)
    return item


def assert_nuclear_data_allowed(cross_section_path: str) -> dict:
    """Локальный файл сечений без лицензионной метки не запускает частицы."""
    sidecar = Path(str(cross_section_path) + ".license.json")
    if not sidecar.is_file():
        raise DomainError(
            "Рядом с ядерными данными нет license.json. Допустим только ENDF/B с NNDC"
        )
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    source_id = meta.get("source_id")
    if source_id not in _ALLOWED_EXTERNAL:
        raise DomainError("Источник %s не входит в разрешённые ядерные данные" % source_id)
    catalog = source_by_id(source_id)
    if not catalog["redistribution_allowed"] or meta.get("redistribution_allowed") is not True:
        raise DomainError("Лицензия %s не позволяет использовать файл в этом проекте" % source_id)
    return catalog
