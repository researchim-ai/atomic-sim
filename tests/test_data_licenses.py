"""В репозиторий попадают только данные с разрешённой лицензией."""

import json
from pathlib import Path

import pytest

from core.status import DomainError
from data.catalog import assert_bundled_checksum, assert_nuclear_data_allowed, load_sources


def test_bundled_library_is_apache_and_restricted_sources_stay_out():
    bundled = assert_bundled_checksum("project-open-two-group")
    assert bundled["license"] == "Apache-2.0"
    sources = {item["id"]: item for item in load_sources()}
    for source_id in ("ans-5.1", "lwr-uam", "plant-measurements"):
        assert sources[source_id]["bundled"] is False
        assert sources[source_id]["redistribution_allowed"] is False
        with pytest.raises(DomainError):
            assert_bundled_checksum(source_id)
    assert sources["endf-b-nndc"]["bundled"] is False
    assert sources["endf-b-nndc"]["redistribution_allowed"] is True
    assert "nndc.bnl.gov" in sources["endf-b-nndc"]["url"]


def test_openmc_path_without_endf_license_is_refused(tmp_path):
    library = tmp_path / "cross_sections.xml"
    library.write_text("<cross_sections/>", encoding="utf-8")
    with pytest.raises(DomainError):
        assert_nuclear_data_allowed(str(library))
    sidecar = Path(str(library) + ".license.json")
    sidecar.write_text(
        json.dumps({"source_id": "ans-5.1", "redistribution_allowed": False}),
        encoding="utf-8",
    )
    with pytest.raises(DomainError):
        assert_nuclear_data_allowed(str(library))
    sidecar.write_text(
        json.dumps({"source_id": "endf-b-nndc", "redistribution_allowed": True}),
        encoding="utf-8",
    )
    accepted = assert_nuclear_data_allowed(str(library))
    assert accepted["id"] == "endf-b-nndc"
