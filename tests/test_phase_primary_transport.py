"""Offline resource-bound and independent-reader checks."""


import pytest

from scripts.fetch_pool_primary import SIZE, NoRedirect, RangeReader
from scripts.verify_pool_primary import verify


def test_range_reader_bounds_without_network():
    reader = RangeReader()
    assert reader.seek(-5, 2) == SIZE - 5
    assert reader.read(0) == b""
    with pytest.raises(ValueError):
        reader.seek(-1)
    with pytest.raises(ValueError):
        reader.seek(SIZE + 1)
    reader.seek(0)
    with pytest.raises(ValueError, match="budget"):
        reader.read(2_000_001)
    with pytest.raises(ValueError):
        NoRedirect().redirect_request(None, None, 302, "", {}, "http://localhost")


def test_independent_reader_rejects_unpinned_csv(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("not primary")
    with pytest.raises(ValueError, match="identity"):
        verify(path)
