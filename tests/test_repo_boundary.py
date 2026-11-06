"""Do not vendor transport libraries/data; ignored run evidence may retain HDF5."""

from pathlib import Path


def violations(root):
    skip = {".git", "__pycache__", ".pytest_cache", ".cache", "build", "dist", ".eggs"}
    found = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in skip for part in relative.parts) or not path.is_file():
            continue
        name = path.name.lower()
        # Statepoints and summary.h5 are generated transport results, not a
        # cross-section library. They are retained as ignored local evidence.
        generated = name in {"summary.h5", "tallies.out"} or (
            name.startswith("statepoint.") and name.endswith(".h5")
        )
        if generated and relative.parts[0] == "artifacts":
            continue
        if name in {"cross_sections.xml", "summary.h5", "tallies.out"} or name.startswith(
            "statepoint."
        ):
            found.append(relative.as_posix())
        if "endfb" in name or name.endswith(".tar.xz"):
            found.append(relative.as_posix())
    return found


def test_openmc_library_and_nuclear_data_stay_out_of_the_repository():
    root = Path(__file__).resolve().parents[1]
    project = (root / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "openmc" not in project
    assert not (root / "src/openmc").exists()
    assert not (root / "openmc").exists()
    assert violations(root) == []


def test_ignored_transport_outputs_are_allowed_but_source_outputs_are_not(tmp_path):
    generated = tmp_path / "artifacts/run"
    generated.mkdir(parents=True)
    for name in ("summary.h5", "statepoint.45.h5", "tallies.out"):
        (generated / name).write_bytes(b"retained numerical evidence")
    assert violations(tmp_path) == []
    source = tmp_path / "src/model"
    source.mkdir(parents=True)
    (source / "summary.h5").write_bytes(b"must not be bundled")
    assert violations(tmp_path) == ["src/model/summary.h5"]


def test_nuclear_library_is_still_refused_inside_artifacts(tmp_path):
    folder = tmp_path / "artifacts"
    folder.mkdir()
    (folder / "cross_sections.xml").write_text("nuclear data index")
    (folder / "endfb.tar.xz").write_bytes(b"nuclear library archive")
    found = violations(tmp_path)
    assert "artifacts/cross_sections.xml" in found
    assert "artifacts/endfb.tar.xz" in found
