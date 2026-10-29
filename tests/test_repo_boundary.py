"""OpenMC и ядерные данные в рабочее дерево не кладутся."""

from pathlib import Path


def test_openmc_library_and_nuclear_data_stay_out_of_the_repository():
    root = Path(__file__).resolve().parents[1]
    project = (root / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "openmc" not in project
    skip = {".git", "__pycache__", ".pytest_cache", "build", "dist", ".eggs"}
    found = []
    for path in root.rglob("*"):
        if any(part in skip for part in path.parts) or not path.is_file():
            continue
        name = path.name.lower()
        if name in {"cross_sections.xml", "summary.h5", "tallies.out"} or name.startswith(
            "statepoint."
        ):
            found.append(str(path.relative_to(root)))
        if "endfb" in name or name.endswith(".tar.xz"):
            found.append(str(path.relative_to(root)))
    assert found == []
