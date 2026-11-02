"""Portable delivery integrity, hostile archives, bounded IO and rollback."""

import gzip
import hashlib
import io
import json
import tarfile
import zipfile
from copy import deepcopy

import pytest

from scripts import evidence_bundle as bundle


@pytest.fixture
def selection(tmp_path):
    (tmp_path / "input.json").write_text('{"power_w":100}\n')
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "result.txt").write_text("Numerical scope only\n")
    return dict(
        schema_version=1,
        bundle_id="test-release",
        scope="Byte integrity test",
        entries=[
            dict(source="input.json", path="case/input.json"),
            dict(source="logs", path="case/logs"),
        ],
    )


def members(path):
    with tarfile.open(path, "r:gz") as archive:
        return [(member, archive.extractfile(member).read()) for member in archive]


def repack(path, rows):
    with tarfile.open(path, "w:gz") as archive:
        for member, data in rows:
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def change_payload(path):
    rows = members(path)
    repack(path, [(m, b"corrupted" if m.name == "case/input.json" else d) for m, d in rows])


def test_deterministic_roundtrip_preserves_all_bytes_without_acceptance_claim(selection, tmp_path):
    first, second = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    result = bundle.create_bundle(selection, tmp_path, first)
    other = bundle.create_bundle(selection, tmp_path, second)
    assert first.read_bytes() == second.read_bytes()
    assert result["archive_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert result["manifest_sha256"] == other["manifest_sha256"]
    assert result["integrity_verified"] and not result["runtime_source_verified"]
    assert not result["numerical_acceptance_recomputed"] and not result["publisher_authenticated"]
    output = tmp_path / "extracted"
    bundle.extract_bundle(first, output, result["manifest_sha256"])
    assert (output / "case/input.json").read_bytes() == (tmp_path / "input.json").read_bytes()
    assert (output / "case/logs/result.txt").read_bytes() == (
        tmp_path / "logs/result.txt"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        bundle.extract_bundle(first, output)
    assert (output / "case/input.json").exists()


def test_publisher_checksum_rejects_rewritten_manifest(selection, tmp_path):
    path = tmp_path / "bundle.tar.gz"
    result = bundle.create_bundle(selection, tmp_path, path)
    rows = members(path)
    rows = [
        (
            m,
            bundle.canonical(dict(json.loads(d), scope="Changed scope"))
            if m.name == "MANIFEST.json"
            else d,
        )
        for m, d in rows
    ]
    repack(path, rows)
    assert bundle.verify_bundle(path)["integrity_verified"]
    with pytest.raises(ValueError, match="Trusted manifest checksum"):
        bundle.verify_bundle(path, result["manifest_sha256"])


@pytest.mark.parametrize(
    "path",
    [
        "",
        ".",
        "..",
        "../escape",
        "/absolute",
        "a/../b",
        "a//b",
        "./a",
        "a/",
        "a\\b",
        "C:drive",
        "NUL.txt",
        "a/COM1",
        "a.",
        "a ",
        "a\x00b",
        "a?b",
        'a"b',
    ],
)
def test_unsafe_selection_destinations_are_rejected(selection, tmp_path, path):
    selection["entries"][0]["path"] = path
    with pytest.raises(ValueError):
        bundle.create_bundle(selection, tmp_path, tmp_path / "bundle.tar.gz")
    assert not (tmp_path / "bundle.tar.gz").exists()


@pytest.mark.parametrize("name", ["case/INPUT.json", "MANIFEST.json", "inventory.JSON", "case"])
def test_collisions_and_reserved_destinations_reject(selection, tmp_path, name):
    selection["entries"].append(dict(source="input.json", path=name))
    with pytest.raises(ValueError):
        bundle.create_bundle(selection, tmp_path, tmp_path / "bundle.tar.gz")


@pytest.mark.parametrize("directory", [False, True])
def test_source_symlinks_are_not_followed(selection, tmp_path, directory):
    (tmp_path / "link").symlink_to(tmp_path / ("logs" if directory else "input.json"))
    selection["entries"] = [dict(source="link", path="payload")]
    with pytest.raises(ValueError, match="symlink"):
        bundle.create_bundle(selection, tmp_path, tmp_path / "bundle.tar.gz")


def test_symlink_inside_selected_directory_rejects(selection, tmp_path):
    (tmp_path / "logs/link").symlink_to(tmp_path / "input.json")
    with pytest.raises(ValueError, match="symlink"):
        bundle.create_bundle(selection, tmp_path, tmp_path / "bundle.tar.gz")


@pytest.mark.parametrize("source", ["../outside", "/etc/passwd"])
def test_sources_must_be_within_explicit_root(selection, tmp_path, source):
    selection["entries"][0]["source"] = source
    with pytest.raises(ValueError):
        bundle.create_bundle(selection, tmp_path, tmp_path / "bundle.tar.gz")


def test_corruption_fails_before_creating_destination(selection, tmp_path):
    path = tmp_path / "bundle.tar.gz"
    bundle.create_bundle(selection, tmp_path, path)
    change_payload(path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        bundle.extract_bundle(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("attack", ["duplicate", "missing", "extra", "case_collision", "ancestor"])
def test_archive_member_set_cannot_be_substituted(selection, tmp_path, attack):
    path = tmp_path / "bundle.tar.gz"
    bundle.create_bundle(selection, tmp_path, path)
    rows = members(path)
    if attack == "duplicate":
        rows.append(rows[0])
    elif attack == "missing":
        rows.pop(0)
    else:
        name = {"extra": "extra.txt", "case_collision": "CASE/INPUT.json", "ancestor": "case"}[
            attack
        ]
        rows.append((tarfile.TarInfo(name), b"injected"))
    repack(path, rows)
    with pytest.raises(ValueError):
        bundle.verify_bundle(path)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_link_and_special_members_reject_before_extraction(tmp_path, kind):
    path = tmp_path / "hostile.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("link")
        info.type, info.linkname = kind, "../../escape"
        archive.addfile(info)
    with pytest.raises(ValueError):
        bundle.extract_bundle(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_pax_path_override_is_validated(tmp_path):
    path = tmp_path / "pax.tar.gz"
    with tarfile.open(path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        info = tarfile.TarInfo("safe.txt")
        info.pax_headers = {"path": "../escaped.txt"}
        archive.addfile(info, io.BytesIO())
    with pytest.raises(ValueError):
        bundle.extract_bundle(path, tmp_path / "output")
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.parametrize("kind", [tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME])
def test_oversized_extension_header_rejects_before_payload_is_read(tmp_path, kind):
    member = tarfile.TarInfo("extension")
    member.type = kind
    member.size = bundle.Limits().max_manifest_bytes + 1
    path = tmp_path / "oversized-header.tar.gz"
    path.write_bytes(gzip.compress(member.tobuf(format=tarfile.USTAR_FORMAT) + b"\0" * 1024))
    with pytest.raises(ValueError, match="Raw tar header resource"):
        bundle.verify_bundle(path)


@pytest.mark.parametrize(
    "limits",
    [
        bundle.Limits(max_files=1),
        bundle.Limits(max_file_bytes=1),
        bundle.Limits(max_total_bytes=1),
        bundle.Limits(max_manifest_bytes=1),
    ],
)
def test_resource_limits_fail_without_publishing(selection, tmp_path, limits):
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError):
        bundle.create_bundle(selection, tmp_path, tmp_path / "bundle.tar.gz", limits)
    assert set(tmp_path.iterdir()) == before


def test_existing_archive_is_never_replaced(selection, tmp_path):
    path = tmp_path / "bundle.tar.gz"
    path.write_bytes(b"user-owned")
    with pytest.raises(FileExistsError):
        bundle.create_bundle(selection, tmp_path, path)
    assert path.read_bytes() == b"user-owned"


def test_extraction_rechecks_changed_input_and_rolls_back(selection, tmp_path, monkeypatch):
    path = tmp_path / "bundle.tar.gz"
    bundle.create_bundle(selection, tmp_path, path)
    original = bundle.verify_bundle

    def verify_then_change(*args):
        result = original(*args)
        change_payload(path)
        return result

    monkeypatch.setattr(bundle, "verify_bundle", verify_then_change)
    with pytest.raises(ValueError, match="changed|checksum"):
        bundle.extract_bundle(path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_verification_and_extraction_never_execute_payloads(selection, tmp_path):
    sentinel = tmp_path / "executed"
    (tmp_path / "payload.py").write_text(
        f"from pathlib import Path; Path({str(sentinel)!r}).touch()"
    )
    selection["entries"].append(dict(source="payload.py", path="scripts/payload.py"))
    path = tmp_path / "bundle.tar.gz"
    bundle.create_bundle(selection, tmp_path, path)
    bundle.verify_bundle(path)
    bundle.extract_bundle(path, tmp_path / "output")
    assert not sentinel.exists()


@pytest.mark.parametrize("attack", ["duplicate_key", "nan", "boolean_size", "missing_hash"])
def test_invalid_manifest_metadata_is_rejected(selection, tmp_path, attack):
    path = tmp_path / "bundle.tar.gz"
    bundle.create_bundle(selection, tmp_path, path)
    rows = members(path)
    changed = []
    for member, data in rows:
        if member.name == "MANIFEST.json":
            value = json.loads(data)
            if attack == "duplicate_key":
                data = b'{"schema_version":1,' + data[1:]
            elif attack == "nan":
                value["metadata"] = {"invalid": float("nan")}
                data = json.dumps(value).encode()
            elif attack == "boolean_size":
                value["files"][0]["size_bytes"] = True
                data = bundle.canonical(value)
            else:
                del value["files"][0]["sha256"]
                data = bundle.canonical(value)
        changed.append((member, data))
    repack(path, changed)
    with pytest.raises(ValueError):
        bundle.verify_bundle(path)


def runtimes(tmp_path, change_sdist=False):
    sources = {f"{name}/__init__.py": f'"""{name}"""\n'.encode() for name in bundle.PACKAGES}
    wheel, sdist = tmp_path / "runtime.whl", tmp_path / "runtime.tar.gz"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in sources.items():
            archive.writestr(name, data)
        archive.writestr(
            "atomic_sim-2.0.dist-info/METADATA",
            "Metadata-Version: 2.3\nName: atomic-sim\n"
            "Version: 2.0\nLicense: Apache-2.0\nRequires-Dist: numpy>=1.24\n",
        )
    with tarfile.open(sdist, "w:gz") as archive:
        for name, data in sources.items():
            if change_sdist and name.startswith("physics/"):
                data += b"# modified\n"
            info = tarfile.TarInfo("atomic_sim-2.0/" + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    fingerprint = bundle.runtime_fingerprint(wheel.read_bytes(), "wheel")
    return dict(
        wheel="release/runtime.whl", sdist="release/runtime.tar.gz", source_sha256=fingerprint
    )


def test_wheel_and_sdist_sources_are_bound_to_same_fingerprint(selection, tmp_path):
    selection["runtime"] = runtimes(tmp_path)
    selection["entries"] += [
        dict(source=n, path="release/" + n) for n in ("runtime.whl", "runtime.tar.gz")
    ]
    path = tmp_path / "bundle.tar.gz"
    result = bundle.create_bundle(selection, tmp_path, path)
    assert result["runtime_source_verified"]
    assert result["source_sha256"] == selection["runtime"]["source_sha256"]
    with tarfile.open(path, "r:gz") as archive:
        inventory = json.load(archive.extractfile("INVENTORY.json"))
    assert inventory["project"]["license_metadata"] == "Apache-2.0"
    assert inventory["distributions"][0]["name"] == "numpy"


def test_mixing_different_runtime_sources_is_not_published(selection, tmp_path):
    selection["runtime"] = runtimes(tmp_path, change_sdist=True)
    selection["entries"] += [
        dict(source=n, path="release/" + n) for n in ("runtime.whl", "runtime.tar.gz")
    ]
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        bundle.create_bundle(selection, tmp_path, tmp_path / "bundle.tar.gz")
    assert not (tmp_path / "bundle.tar.gz").exists()


def test_incomplete_runtime_is_rejected(tmp_path):
    runtimes(tmp_path)
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr("sdk/__init__.py", b"")
    with pytest.raises(ValueError, match="Incomplete runtime"):
        bundle.runtime_fingerprint(target.getvalue(), "wheel")


def test_nested_unselected_files_are_also_resource_bounded(tmp_path):
    runtimes(tmp_path)
    with zipfile.ZipFile(tmp_path / "runtime.whl", "a") as archive:
        archive.writestr("unrelated.txt", b"x" * 300)
    with pytest.raises(ValueError, match="Nested archive resource"):
        bundle.runtime_fingerprint(
            (tmp_path / "runtime.whl").read_bytes(), "wheel", bundle.Limits(max_file_bytes=200)
        )


def test_runtime_fingerprint_uses_path_component_order_like_sdk(tmp_path):
    runtimes(tmp_path)
    with zipfile.ZipFile(tmp_path / "runtime.whl", "a") as archive:
        archive.writestr("sdk/a.py", b"# flat module\n")
        archive.writestr("sdk/a/nested.py", b"# directory with the same stem\n")
    digest = hashlib.sha256()
    with zipfile.ZipFile(tmp_path / "runtime.whl") as archive:
        for package in bundle.PACKAGES:
            names = [n for n in archive.namelist() if n.startswith(package + "/")]
            # Path ordering differs from string ordering for a.py vs a/nested.py.
            for name in sorted(names, key=lambda p: tuple(p.split("/"))):
                digest.update(name.encode())
                digest.update(b"\0")
                digest.update(archive.read(name))
    assert (
        bundle.runtime_fingerprint((tmp_path / "runtime.whl").read_bytes(), "wheel")
        == digest.hexdigest()
    )


def test_cli_create_verify_extract(selection, tmp_path, monkeypatch, capsys):
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(selection))
    output = tmp_path / "bundle.tar.gz"
    monkeypatch.setattr(
        "sys.argv",
        ["bundle", "create", str(path), "--root", str(tmp_path), "--output", str(output)],
    )
    bundle.main()
    created = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(
        "sys.argv",
        ["bundle", "verify", str(output), "--manifest-sha256", created["manifest_sha256"]],
    )
    bundle.main()
    assert json.loads(capsys.readouterr().out)["integrity_verified"]
    monkeypatch.setattr(
        "sys.argv", ["bundle", "extract", str(output), "--output", str(tmp_path / "out")]
    )
    bundle.main()
    assert json.loads(capsys.readouterr().out)["output"] == str(tmp_path / "out")


def test_invalid_schema_and_nonfinite_metadata_do_not_create_archive(selection, tmp_path):
    for bad in (
        dict(selection, schema_version=True),
        dict(selection, entries=[]),
        dict(selection, metadata={"invalid": float("nan")}),
    ):
        with pytest.raises(ValueError):
            bundle.create_bundle(bad, tmp_path, tmp_path / "bundle.tar.gz")
    assert not (tmp_path / "bundle.tar.gz").exists()


def test_declared_runtime_must_exist_and_use_exact_identity_fields(selection, tmp_path):
    runtime = runtimes(tmp_path)
    bad = deepcopy(selection)
    bad["runtime"] = runtime
    with pytest.raises(ValueError, match="missing"):
        bundle.create_bundle(bad, tmp_path, tmp_path / "bundle.tar.gz")
    bad["runtime"]["source_sha256"] = hashlib.sha256(b"wrong").hexdigest()
    bad["runtime"]["unrecognized"] = "field"
    with pytest.raises(ValueError, match="Invalid runtime identity"):
        bundle.create_bundle(bad, tmp_path, tmp_path / "bundle.tar.gz")
