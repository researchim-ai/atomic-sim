"""Independent legacy-order oracle, fresh bytes and snapshot identity checks."""

import hashlib
import os
from pathlib import Path

import pytest

from sdk.experiment import _source_fingerprint, _source_fingerprint_at

PACKAGES = (
    "sdk",
    "core",
    "plants",
    "physics",
    "components",
    "control",
    "instrumentation",
    "agent_envs",
    "data",
    "scenarios",
)


def legacy(root, resource_root=None):
    h = hashlib.sha256()
    for package in PACKAGES:
        base = (
            resource_root
            if resource_root is not None and package in ("data", "scenarios")
            else root
        ) / package
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in (".py", ".json", ".yaml"):
                h.update(str(Path(package) / p.relative_to(base)).encode())
                h.update(b"\0")
                h.update(p.read_bytes())
    return h.hexdigest()


def put(root, name, data="text"):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data)
    return p


def test_real_tree_byte_stream_matches_independent_legacy_oracle():
    import sdk.experiment as module

    root = Path(module.__file__).resolve().parents[1]
    import scenarios

    resources = Path(scenarios.__file__).resolve().parents[1]
    assert (
        _source_fingerprint()
        == legacy(root, resources)
        == _source_fingerprint_at(root, resource_root=resources)
    )


@pytest.mark.parametrize(
    "names",
    [
        ["sdk/a.py", "sdk/a/child.py", "sdk/a-0.py", "sdk/a0.py"],
        ["physics/z.yaml", "physics/a.json", "core/a.py", "sdk/z.py"],
        ["sdk/é.py", "sdk/Ω/x.py", "sdk/Z.py", "sdk/z.py"],
        ["sdk/.hidden.py", "sdk/__pycache__/unexpected.py", "sdk/cache.pyc", "sdk/ignored.txt"],
        ["sdk/a.PY", "sdk/a.JSON", "sdk/a.yml", "scenarios/a.yaml"],
        [],
    ],
)
def test_order_suffix_and_hidden_file_semantics_match_legacy(tmp_path, names):
    for i, name in enumerate(reversed(names)):
        put(tmp_path, name, str(i))
    assert _source_fingerprint_at(tmp_path) == legacy(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "bytes",
        "backdated_same_size",
        "add",
        "delete",
        "rename",
        "extension",
        "empty_file",
        "ignored",
    ],
)
def test_every_call_scans_fresh_names_and_contents(tmp_path, mutation):
    p = put(tmp_path, "sdk/source.py", "AAAA")
    before = _source_fingerprint_at(tmp_path)
    if mutation == "bytes":
        p.write_text("BBBBB")
    elif mutation == "backdated_same_size":
        stat = p.stat()
        p.write_text("BBBB")
        os.utime(p, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    elif mutation == "add":
        put(tmp_path, "core/new.py", "B")
    elif mutation == "delete":
        p.unlink()
    elif mutation == "rename":
        p.rename(p.with_name("renamed.py"))
    elif mutation == "extension":
        p.rename(p.with_suffix(".txt"))
    elif mutation == "empty_file":
        put(tmp_path, "core/empty.json", "")
    elif mutation == "ignored":
        put(tmp_path, "core/not_source.txt", "B")
    after = _source_fingerprint_at(tmp_path)
    assert after == legacy(tmp_path)
    assert (after == before) == (mutation == "ignored")


def test_file_symlinks_and_directory_symlinks_keep_legacy_policy(tmp_path):
    put(tmp_path, "sdk/real.py", "A")
    put(tmp_path, "external/out.py", "B")
    (tmp_path / "sdk/link.py").symlink_to("../external/out.py")
    (tmp_path / "sdk/dir").symlink_to("../external", target_is_directory=True)
    (tmp_path / "sdk/broken.py").symlink_to("missing.py")
    assert _source_fingerprint_at(tmp_path) == legacy(tmp_path)
    old = _source_fingerprint_at(tmp_path)
    (tmp_path / "external/out.py").write_text("C")
    assert _source_fingerprint_at(tmp_path) == legacy(tmp_path) != old


def test_top_level_package_link_matches_legacy(tmp_path):
    put(tmp_path, "actual/nested/a.py", "X")
    (tmp_path / "sdk").symlink_to("actual", target_is_directory=True)
    assert _source_fingerprint_at(tmp_path) == legacy(tmp_path)


def test_checkpoint_source_guard_still_rejects_changed_tree(monkeypatch):
    import scripts.compressible_storage_graph as module

    from atomic_model_runtime import create_eos_episode

    env = create_eos_episode()
    before = env.export_episode()
    monkeypatch.setattr(module, "_source_fingerprint", lambda: "0" * 64)
    with pytest.raises(ValueError):
        env.export_episode()
    monkeypatch.undo()
    assert env.export_episode() == before


def test_unreadable_directory_fails_closed_instead_of_using_previous_digest(tmp_path, monkeypatch):
    put(tmp_path, "sdk/source.py", "X")
    _source_fingerprint_at(tmp_path)
    original = os.scandir

    def deny(path):
        if os.fspath(path) == os.fspath(tmp_path / "sdk"):
            raise PermissionError("injected")
        return original(path)

    monkeypatch.setattr(os, "scandir", deny)
    with pytest.raises(PermissionError):
        _source_fingerprint_at(tmp_path)


def test_unreadable_source_bytes_fail_closed(tmp_path, monkeypatch):
    import builtins

    p = put(tmp_path, "sdk/source.py", "X")
    _source_fingerprint_at(tmp_path)
    original = builtins.open

    def deny(file, *args, **kwargs):
        if os.fspath(file) == os.fspath(p):
            raise PermissionError("injected")
        return original(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny)
    with pytest.raises(PermissionError):
        _source_fingerprint_at(tmp_path)


def test_split_source_and_resource_roots_match_oracle(tmp_path):
    source, resources = tmp_path / "src", tmp_path / "resources"
    put(source, "sdk/module.py", "source")
    put(resources, "scenarios/case.yaml", "case")
    put(resources, "data/sources.json", "{}")
    assert _source_fingerprint_at(source, resource_root=resources) == legacy(source, resources)
    old = _source_fingerprint_at(source, resource_root=resources)
    put(resources, "data/sources.json", "changed")
    assert _source_fingerprint_at(source, resource_root=resources) != old
