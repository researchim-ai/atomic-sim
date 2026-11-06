"""Immutable JSON revisions, relative content-addressed data, bounded portable ZIP."""

import hashlib
import io
import json
import os
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath

from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_training.runner import Campaign
from atomic_model_training.spec import admit_campaign, atomic_json, exact, read_json
from atomic_model_workspace.common import digest

MAX_FILE = 16 * 1024**2
MAX_TOTAL = 128 * 1024**2
MAX_REVISIONS = 64


def local(root, name):
    if not isinstance(name, str) or "\\" in name or ":" in name:
        raise ValueError("Portable relative path required")
    path = PurePosixPath(name)
    if path.is_absolute() or str(path) != name or any(p in (".", "..") for p in path.parts):
        raise ValueError("Unsafe relative path")
    target = Path(root)
    for part in path.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError("Symlink project input")
    return target


def regular_bytes(path, limit=MAX_FILE):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Project input file budget/type")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("Project file budget exceeded")
        return raw
    finally:
        os.close(fd)


def normalize(spec):
    exact(spec, {"schema_version", "model", "model_config", "cases", "data", "agent"})
    if (
        type(spec["schema_version"]) is not int
        or spec["schema_version"] != 1
        or spec["model"] != "pwr-reduced-flow-controlled"
        or spec["model_config"] != {}
    ):
        raise ValueError("Only the qualified default controlled-core configuration is supported")
    if spec["agent"] != {"kind": "declarative-json", "trusted_code": False}:
        raise ValueError("Project accepts declarative agent data only")
    cases = admit_campaign(spec["cases"])
    if not isinstance(spec["data"], list) or len(spec["data"]) > 64:
        raise ValueError("Data inventory budget")
    seen = set()
    for item in spec["data"]:
        exact(item, {"id", "path", "sha256", "license"})
        if not isinstance(item["id"], str) or item["id"] in seen or not item["license"]:
            raise ValueError("Duplicate or undocumented data identity")
        seen.add(item["id"])
        local(".", item["path"])
    return dict(spec, cases=cases)


class Project:
    def __init__(self, root):
        self.root = Path(root).absolute()
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError("Symlink project root")

    @classmethod
    def create(cls, root, spec, *, data_root=None):
        spec = normalize(spec)
        project = cls(root)
        project.root.mkdir(parents=True, exist_ok=False)
        (project.root / "revisions").mkdir()
        (project.root / "data").mkdir()
        try:
            rewritten = []
            for item in spec["data"]:
                raw = regular_bytes(local(data_root or ".", item["path"]))
                sha = hashlib.sha256(raw).hexdigest()
                if sha != item["sha256"]:
                    raise ValueError("Data checksum differs")
                relative = "data/" + sha
                target = project.root / relative
                if not target.exists():
                    target.write_bytes(raw)
                rewritten.append(dict(item, path=relative))
            spec = dict(spec, data=rewritten)
            project._commit(spec, parent=None)
        except BaseException:
            shutil.rmtree(project.root)
            raise
        return project

    def _commit(self, spec, parent):
        from .release import identity

        cursor, count = parent, 0
        while cursor is not None:
            count += 1
            if count >= MAX_REVISIONS:
                raise ValueError("Project revision budget; HEAD left unchanged")
            cursor = unseal(read_json(local(self.root, "revisions/" + cursor + ".json")))["parent"]
        revision = seal(dict(spec=spec, parent=parent, source=identity()))
        sha = revision["sha256"]
        p = self.root / "revisions" / (sha + ".json")
        if not p.exists():
            atomic_json(p, revision)
        atomic_json(self.root / "HEAD.json", seal(dict(revision=sha)))
        return sha

    def read(self):
        from .release import identity

        head = unseal(read_json(self.root / "HEAD.json"))
        sha = head["revision"]
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("Invalid revision reference")
        revision = unseal(read_json(local(self.root, "revisions/" + sha + ".json")))
        if digest(revision) != sha or revision["source"] != identity():
            raise ValueError("Revision/source identity differs")
        previous = revision["parent"]
        visited = {sha}
        while previous is not None:
            if (
                not isinstance(previous, str)
                or len(previous) != 64
                or any(c not in "0123456789abcdef" for c in previous)
                or previous in visited
                or len(visited) >= MAX_REVISIONS
            ):
                raise ValueError("Invalid or excessive revision ancestry")
            ancestor = unseal(read_json(local(self.root, "revisions/" + previous + ".json")))
            if digest(ancestor) != previous or ancestor["source"] != identity():
                raise ValueError("Revision ancestry changed")
            normalize(ancestor["spec"])
            visited.add(previous)
            previous = ancestor["parent"]
        spec = normalize(revision["spec"])
        for item in spec["data"]:
            if (
                hashlib.sha256(regular_bytes(local(self.root, item["path"]))).hexdigest()
                != item["sha256"]
            ):
                raise ValueError("Versioned project data changed")
        return spec, sha

    def revise(self, spec, *, expected_revision):
        current, sha = self.read()
        if expected_revision != sha:
            raise ValueError("Stale project revision")
        spec = normalize(spec)
        if spec["data"] != current["data"]:
            raise ValueError("Data revisions require new content-addressed project import")
        # Serialize competing writers using exclusive ownership; no blind HEAD overwrite.
        lock = self.root / ".edit.lock"
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            if self.read()[1] != sha:
                raise ValueError("Project changed during revision")
            return self._commit(spec, sha)
        finally:
            os.close(fd)
            lock.unlink()

    def run(self, directory, *, resume=False, workers=1):
        spec, revision = self.read()
        report = Campaign(spec["cases"], directory, workers=workers).run(resume=resume)
        receipt = seal(
            dict(
                revision=revision,
                report=report,
                export_schema="controlled-core-state-with-SI-units-v1",
            )
        )
        atomic_json(self.root / "last-run.json", receipt)
        return receipt

    def export(self, output):
        self.read()
        output = Path(output).absolute()
        if output.is_relative_to(self.root):
            raise ValueError("Archive cannot be inside project")
        files = {}
        total = 0
        for p in sorted(self.root.rglob("*")):
            if p.is_symlink():
                raise ValueError("Symlink in project")
            if p.is_file():
                raw = regular_bytes(p)
                total += len(raw)
                if total > MAX_TOTAL or len(files) >= 500:
                    raise ValueError("Project archive budget")
                files[str(p.relative_to(self.root))] = raw
        manifest = {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}
        if "INVENTORY.json" in files:
            raise ValueError("Reserved project archive filename")
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in files.items():
                archive.writestr(name, raw)
            archive.writestr("INVENTORY.json", json.dumps(seal(manifest), sort_keys=True))
        return dict(
            archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), files=len(files)
        )

    @classmethod
    def import_archive(cls, archive, output):
        target = cls(output)
        target.root.mkdir(parents=True, exist_ok=False)
        try:
            with zipfile.ZipFile(io.BytesIO(regular_bytes(archive, limit=MAX_TOTAL))) as z:
                members = z.infolist()
                names = [m.filename for m in members]
                if (
                    len(names) != len(set(names))
                    or len(names) > 501
                    or sum(m.file_size for m in members) > MAX_TOTAL
                ):
                    raise ValueError("Duplicate or oversized ZIP inventory")
                if "INVENTORY.json" not in names:
                    raise ValueError("Missing project inventory")
                inventory = None
                content = {}
                for m in members:
                    local(target.root, m.filename)
                    if m.file_size > MAX_FILE or stat.S_ISLNK(m.external_attr >> 16) or m.is_dir():
                        raise ValueError("Unsupported ZIP member")
                    with z.open(m) as stream:
                        raw = stream.read(MAX_FILE + 1)
                    if len(raw) != m.file_size:
                        raise ValueError("ZIP member length differs")
                    if m.filename == "INVENTORY.json":
                        inventory = unseal(json.loads(raw))
                    else:
                        content[m.filename] = raw
                if inventory != {
                    name: hashlib.sha256(raw).hexdigest() for name, raw in content.items()
                }:
                    raise ValueError("Portable project inventory differs")
                for name, raw in content.items():
                    p = local(target.root, name)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(raw)
            target.read()
        except BaseException:
            shutil.rmtree(target.root)
            raise
        return target
