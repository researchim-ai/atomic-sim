"""Data-only deterministic project archives with bounded verification before import.

Integrity is not publisher authentication. Import never executes archive code.
"""

import hashlib
import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from atomic_model_workspace.campaign import load_campaign
from atomic_model_workspace.common import canonical, digest, keys, loads, portable, read
from atomic_model_workspace.jobs import load_job
from atomic_model_workspace.project import load_project
from atomic_model_workspace.reports import campaign_report

MAX_FILES = 4096
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024


def _encoded(value):
    return (canonical(value) + "\n").encode()


def _project_files(project):
    manifest = {k: project[k] for k in ("schema_version", "project_id", "model_id", "limits")}
    files = {}
    for group in ("scenarios", "policies"):
        manifest[group] = {}
        for name, value in project[group].items():
            path = f"{group}/{name}.json"
            manifest[group][name] = path
            files["project/" + path] = _encoded(value)
    files["project/project.json"] = _encoded(manifest)
    return files


def export_bundle(project_path, output, *, campaign=None):
    project = load_project(project_path)
    files = _project_files(project)
    if campaign is not None:
        campaign = Path(campaign)
        # Holding the campaign lock prevents its supervisor; job locks prevent
        # direct writers. A second read below ensures one consistent checkpoint.
        from contextlib import ExitStack

        from atomic_model_workspace.common import Lock

        with ExitStack() as stack:
            stack.enter_context(Lock(campaign))
            state = load_campaign(campaign)
            if state["project"] != project:
                raise ValueError("Bundle project differs from campaign")
            for entry in state["entries"]:
                stack.enter_context(Lock(campaign / entry["job"]))
            files["campaign/campaign.json"] = _encoded(state)
            for marker in ("CANCEL.json", "PAUSE.json"):
                if (campaign / marker).exists():
                    files["campaign/" + marker] = (campaign / marker).read_bytes()
            for entry in state["entries"]:
                folder = campaign / entry["job"]
                load_job(folder)
                head = read(folder / "HEAD.json")
                for marker in ("CANCEL.json", "PAUSE.json"):
                    if (folder / marker).exists():
                        files[f"campaign/{entry['job']}/{marker}"] = (folder / marker).read_bytes()
                files[f"campaign/{entry['job']}/HEAD.json"] = _encoded(head)
                files[f"campaign/{entry['job']}/{head['checkpoint']}"] = (
                    folder / head["checkpoint"]
                ).read_bytes()
            files["report.json"] = _encoded(campaign_report(campaign))
    if (
        len(files) > MAX_FILES
        or sum(map(len, files.values())) > MAX_TOTAL_BYTES
        or any(len(v) > MAX_FILE_BYTES for v in files.values())
    ):
        raise ValueError("Bundle resource limits")
    manifest = dict(
        schema_version=1,
        format="atomic-workspace-data-v1",
        project_sha256=digest(project),
        contains_campaign=campaign is not None,
        artifacts={
            name: dict(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            for name, data in sorted(files.items())
        },
        authenticated=False,
        validated=False,
        plant_referenced=False,
    )
    files["MANIFEST.json"] = _encoded(manifest)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temp = Path(stream.name)
            with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as z:
                for name, data in sorted(files.items()):
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.external_attr = (stat.S_IFREG | 0o644) << 16
                    z.writestr(info, data)
        verify_bundle(temp)
        os.link(temp, output)  # No replacement of an existing archive.
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)
    return dict(
        archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        manifest_sha256=digest(manifest),
        files=len(files),
    )


def _archive_files(path):
    path = Path(path)
    if path.stat().st_size > MAX_TOTAL_BYTES + 4 * 1024 * 1024:
        raise ValueError("Archive byte limit")
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            if not 1 <= len(infos) <= MAX_FILES + 1:
                raise ValueError("Archive file count limit")
            names = set()
            total = 0
            for info in infos:
                portable(info.filename)
                name = info.filename.casefold()
                mode = info.external_attr >> 16
                if (
                    info.is_dir()
                    or stat.S_ISLNK(mode)
                    or (stat.S_IFMT(mode) not in (0, stat.S_IFREG))
                    or info.flag_bits & 1
                ):
                    raise ValueError("Archive must contain regular unencrypted files only")
                if name in names or any(
                    name.startswith(old + "/") or old.startswith(name + "/") for old in names
                ):
                    raise ValueError("Duplicate/colliding archive path")
                names.add(name)
                total += info.file_size
                if info.file_size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES:
                    raise ValueError("Archive expanded size limit")
            return {info.filename: z.read(info) for info in infos}
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ValueError("Malformed/unsupported workspace archive") from exc


def verify_bundle(path):
    files = _archive_files(path)
    if "MANIFEST.json" not in files:
        raise ValueError("Bundle manifest missing")
    manifest = loads(files.pop("MANIFEST.json"))
    keys(
        manifest,
        (
            "schema_version",
            "format",
            "project_sha256",
            "contains_campaign",
            "artifacts",
            "authenticated",
            "validated",
            "plant_referenced",
        ),
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["format"] != "atomic-workspace-data-v1"
        or type(manifest["contains_campaign"]) is not bool
        or any(manifest[k] is not False for k in ("authenticated", "validated", "plant_referenced"))
        or not isinstance(manifest["artifacts"], dict)
        or set(files) != set(manifest["artifacts"])
    ):
        raise ValueError("Unknown bundle schema or inventory mismatch")
    for name, data in files.items():
        expected = manifest["artifacts"][name]
        if expected != dict(size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()):
            raise ValueError("Bundle artifact hash/size mismatch")
        # Only data belonging to this exact format can be materialized.
        if not (
            name == "report.json"
            or name.startswith("project/")
            or (manifest["contains_campaign"] and name.startswith("campaign/"))
        ) or not name.endswith(".json"):
            raise ValueError("Unexpected bundle payload")
        loads(data)
    # Semantic/source checks in disposable storage, never in the destination.
    with tempfile.TemporaryDirectory(prefix="atomic-workspace-verify-") as tmp:
        root = Path(tmp)
        for name, data in files.items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        project = load_project(root / "project/project.json")
        if digest(project) != manifest["project_sha256"]:
            raise ValueError("Bundle project identity mismatch")
        if manifest["contains_campaign"]:
            state = load_campaign(root / "campaign")
            if state["project"] != project or loads(
                files.get("report.json", b"null")
            ) != campaign_report(root / "campaign"):
                raise ValueError("Bundle campaign/report identity mismatch")
        elif "report.json" in files or any(n.startswith("campaign/") for n in files):
            raise ValueError("Unexpected campaign data")
    return dict(manifest=manifest, files=files)


def import_bundle(path, output):
    verified = verify_bundle(path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        for name, data in verified["files"].items():
            p = output / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        (output / "MANIFEST.json").write_bytes(_encoded(verified["manifest"]))
    except BaseException:
        shutil.rmtree(output)
        raise
    return dict(
        project=str(output / "project/project.json"),
        campaign=str(output / "campaign") if verified["manifest"]["contains_campaign"] else None,
        manifest_sha256=digest(verified["manifest"]),
    )
