"""Deterministic, offline evidence archives; verification never executes payloads.

This proves byte integrity and, optionally, wheel/sdist source identity. It does
not recompute numerical acceptance, authenticate a publisher or resolve a lock.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import re
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from importlib import metadata
from pathlib import Path, PurePosixPath

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
RESERVED = {"MANIFEST.json", "INVENTORY.json"}


@dataclass(frozen=True)
class Limits:
    max_files: int = 10000
    max_file_bytes: int = 128 * 1024 * 1024
    max_total_bytes: int = 2 * 1024 * 1024 * 1024
    max_manifest_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in vars(self).values()):
            raise ValueError("Archive limits must be positive integers")


def bounded_tarinfo(limits):
    """Bound raw PAX/GNU headers before tarfile reads extension payloads."""

    class BoundedTarInfo(tarfile.TarInfo):
        total_bytes = 0
        headers = 0

        @classmethod
        def frombuf(cls, buf, encoding, errors):
            info = super().frombuf(buf, encoding, errors)
            cls.headers += 1
            cls.total_bytes += 512 + ((info.size + 511) // 512) * 512
            extension = info.type in (
                tarfile.XHDTYPE,
                tarfile.XGLTYPE,
                tarfile.GNUTYPE_LONGNAME,
                tarfile.GNUTYPE_LONGLINK,
            )
            cap = (
                min(limits.max_manifest_bytes, limits.max_file_bytes)
                if extension
                else limits.max_file_bytes
            )
            if (
                info.size < 0
                or info.size > cap
                or cls.headers > limits.max_files * 3 + 8
                or cls.total_bytes > limits.max_total_bytes + (limits.max_files * 6 + 16) * 512
            ):
                raise ValueError("Raw tar header resource limit exceeded")
            return info

    return BoundedTarInfo


def canonical(value):
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode()


def safe_name(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("A portable relative POSIX path is required")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or path.as_posix() != value:
        raise ValueError("Noncanonical or absolute archive path")
    devices = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *[f"COM{i}" for i in range(1, 10)],
        *[f"LPT{i}" for i in range(1, 10)],
    }
    if any(
        part in (".", "..")
        or part.endswith((".", " "))
        or part.split(".")[0].upper() in devices
        or any(ord(c) < 32 or c in '<>"|?*' for c in part)
        for part in path.parts
    ):
        raise ValueError("Unsafe or nonportable archive path")
    return value


def _digest(stream, size, limit):
    if type(size) is not int or size < 0 or size > limit:
        raise ValueError("Archive member size limit exceeded")
    digest, count = hashlib.sha256(), 0
    while chunk := stream.read(min(1024 * 1024, size - count + 1)):
        count += len(chunk)
        if count > size:
            raise ValueError("Archive member exceeds declared size")
        digest.update(chunk)
    if count != size:
        raise ValueError("Truncated archive member")
    return digest.hexdigest()


def _source_path(root, name):
    safe_name(name)
    current = root
    for part in PurePosixPath(name).parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Source symlinks are not bundled")
    if not current.is_relative_to(root) or not current.exists():
        raise ValueError("Missing source inside the selected root")
    return current


def _selection(selection, root, limits):
    required = {"schema_version", "bundle_id", "scope", "entries"}
    if (
        not isinstance(selection, dict)
        or not required <= set(selection)
        or set(selection) - required - {"metadata", "runtime"}
        or type(selection["schema_version"]) is not int
        or selection["schema_version"] != 1
        or not isinstance(selection["bundle_id"], str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", selection["bundle_id"])
        or not isinstance(selection["scope"], str)
        or not selection["scope"].strip()
        or not isinstance(selection["entries"], list)
        or not selection["entries"]
    ):
        raise ValueError("Invalid evidence selection schema 1")
    canonical(selection)
    files, folded, total = {}, set(), 0
    for entry in selection["entries"]:
        if not isinstance(entry, dict) or set(entry) != {"source", "path"}:
            raise ValueError("Each selection entry requires source and path")
        source = _source_path(root, entry["source"])
        prefix = safe_name(entry["path"])
        candidates = sorted(source.rglob("*")) if source.is_dir() else [source]
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError("Source symlinks are not bundled")
            if candidate.is_dir():
                continue
            if not candidate.is_file():
                raise ValueError("Only regular evidence files are supported")
            name = safe_name(
                prefix
                if source.is_file()
                else prefix + "/" + candidate.relative_to(source).as_posix()
            )
            if name.upper() in RESERVED or name.casefold() in folded:
                raise ValueError("Duplicate or reserved archive destination")
            # A file must never also be an ancestor directory of another file.
            if any(
                name.casefold().startswith(old + "/") or old.startswith(name.casefold() + "/")
                for old in folded
            ):
                raise ValueError("Archive file/directory destination collision")
            size = candidate.stat().st_size
            total += size
            if (
                size > limits.max_file_bytes
                or total > limits.max_total_bytes
                or len(files) + 3 > limits.max_files
            ):
                raise ValueError("Selected evidence exceeds archive limits")
            files[name] = candidate
            folded.add(name.casefold())
    if not files:
        raise ValueError("Empty evidence selection")
    return files


def runtime_fingerprint(data, kind, limits=Limits()):
    """Read nested runtime sources without extraction or execution."""
    sources, total, nested_total, nested_count = {}, 0, 0, 0

    def bound_member(size):
        nonlocal nested_total, nested_count
        nested_total += size
        nested_count += 1
        if (
            size < 0
            or size > limits.max_file_bytes
            or nested_count > limits.max_files
            or nested_total > limits.max_total_bytes
        ):
            raise ValueError("Nested archive resource limit exceeded")

    def accept(name, size, stream):
        nonlocal total
        parts = PurePosixPath(name).parts
        if (
            not parts
            or parts[0] not in PACKAGES
            or PurePosixPath(name).suffix not in (".py", ".json", ".yaml")
        ):
            return
        safe_name(name)
        total += size
        if (
            name in sources
            or size > limits.max_file_bytes
            or total > limits.max_total_bytes
            or len(sources) >= limits.max_files
        ):
            raise ValueError("Invalid or oversized nested runtime")
        content = stream.read(size + 1)
        if len(content) != size:
            raise ValueError("Truncated nested runtime source")
        sources[name] = content

    if kind == "wheel":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.infolist():
                if not member.is_dir():
                    bound_member(member.file_size)
                    with archive.open(member) as stream:
                        accept(member.filename, member.file_size, stream)
    elif kind == "sdist":
        with tarfile.open(
            fileobj=io.BytesIO(data), mode="r:*", tarinfo=bounded_tarinfo(limits)
        ) as archive:
            roots = set()
            for member in archive:
                if member.isdir():
                    continue
                bound_member(member.size)
                safe_name(member.name)
                parts = PurePosixPath(member.name).parts
                if len(parts) < 2 or not member.isfile():
                    raise ValueError("Invalid sdist member")
                roots.add(parts[0])
                if len(roots) != 1:
                    raise ValueError("Multiple sdist roots")
                with archive.extractfile(member) as stream:
                    accept(PurePosixPath(*parts[1:]).as_posix(), member.size, stream)
    else:
        raise ValueError("Unknown runtime archive kind")
    if {PurePosixPath(p).parts[0] for p in sources} != set(PACKAGES):
        raise ValueError("Incomplete runtime source packages")
    digest = hashlib.sha256()
    for package in PACKAGES:
        for name in sorted(
            (p for p in sources if p.startswith(package + "/")),
            key=lambda p: PurePosixPath(p).parts,
        ):
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(sources[name])
    return digest.hexdigest()


def _inventory(files, runtime):
    result = dict(
        schema_version=1,
        packaging_environment=dict(python=platform.python_version(), platform=platform.platform()),
        scope="Declared wheel requirements and observed packaging environment; "
        "not a dependency lock, SPDX/CycloneDX attestation or legal license review",
        distributions=[],
        project=None,
    )
    if not runtime:
        return result
    with zipfile.ZipFile(files[runtime["wheel"]]) as archive:
        names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            raise ValueError("Wheel requires one project METADATA")
        project = BytesParser().parsebytes(archive.read(names[0]))
    result["project"] = dict(
        name=project.get("Name"),
        version=project.get("Version"),
        license_metadata=project.get("License"),
        requires_python=project.get("Requires-Python"),
    )
    for requirement in sorted(project.get_all("Requires-Dist", [])):
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9_.-]*", requirement)
        if not match:
            raise ValueError("Invalid declared distribution requirement")
        name = match.group()
        try:
            distribution = metadata.distribution(name)
            observed = dict(
                version=distribution.version,
                license_metadata=distribution.metadata.get("License"),
                license_expression=distribution.metadata.get("License-Expression"),
            )
        except metadata.PackageNotFoundError:
            observed = dict(version=None, license_metadata=None, license_expression=None)
        result["distributions"].append(dict(requirement=requirement, name=name, observed=observed))
    return result


def _runtime(runtime, files, limits):
    if runtime is None:
        return
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"wheel", "sdist", "source_sha256"}
        or not isinstance(runtime["source_sha256"], str)
        or not re.fullmatch("[0-9a-f]{64}", runtime["source_sha256"])
    ):
        raise ValueError("Invalid runtime identity")
    for kind in ("wheel", "sdist"):
        name = safe_name(runtime[kind])
        if name not in files:
            raise ValueError("Runtime archive missing from selection")
        source = files[name]
        if isinstance(source, Path):
            with source.open("rb") as stream:
                data = stream.read(limits.max_file_bytes + 1)
        else:
            data = source
        if len(data) > limits.max_file_bytes:
            raise ValueError("Nested runtime exceeds size limit")
        if runtime_fingerprint(data, kind, limits) != runtime["source_sha256"]:
            raise ValueError("Runtime source fingerprint mismatch")


def _info(name, size):
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = size, 0o644, 0
    return info


def create_bundle(selection, root, output, limits=Limits()):
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    files = _selection(selection, root, limits)
    if output in (p.absolute() for p in files.values()):
        raise ValueError("Output overlaps selected evidence")
    runtime = selection.get("runtime")
    _runtime(runtime, files, limits)
    inventory = canonical(_inventory(files, runtime))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as raw:
            temporary = Path(raw.name)
            records = []
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as archive:
                    for name in sorted([*files, "INVENTORY.json"]):
                        if name == "INVENTORY.json":
                            data = inventory
                        else:
                            path = files[name]
                            size = path.stat().st_size
                            if size > limits.max_file_bytes:
                                raise ValueError("Source grew beyond size limit")
                            data = path.read_bytes()
                            if len(data) != size:
                                raise ValueError("Source changed during collection")
                        records.append(
                            dict(
                                path=name,
                                size_bytes=len(data),
                                sha256=hashlib.sha256(data).hexdigest(),
                            )
                        )
                        archive.addfile(_info(name, len(data)), io.BytesIO(data))
                    manifest = dict(
                        schema_version=1,
                        format="atomic-evidence-bundle",
                        bundle_id=selection["bundle_id"],
                        scope=selection["scope"],
                        metadata=selection.get("metadata", {}),
                        runtime=runtime,
                        files=records,
                        numerical_acceptance_recomputed=False,
                        publisher_authenticated=False,
                    )
                    encoded = canonical(manifest)
                    if len(encoded) > limits.max_manifest_bytes:
                        raise ValueError("Manifest size limit exceeded")
                    archive.addfile(_info("MANIFEST.json", len(encoded)), io.BytesIO(encoded))
        # Verify the actual collected bytes, including runtime identity, before
        # publishing. A hard link publishes without replacing an existing file.
        result = verify_bundle(temporary, limits=limits)
        os.link(temporary, output)
        return dict(**result, archive=str(output))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _manifest(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate manifest JSON key")
            result[key] = value
        return result

    result = json.loads(
        data,
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )
    required = {
        "schema_version",
        "format",
        "bundle_id",
        "scope",
        "metadata",
        "runtime",
        "files",
        "numerical_acceptance_recomputed",
        "publisher_authenticated",
    }
    if (
        not isinstance(result, dict)
        or set(result) != required
        or type(result["schema_version"]) is not int
        or result["schema_version"] != 1
        or result["format"] != "atomic-evidence-bundle"
        or result["numerical_acceptance_recomputed"] is not False
        or result["publisher_authenticated"] is not False
        or not isinstance(result["bundle_id"], str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", result["bundle_id"])
        or not isinstance(result["scope"], str)
        or not result["scope"].strip()
        or not isinstance(result["metadata"], dict)
        or not isinstance(result["files"], list)
    ):
        raise ValueError("Invalid bundle manifest schema")
    return result


def verify_bundle(path, expected_manifest_sha256=None, limits=Limits()):
    path = Path(path)
    before = path.stat()
    if before.st_size > limits.max_total_bytes:
        raise ValueError("Compressed archive resource limit exceeded")
    with path.open("rb") as stream:
        archive_sha256 = _digest(stream, before.st_size, limits.max_total_bytes)
    actual, folded, total, manifest_bytes = {}, set(), 0, None
    with tarfile.open(path, "r:gz", tarinfo=bounded_tarinfo(limits)) as archive:
        for member in archive:
            name = safe_name(member.name)
            key = name.casefold()
            if (
                not member.isfile()
                or key in folded
                or len(folded) >= limits.max_files
                or any(key.startswith(old + "/") or old.startswith(key + "/") for old in folded)
            ):
                raise ValueError("Nonregular, duplicate or colliding archive member")
            folded.add(key)
            total += member.size
            cap = limits.max_manifest_bytes if name == "MANIFEST.json" else limits.max_file_bytes
            if member.size > cap or total > limits.max_total_bytes:
                raise ValueError("Archive resource limit exceeded")
            with archive.extractfile(member) as stream:
                if name == "MANIFEST.json":
                    data = stream.read(cap + 1)
                    if len(data) != member.size:
                        raise ValueError("Truncated or oversized member")
                    digest = hashlib.sha256(data).hexdigest()
                    manifest_bytes = data
                else:
                    digest = _digest(stream, member.size, cap)
            if name != "MANIFEST.json":
                actual[name] = dict(path=name, size_bytes=member.size, sha256=digest)
    if manifest_bytes is None:
        raise ValueError("Missing evidence manifest")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_manifest_sha256 is not None and expected_manifest_sha256 != manifest_sha:
        raise ValueError("Trusted manifest checksum mismatch")
    manifest = _manifest(manifest_bytes)
    declared = {}
    for record in manifest["files"]:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "size_bytes", "sha256"}
            or type(record["size_bytes"]) is not int
            or record["size_bytes"] < 0
            or not isinstance(record["sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", record["sha256"])
        ):
            raise ValueError("Invalid evidence file record")
        name = safe_name(record["path"])
        if name in declared:
            raise ValueError("Duplicate manifest file")
        declared[name] = record
    if declared != actual or "INVENTORY.json" not in declared:
        raise ValueError("Evidence set, size or checksum mismatch")
    if manifest["runtime"] is not None:
        nested = {}
        runtime = manifest["runtime"]
        if not isinstance(runtime, dict) or set(runtime) != {"wheel", "sdist", "source_sha256"}:
            raise ValueError("Invalid runtime identity")
        with tarfile.open(path, "r:gz", tarinfo=bounded_tarinfo(limits)) as archive:
            for kind in ("wheel", "sdist"):
                name = safe_name(runtime[kind])
                if name not in actual:
                    raise ValueError("Missing runtime archive")
                with archive.extractfile(archive.getmember(name)) as stream:
                    data = stream.read(limits.max_file_bytes + 1)
                if (
                    len(data) != actual[name]["size_bytes"]
                    or hashlib.sha256(data).hexdigest() != actual[name]["sha256"]
                ):
                    raise ValueError("Runtime archive changed during verification")
                nested[name] = data
        _runtime(runtime, nested, limits)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Archive changed during verification")
    return dict(
        schema_version=1,
        integrity_verified=True,
        archive_sha256=archive_sha256,
        manifest_sha256=manifest_sha,
        files=len(actual),
        uncompressed_bytes=total,
        runtime_source_verified=manifest["runtime"] is not None,
        source_sha256=manifest["runtime"]["source_sha256"] if manifest["runtime"] else None,
        numerical_acceptance_recomputed=False,
        publisher_authenticated=False,
    )


def extract_bundle(path, output, expected_manifest_sha256=None, limits=Limits()):
    result = verify_bundle(path, expected_manifest_sha256, limits)
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    try:
        # Recheck each extracted byte: the input may have changed since verify.
        with tarfile.open(path, "r:gz", tarinfo=bounded_tarinfo(limits)) as archive:
            member = archive.getmember("MANIFEST.json")
            data = archive.extractfile(member).read(limits.max_manifest_bytes + 1)
            if hashlib.sha256(data).hexdigest() != result["manifest_sha256"]:
                raise ValueError("Manifest changed before extraction")
            expected = {r["path"]: r for r in _manifest(data)["files"]}
            expected["MANIFEST.json"] = dict(size_bytes=len(data), sha256=result["manifest_sha256"])
            seen = set()
            for member in archive.getmembers():
                name = safe_name(member.name)
                if name not in expected or name in seen or not member.isfile():
                    raise ValueError("Member set changed before extraction")
                seen.add(name)
                record = expected[name]
                if member.size != record["size_bytes"]:
                    raise ValueError("Member size changed before extraction")
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream, target.open("xb") as destination:
                    digest, count = hashlib.sha256(), 0
                    while chunk := stream.read(min(1024 * 1024, member.size - count + 1)):
                        count += len(chunk)
                        if count > member.size:
                            raise ValueError("Member grew during extraction")
                        digest.update(chunk)
                        destination.write(chunk)
                if count != member.size or digest.hexdigest() != record["sha256"]:
                    raise ValueError("Extracted member checksum mismatch")
            if seen != set(expected):
                raise ValueError("Missing extracted evidence")
        return dict(**result, output=str(output))
    except BaseException:
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("selection", type=Path)
    create.add_argument("--root", type=Path, default=Path.cwd())
    create.add_argument("--output", type=Path, required=True)
    for name in ("verify", "extract"):
        command = commands.add_parser(name)
        command.add_argument("archive", type=Path)
        command.add_argument("--manifest-sha256")
        if name == "extract":
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        result = create_bundle(json.loads(args.selection.read_text()), args.root, args.output)
    elif args.command == "verify":
        result = verify_bundle(args.archive, args.manifest_sha256)
    else:
        result = extract_bundle(args.archive, args.output, args.manifest_sha256)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
