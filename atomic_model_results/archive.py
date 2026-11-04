"""Deterministic ZIP transport with bounded admission before output creation."""

import os
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path

from atomic_model_results.bundle import MAX_FILE, MAX_FILES, MAX_TOTAL, verify
from atomic_model_workspace.common import local_file, portable, read


def pack(folder, output):
    folder, output = Path(folder), Path(output)
    summary = verify(folder)
    manifest = read(local_file(folder, "manifest.json"))
    names = sorted([*manifest["artifacts"], "manifest.json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as archive:
                for name in names:
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = (stat.S_IFREG | 0o644) << 16
                    info.compress_type = zipfile.ZIP_DEFLATED
                    archive.writestr(info, local_file(folder, name).read_bytes(), compresslevel=9)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(
        schema_version=1,
        archive=str(output),
        evaluations=len(summary["entries"]),
        validated=False,
        plant_referenced=False,
    )


def unpack(archive, output):
    archive, output = Path(archive), Path(output)
    if archive.stat().st_size > MAX_TOTAL:
        raise ValueError("Compressed results archive size limit")
    with tempfile.TemporaryDirectory(prefix="atomic-result-import-") as tmp:
        stage = Path(tmp)
        with zipfile.ZipFile(archive) as source:
            infos = source.infolist()
            if not 1 <= len(infos) <= MAX_FILES + 1:
                raise ValueError("Archive file count limit")
            names = set()
            folded = set()
            total = 0
            for info in infos:
                name = portable(info.filename)
                total += info.file_size
                kind = stat.S_IFMT(info.external_attr >> 16)
                if (
                    info.orig_filename != info.filename
                    or name in names
                    or name.casefold() in folded
                    or kind not in (0, stat.S_IFREG)
                    or info.is_dir()
                    or info.flag_bits & 1
                    or info.compress_type not in (zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED)
                    or info.file_size > MAX_FILE
                    or total > MAX_TOTAL
                ):
                    raise ValueError("Unsafe or oversized result archive member")
                names.add(name)
                folded.add(name.casefold())
            if "manifest.json" not in names:
                raise ValueError("Archive manifest missing")
            for info in infos:
                with source.open(info) as stream:
                    raw = stream.read(MAX_FILE + 1)
                if len(raw) != info.file_size or len(raw) > MAX_FILE:
                    raise ValueError("Archive member size mismatch")
                target = stage / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
        result = verify(stage)
        output.mkdir(parents=True, exist_ok=False)
        try:
            for name in sorted(names):
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(local_file(stage, name), target)
        except BaseException:
            shutil.rmtree(output)
            raise
    return dict(
        schema_version=1,
        evaluations=len(result["entries"]),
        replay_exact=True,
        validated=False,
        plant_referenced=False,
    )
