"""Bounded JSON, portable references and atomic local checkpoint publication."""

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path, PurePosixPath

MAX_JSON_BYTES = 32 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Missing or unknown fields: expected " + ", ".join(expected))


def number(value, low, high, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} outside finite range [{low}, {high}]")
    return float(value)


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("Expected portable identifier")
    return value


def portable(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("Expected portable relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(p in (".", "..") for p in path.parts):
        raise ValueError("Unsafe or noncanonical relative path")
    return value


def local_file(root, reference):
    portable(reference)
    root = Path(root).resolve()
    current = root
    for part in PurePosixPath(reference).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Project symlinks are not supported")
    if not current.is_file() or not current.resolve().is_relative_to(root):
        raise ValueError("Project reference is missing or outside root")
    return current


def loads(data):
    if len(data.encode() if isinstance(data, str) else data) > MAX_JSON_BYTES:
        raise ValueError("JSON size limit")

    def pairs(items):
        value = {}
        for k, v in items:
            if k in value:
                raise ValueError("Duplicate JSON key")
            value[k] = v
        return value

    def invalid(value):
        raise ValueError("Nonfinite JSON number: " + value)

    try:
        result = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
        canonical(result)  # Also rejects overflowing exponents, e.g. 1e999.
        return result
    except (RecursionError, UnicodeError) as exc:
        raise ValueError("Malformed JSON") from exc


def read(path):
    with Path(path).open("rb") as stream:
        return loads(stream.read(MAX_JSON_BYTES + 1))


def atomic_json(path, value):
    """Publish a whole file; callers serialize writers with the job/campaign lock."""
    path = Path(path)
    data = (canonical(value) + "\n").encode()
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("JSON size limit")
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            name = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name == "posix":
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


class Lock:
    def __init__(self, folder):
        self.path = Path(folder) / ".lock"

    def __enter__(self):
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise ValueError(
                "Another writer or stale lock; inspect owner before manual recovery"
            ) from exc
        os.write(self.fd, str(os.getpid()).encode())
        return self

    def __exit__(self, *args):
        os.close(self.fd)
        self.path.unlink()
