"""Distribution-anchored resources with canonical, machine-independent names."""

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def resolve(path):
    p = Path(os.fspath(path))
    if (
        not p.is_absolute()
        and p.parts
        and (
            p.parts[0] == "atomic_model_runtime" or (p.parts[0] == "scripts" and p.suffix == ".py")
        )
    ):
        return ROOT / p
    return p


@dataclass(frozen=True)
class RuntimeFile(os.PathLike):
    name: str

    def __str__(self):
        return self.name

    def __fspath__(self):
        return str(ROOT / self.name)

    def read_text(self, *args, **kwargs):
        return Path(os.fspath(self)).read_text(*args, **kwargs)

    def read_bytes(self):
        return Path(os.fspath(self)).read_bytes()


def resource(name):
    p = Path(name)
    if (
        p.is_absolute()
        or ".." in p.parts
        or not p.parts
        or p.parts[0] not in {"scripts", "validation", "atomic_model_runtime"}
    ):
        raise ValueError("Invalid distribution resource name")
    return RuntimeFile(p.as_posix())
