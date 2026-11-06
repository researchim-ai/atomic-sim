"""Pinned licensed transient traces; each modality keeps its own native clock."""

import csv
import gzip
import json
from importlib.resources import files

from .importer import RAW_PIN, sha

PIN = "1c362da413828e543c9fdf0b9c4f9d76aa0f50f88f8ecf0267c5c7ee9d57e359"


def load():
    root = files("atomic_model_phase_assessment").joinpath("data/pool_b274")
    manifest_path = root.joinpath("manifest.json")
    if sha(manifest_path) != PIN:
        raise ValueError("Pool dataset manifest changed")
    manifest = json.loads(manifest_path.read_text())
    expected = {
        "PRIMARY_README.md",
        "LICENSE",
        "data_dictionary.csv",
        "event_definition.md",
        "CITATION.cff",
        "NOTICE.md",
        "pressure.csv.gz",
        "thermal.csv.gz",
    }
    if set(manifest["files"]) != expected or {p.name for p in root.iterdir()} != expected | {
        "manifest.json"
    }:
        raise ValueError("Pool dataset inventory mismatch")
    from . import importer

    if manifest["parser_sha256"] != sha(importer.__file__):
        raise ValueError("Primary importer source changed")
    if manifest["primary_sha256"] != RAW_PIN:
        raise ValueError("Primary identity changed")
    for name, digest in manifest["files"].items():
        if sha(root.joinpath(name)) != digest:
            raise ValueError("Pool dataset file changed:" + name)
    result = {}
    for kind in ["pressure", "thermal"]:
        with gzip.open(root.joinpath(kind + ".csv.gz"), "rt", newline="") as stream:
            result[kind] = [
                {
                    k: (int(v) if k == "source_row" else float(v) if v else None)
                    for k, v in row.items()
                }
                for row in csv.DictReader(stream)
            ]
        if len(result[kind]) != manifest["retained_counts"][kind]:
            raise ValueError("Trace row count changed")
    return manifest, result
