"""Portable data-only reference bundles and conservative split leakage checks."""

import hashlib
import shutil
from copy import deepcopy
from pathlib import Path

from atomic_model_reference.data import load_input
from atomic_model_workspace.common import (
    atomic_json,
    canonical,
    digest,
    keys,
    local_file,
    portable,
    read,
)
from atomic_model_workspace.project import provenance


def source_identity():
    return dict(
        reference_code_sha256=digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        ),
        workspace=provenance(),
    )


def import_reference(path, output):
    spec, raw, reference = load_input(path)
    portable = deepcopy(spec)
    portable["csv"]["path"] = "data.csv"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        (output / "data.csv").write_bytes(raw)
        atomic_json(output / "source_input.json", spec)
        atomic_json(output / "input.json", portable)
        atomic_json(output / "reference.json", reference)
        atomic_json(
            output / "manifest.json",
            dict(
                schema_version=1,
                format="atomic-reference-data-v1",
                original_input_sha256=digest(spec),
                normalized_reference_sha256=digest(reference),
                importer_source=source_identity(),
                artifacts={
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in output.iterdir()
                    if p.is_file()
                },
                authenticated=False,
                validated=False,
                plant_referenced=False,
            ),
        )
        load_bundle(output)
    except BaseException:
        shutil.rmtree(output)
        raise
    return reference


def load_bundle(folder):
    folder = Path(folder)
    manifest = read(local_file(folder, "manifest.json"))
    keys(
        manifest,
        (
            "schema_version",
            "format",
            "original_input_sha256",
            "normalized_reference_sha256",
            "importer_source",
            "artifacts",
            "authenticated",
            "validated",
            "plant_referenced",
        ),
    )
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["format"] != "atomic-reference-data-v1"
        or any(manifest[k] is not False for k in ("authenticated", "validated", "plant_referenced"))
        or set(manifest["artifacts"])
        != {"input.json", "reference.json", "source_input.json", "data.csv"}
    ):
        raise ValueError("Unsupported reference bundle schema/inventory")
    for name, sha in manifest["artifacts"].items():
        p = local_file(folder, name)
        with p.open("rb") as stream:
            data = stream.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("Reference artifact bytes/hash mismatch")
    source = read(local_file(folder, "source_input.json"))
    if digest(source) != manifest["original_input_sha256"]:
        raise ValueError("Original canonical input identity mismatch")
    keys(source["csv"], ("path", "sha256", "time_column", "time_unit"))
    portable(source["csv"]["path"])
    expected = deepcopy(source)
    expected["csv"]["path"] = "data.csv"
    if canonical(expected) != canonical(read(local_file(folder, "input.json"))):
        raise ValueError("Portable declaration changed original metadata")
    spec, raw, parsed = load_input(local_file(folder, "input.json"))
    if spec["csv"]["path"] != "data.csv":
        raise ValueError("Bundle data must remain inside inventory")
    stored = read(local_file(folder, "reference.json"))
    if (
        canonical(parsed) != canonical(stored)
        or digest(stored) != manifest["normalized_reference_sha256"]
    ):
        raise ValueError("Normalized reference disagrees with raw data/schema")
    return stored, manifest


def check_catalog(folders):
    if not isinstance(folders, list) or not 1 <= len(folders) <= 256:
        raise ValueError("Reference catalog count limit")
    seen = set()
    physical = {}
    conditions = {}
    rows = []
    raw_roles = {}
    curve_roles = {}
    for folder in folders:
        ref, manifest = load_bundle(folder)
        if ref["reference_id"] in seen:
            raise ValueError("Duplicate reference ID in catalog")
        seen.add(ref["reference_id"])
        for mapping, key in (
            (physical, ref["physical_case_id"]),
            (conditions, ref["physical_conditions_sha256"]),
        ):
            if key in mapping and mapping[key] != ref["role"]:
                raise ValueError(
                    "Physical case/conditions leak across calibration, validation or test"
                )
            mapping[key] = ref["role"]
        if ref["csv_sha256"] in raw_roles and raw_roles[ref["csv_sha256"]] != ref["role"]:
            raise ValueError("Raw reference data leakage across roles")
        raw_roles[ref["csv_sha256"]] = ref["role"]
        curve_hashes = []
        for name, channel in ref["channels"].items():
            curve = digest(
                dict(
                    times_s=ref["times_s"],
                    unit=channel["canonical_unit"],
                    values=ref["values"][name],
                )
            )
            if curve in curve_roles and curve_roles[curve] != ref["role"]:
                raise ValueError("Normalized reference curve leakage across roles")
            curve_roles[curve] = ref["role"]
            curve_hashes.append(curve)
        rows.append(
            dict(
                normalized_curve_sha256=curve_hashes,
                reference_id=ref["reference_id"],
                physical_case_id=ref["physical_case_id"],
                physical_conditions_sha256=ref["physical_conditions_sha256"],
                role=ref["role"],
                kind=ref["kind"],
                normalized_reference_sha256=manifest["normalized_reference_sha256"],
            )
        )
    return dict(
        schema_version=1,
        accepted=True,
        references=rows,
        split_guard="physical_case_id, normalized physical conditions, raw CSV and exact canonical-unit curves; changing numeric step, horizon, labels or uncertainty does not create held-out data",
        validated=False,
        plant_referenced=False,
    )
