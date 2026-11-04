"""Offline exact replay of data/metrics with the same declared comparator source."""

import hashlib
import shutil
import tempfile
from pathlib import Path

from atomic_model_reference.bundle import import_reference, source_identity
from atomic_model_reference.compare import compare_job
from atomic_model_workspace.common import atomic_json, canonical, digest, keys, local_file, read

FILES = {
    "report.json",
    "reference_snapshot.json",
    "reference_input.json",
    "data.csv",
    "job_snapshot.json",
}


def replay_report(folder, output):
    folder = Path(folder)
    manifest = read(local_file(folder, "manifest.json"))
    keys(manifest, ("schema_version", "source", "artifacts", "validated", "plant_referenced"))
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or set(manifest["artifacts"]) != FILES
        or manifest["source"] != source_identity()
        or manifest["validated"] is not False
        or manifest["plant_referenced"] is not False
    ):
        raise ValueError("Report schema/source/inventory mismatch")
    for name, sha in manifest["artifacts"].items():
        with local_file(folder, name).open("rb") as stream:
            data = stream.read(32 * 1024 * 1024 + 1)
        if len(data) > 32 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError("Report artifact bytes/hash mismatch")
    expected = read(local_file(folder, "report.json"))
    state = read(local_file(folder, "job_snapshot.json"))
    if expected["source"] != manifest["source"]:
        raise ValueError("Comparison source mismatch")
    with tempfile.TemporaryDirectory(prefix="atomic-reference-replay-") as tmp:
        root = Path(tmp)
        job = root / "job"
        (job / "revisions").mkdir(parents=True)
        atomic_json(job / "revisions/replay.json", dict(payload=state, sha256=digest(state)))
        atomic_json(
            job / "HEAD.json",
            dict(schema_version=1, checkpoint="revisions/replay.json", sha256=digest(state)),
        )
        if read(local_file(folder, "reference_input.json"))["csv"]["path"] != "data.csv":
            raise ValueError("Report reference points outside declared data inventory")
        ref = import_reference(local_file(folder, "reference_input.json"), root / "reference")
        if canonical(ref) != canonical(read(local_file(folder, "reference_snapshot.json"))):
            raise ValueError("Reference snapshot mismatch")
        result = compare_job(job, root / "reference", root / "result")
        if canonical(result) != canonical(expected):
            raise ValueError("Recomputed reference metrics differ from report")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        try:
            for p in (root / "result").iterdir():
                shutil.copy2(p, output / p.name)
        except BaseException:
            shutil.rmtree(output)
            raise
    return dict(
        schema_version=1,
        replay_exact=True,
        comparison_passed=result["passed"],
        reference_id=result["reference_id"],
        validated=False,
        plant_referenced=False,
    )
