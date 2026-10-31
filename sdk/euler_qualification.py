"""Conservative admission gate for exact, hashed qualification fixtures.

Admission is not extrapolated to untested configurations, grids or horizons.
This is an input-scope gate, not OS isolation or experimental validation.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from core.status import Applicability, DomainError
from sdk.experiment import _source_fingerprint


class EulerQualification:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        manifest = json.loads((self.folder / "manifest.json").read_text())
        self.hashes = manifest["sha256"]
        if "qualification.json" not in self.hashes:
            raise ValueError("Qualification matrix must be checksummed")
        for name, digest in manifest["sha256"].items():
            path = (self.folder / name).resolve()
            if (
                not path.is_relative_to(self.folder)
                or hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                raise ValueError("Qualification evidence checksum/path mismatch")
        self.matrix = json.loads((self.folder / "qualification.json").read_text())
        if self.matrix["source_sha256"] != _source_fingerprint():
            raise DomainError(
                "Qualification requires matching runtime source", Applicability.UNSUPPORTED
            )

    def approved_input(self, case_id, cells):
        matches = [e for e in self.matrix["entries"] if e["id"] == case_id]
        if (
            len(matches) != 1
            or not matches[0]["accepted"]
            or cells not in matches[0].get("qualified_cells", [])
        ):
            raise DomainError("Case/grid has no accepted qualification", Applicability.UNSUPPORTED)
        expected = f"{case_id}_{cells}"
        if f"{expected}/input.json" not in self.hashes:
            raise ValueError("Qualified input must be checksummed")
        paths = [Path(p).name for p in matches[0]["paths"] if Path(p).name == expected]
        if len(paths) != 1:
            raise ValueError("Ambiguous qualified input")
        path = (self.folder / expected / "input.json").resolve()
        if (
            not path.is_relative_to(self.folder)
            or hashlib.sha256(path.read_bytes()).hexdigest()
            != self.hashes[f"{expected}/input.json"]
        ):
            raise ValueError("Qualified input changed after verification")
        return deepcopy(json.loads(path.read_text()))

    def verify_request(self, case_id, cells, config, duration_s, solver):
        expected = self.approved_input(case_id, cells)
        if (
            expected["config"] != config
            or expected["duration_s"] != duration_s
            or expected["solver"] != solver
        ):
            raise DomainError("Request differs from qualified fixture", Applicability.UNSUPPORTED)
        return True
