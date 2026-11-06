"""Measured runtime matrix, exact constraints and dependency/license inventory."""

import importlib.metadata as metadata
import platform
from pathlib import Path

from packaging.requirements import Requirement

from atomic_model_multiphysics.provenance import identity as model_identity
from atomic_model_multiphysics.provenance import sha
from atomic_model_workspace.common import digest


def identity():
    return dict(
        model=model_identity(),
        research={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    )


def environment():
    import numpy
    import scipy

    return dict(
        system=platform.system(),
        machine=platform.machine(),
        python=platform.python_version(),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
    )


def matrix_entry():
    current = environment()
    pinned = dict(system="Linux", machine="x86_64", python="3.12.9", numpy="2.2.6", scipy="1.15.2")
    return dict(
        scope="Headless controlled-PWR and steady spatial research paths",
        observed=current,
        supported=[dict(pinned, minimum=pinned, qualification="exact pinned runtime only")],
        observed_matches_supported=current == pinned,
        excluded=[
            "Untested OS/architectures",
            "Other Python/NumPy/SciPy versions",
            "GPU/MPI reactor solver",
        ],
        note="Package installability >=3.10 is not a qualification claim; only this exact profile is admitted",
    )


def dependency_manifest():
    pending = ["atomic-sim"]
    seen = {}
    missing = []
    while pending:
        name = pending.pop()
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        meta = dist.metadata
        record = dict(
            name=meta["Name"],
            version=dist.version,
            license_expression=meta.get("License-Expression")
            or meta.get("License")
            or "NOASSERTION",
            license_files=[],
        )
        for f in dist.files or []:
            if ("license" in f.name.lower() or "copying" in f.name.lower()) and ".dist-info" in str(
                f
            ):
                p = Path(dist.locate_file(f))
                if p.is_file() and p.stat().st_size < 2 * 1024**2:
                    record["license_files"].append(dict(path=str(f), sha256=sha(p)))
        seen[key] = record
        for text in dist.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return dict(
        format="dependency-license-inventory-v1",
        packages=sorted(seen.values(), key=lambda x: x["name"].lower()),
        missing=sorted(set(missing)),
        legal_approval_claimed=False,
        note="Installed metadata and license-file hashes, not a legal opinion or provenance signature",
    )


def sbom():
    manifest = dependency_manifest()
    return dict(
        spdxVersion="SPDX-2.3",
        dataLicense="CC0-1.0",
        SPDXID="SPDXRef-DOCUMENT",
        name="atomic-sim-local-research-build",
        documentNamespace="https://github.com/researchim-ai/atomic-sim/spdx/" + digest(manifest),
        creationInfo=dict(creators=["Tool: atomic-model-research"], created="2026-11-06T00:00:00Z"),
        packages=[
            dict(
                SPDXID="SPDXRef-Package-" + str(i),
                name=p["name"],
                versionInfo=p["version"],
                downloadLocation="NOASSERTION",
                filesAnalyzed=False,
                licenseConcluded="NOASSERTION",
                licenseDeclared="NOASSERTION",
                copyrightText="NOASSERTION",
            )
            for i, p in enumerate(manifest["packages"])
        ],
        annotations=[
            dict(
                annotationDate="2026-11-06T00:00:00Z",
                annotationType="OTHER",
                annotator="Tool: atomic-model-research",
                comment="Observed installed inventory; no publisher authentication",
            )
        ],
    )
