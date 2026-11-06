"""Publish complete numerical evidence with exact wheel/sdist byte witnesses."""

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

from atomic_model_controlled_plant.session import identity
from scripts.evidence_bundle import create_bundle, verify_bundle


def files_under(root, relative):
    base = root / relative
    return [p for p in sorted(base.rglob("*")) if p.is_file() and "__pycache__" not in p.parts]


def check_packages(root, wheel, sdist):
    paths = []
    for name in [
        "atomic_model_plant_extension",
        "atomic_model_phase_assessment",
        "atomic_model_controlled_plant",
    ]:
        paths.extend(files_under(root, name))
    dependencies = json.loads(
        (root / "validation/ai050607_external_dependencies_v1.json").read_text()
    )
    for name, sha in dependencies["files"].items():
        p = root / name
        if hashlib.sha256(p.read_bytes()).hexdigest() != sha:
            raise ValueError("Shared dependency changed:" + name)
        paths.append(p)
    with zipfile.ZipFile(wheel) as wh, tarfile.open(sdist) as source:
        top = source.getnames()[0].split("/")[0]
        for path in paths:
            name = path.relative_to(root).as_posix()
            raw = path.read_bytes()
            if wh.read(name) != raw:
                raise ValueError("Wheel bytes differ:" + name)
            member = source.extractfile(top + "/" + name)
            if member is None or member.read() != raw:
                raise ValueError("Sdist bytes differ:" + name)
    return dict(
        checked_files=len(paths),
        wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        sdist_sha256=hashlib.sha256(sdist.read_bytes()).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", required=True, type=Path)
    p.add_argument("--selection", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--receipt", required=True, type=Path)
    a = p.parse_args()
    root = Path.cwd()
    wheel = next(a.build.glob("*.whl"))
    sdist = next(a.build.glob("*.tar.gz"))
    witness = check_packages(root, wheel, sdist)
    selected = json.loads(a.selection.read_text())
    if selected["metadata"]["source"] != identity():
        raise ValueError("Evidence source changed")
    result = create_bundle(selected, root, a.output)
    verified = verify_bundle(a.output)
    receipt = dict(
        archive=result,
        verified=verified,
        package_byte_witness=witness,
        validated=False,
        plant_referenced=False,
    )
    with a.receipt.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print("Verified evidence archive", a.output)


if __name__ == "__main__":
    main()
