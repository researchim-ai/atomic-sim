"""Execute an exact admitted Euler fixture through the public experiment SDK."""

import argparse
import hashlib
import json
from pathlib import Path

from sdk.euler_experiment import run_euler_experiment
from sdk.euler_qualification import EulerQualification


def run(qualification, case, cells, output, *, wall_limit_s=None):
    folder = Path(qualification)
    gate = EulerQualification(folder)
    fixture = gate.approved_input(case, cells)
    gate.verify_request(case, cells, **fixture)
    budget = json.loads((folder / "budget.json").read_text())
    spec = dict(
        schema_version=1,
        engine="contact-euler-liquid-channel",
        name=case,
        channel=fixture["config"],
        duration_s=fixture["duration_s"],
        step_s=fixture["duration_s"] / 10,
        solver=fixture["solver"],
        budgets=dict(
            mass_relative=budget["mass_relative"],
            energy_absolute_j=budget["energy_abs_j"],
            momentum_absolute_kg_m_s=budget["momentum_abs_kg_m_s"],
        ),
    )
    report = run_euler_experiment(spec, output, wall_limit_s=wall_limit_s)
    admission = dict(
        case=case,
        cells=cells,
        source_sha256=gate.matrix["source_sha256"],
        qualification_matrix_sha256=hashlib.sha256(
            (folder / "qualification.json").read_bytes()
        ).hexdigest(),
        canonical_input_sha256=gate.hashes[f"{case}_{cells}/input.json"],
        validated=False,
        plant_referenced=False,
        scope="Exact component fixture admission; no whole-plant qualification",
    )
    output = Path(output)
    with (output / "admission.json").open("x") as stream:
        json.dump(admission, stream, indent=2)
        stream.write("\n")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["admission.json"] = hashlib.sha256(
        (output / "admission.json").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-limit-s", type=float)
    args = parser.parse_args()
    result = run(
        args.qualification, args.case, args.cells, args.output, wall_limit_s=args.wall_limit_s
    )
    print(
        json.dumps(
            dict(status=result["status"], numerical_acceptance=result["numerical_acceptance"])
        )
    )
    return 0 if result["numerical_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
