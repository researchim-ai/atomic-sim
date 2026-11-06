"""Cold wheel verification for the separate plant and phase packages."""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    from atomic_model_phase_assessment.assessment import evaluate
    from atomic_model_phase_assessment.dataset import load
    from atomic_model_phase_assessment.vessel import Vessel, cases
    from atomic_model_plant_extension import session

    root = Path(session.__file__).resolve().parent.parent
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Package imported from checkout")
    expected = json.loads(a.source.read_text())
    if session.identity() != expected:
        raise ValueError("Installed source identity differs")
    s = session.Session()
    s.advance(0.1, commands={"scram": True})
    saved = json.loads(json.dumps(s.checkpoint()))
    r = session.Session.restore(saved)
    if r.checkpoint() != saved:
        raise ValueError("Installed plant replay failed")
    manifest, traces = load()
    assessment = evaluate()
    checks = []
    for spec in cases():
        model = Vessel(spec)
        state = model.advance(model.initial, 0.5, 0.2)
        snapshot = json.loads(json.dumps(model.checkpoint(state)))
        if model.restore(snapshot) != state:
            raise ValueError("Installed phase replay failed")
        checks.append(dict(case=spec["name"], restart_equal=True))
    result = dict(
        installed_root=str(root),
        source=session.identity(),
        plant_restart_equal=True,
        phase_restarts=checks,
        pool_counts=manifest["retained_counts"],
        primary_raw_sha256=manifest["primary_sha256"],
        admitted_pool_rows=assessment["production_forced_boiling_admitted_rows"],
        validated=False,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    with a.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
