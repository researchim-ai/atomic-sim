"""Cold installed runtime/source/resource verification outside checkout."""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    from atomic_model_controlled_plant import Session
    from atomic_model_controlled_plant.session import identity
    from atomic_model_phase_assessment.assessment import evaluate
    from atomic_model_phase_assessment.dataset import load

    root = Path(sys.modules[Session.__module__].__file__).resolve().parent.parent
    if not root.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Runtime imported from checkout")
    expected = json.loads(a.source.read_text())
    if identity() != expected:
        raise ValueError("Installed complete source differs")
    s = Session()
    s.advance(0.1, commands=dict(scram=True, motor_off=True, throttle=1.0))
    saved = json.loads(json.dumps(s.checkpoint()))
    resumed = Session.restore(saved)
    if resumed.checkpoint() != saved:
        raise ValueError("Installed controlled replay failed")
    m, _ = load()
    screen = evaluate()
    receipt = dict(
        source=identity(),
        installed_root=str(root),
        controlled_restart_equal=True,
        controller=s.plant.feed_controller,
        model_id=s.plant.passport.model_id,
        pool_retained_counts=m["retained_counts"],
        forced_boiling_admitted_rows=screen["production_forced_boiling_admitted_rows"],
        validated=False,
        plant_referenced=False,
    )
    with a.output.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
