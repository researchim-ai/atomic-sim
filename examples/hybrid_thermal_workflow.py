"""Portable declarative run, full replay and coordinated controller/plant continuation."""

import argparse
from copy import deepcopy
from pathlib import Path

from atomic_model_workspace.common import atomic_json
from scripts.hybrid_graph_cli import replay, run
from scripts.hybrid_graph_references import cases


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    a.output.mkdir(parents=True, exist_ok=False)
    _, model, program = next(c for c in cases() if c[0] == "explicit_online_state")
    spec = dict(
        schema_version=1,
        model=model,
        max_macro_step_s=1.0,
        program=program[:3],
        initial_checkpoint=None,
    )
    partial = run(spec, a.output / "partial")
    assert replay(a.output / "partial") == partial
    continued = deepcopy(spec)
    continued["initial_checkpoint"] = partial["final_checkpoint"]
    continued["program"] = program[3:]
    final = run(continued, a.output / "continued")
    assert replay(a.output / "continued") == final
    full = deepcopy(spec)
    full["program"] = program
    control = run(full, a.output / "control")
    assert final["final_checkpoint"] == control["final_checkpoint"]
    proof = dict(
        schema_version=1,
        final_time_s=40.0,
        continued_checkpoint_exact=True,
        controller_state=final["observations"][-1]["controller_state"],
        transition_count=len(final["observations"][-1]["history"]),
        validated=False,
        plant_referenced=False,
    )
    atomic_json(a.output / "proof.json", proof)
    print(proof)


if __name__ == "__main__":
    main()
