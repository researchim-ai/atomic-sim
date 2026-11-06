"""Run/replay the equilibrium-vessel foundation and its IF97 qualification."""

import argparse
from pathlib import Path

from atomic_model_hem import workflow
from atomic_model_workspace.common import atomic_json, read


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["demo", "qualify"]:
        a = sub.add_parser(name)
        a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("run")
    a.add_argument("input", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("replay")
    a.add_argument("folder", type=Path)
    a = p.parse_args(argv)
    try:
        if a.command == "qualify":
            r = workflow.qualify()
            if a.output.exists():
                raise FileExistsError(a.output)
            atomic_json(a.output, r)
            print(
                dict(
                    inverse_grid_count=r["inverse_grid_count"],
                    vessel_trajectories=r["vessel_trajectories"],
                    validated=False,
                )
            )
            return 0
        if a.command == "demo":
            r = workflow.demo(a.output)
        elif a.command == "run":
            r = workflow.run(read(a.input), a.output)
        else:
            r = workflow.replay(a.folder)
    except (ValueError, OSError, TypeError, KeyError) as exc:
        p.error(str(exc))
    state = r["observations"][-1]
    print(
        dict(
            time_s=state["time_s"],
            phase=state["thermodynamic_state"]["phase"],
            phase_events=len(state["events"]),
            validated=False,
            plant_referenced=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
