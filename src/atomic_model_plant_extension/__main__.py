"""Run source-bound reduced-PWR campaigns and replay checkpoints."""

import argparse
import json
from pathlib import Path

from .qualification import CASES, publish, run_case
from .session import Session


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("case", choices=sorted(CASES))
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--rtol", type=float, default=1e-6)
    run.add_argument("--atol", type=float, default=1e-5)
    run.add_argument("--max-step", type=float, default=0.5)
    run.add_argument("--output-step", type=float, default=5.0)
    run.add_argument("--method", choices=["Radau", "BDF"], default="Radau")
    replay = sub.add_parser("replay")
    replay.add_argument("checkpoint", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == "run":
        result = run_case(
            args.case,
            output=args.output,
            rtol=args.rtol,
            atol=args.atol,
            max_step_s=args.max_step,
            output_step_s=args.output_step,
            method=args.method,
        )
        print(
            json.dumps(
                dict(
                    case=args.case,
                    accepted=result["full_horizon_accepted"],
                    failure=result["failure"],
                )
            )
        )
        return 0 if result["full_horizon_accepted"] else 1
    if args.output.exists():
        raise FileExistsError(args.output)
    envelope = json.loads(args.checkpoint.read_text())
    from atomic_model_workspace.common import digest

    if set(envelope) != {"payload", "sha256"} or digest(envelope["payload"]) != envelope["sha256"]:
        raise ValueError("Invalid checkpoint-file envelope")
    saved = envelope["payload"]
    session = Session.restore(saved)
    publish(
        args.output,
        dict(
            replay_equal=True,
            revision=session.revision,
            checkpoint_sha256=saved["sha256"],
            validated=False,
        ),
    )
    print("Replay equal:", session.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
