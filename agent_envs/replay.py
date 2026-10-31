"""Continue a full episode in a fresh process using strict JSON input."""

import argparse
import json
from pathlib import Path

from agent_envs.pwr_env import PWRPlantEnv
from core.status import DomainError


def continue_episode(payload, action):
    env = PWRPlantEnv.from_episode(payload)
    if env._done:
        raise ValueError("Cannot continue a completed episode; reset required")
    obs, reward, terminated, truncated, info = env.step(action)
    return dict(
        observation=obs.tolist(),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info=info,
        metrics=env.evaluator_metrics(),
        checkpoint=env.export_episode(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--action-json", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        # Open exclusively before computing to avoid a failed overwrite attempt
        # needlessly running a costly physical decision.
        if args.output.exists():
            raise FileExistsError(str(args.output))
        result = continue_episode(
            json.loads(args.checkpoint.read_text()), json.loads(args.action_json)
        )
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (ValueError, DomainError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
