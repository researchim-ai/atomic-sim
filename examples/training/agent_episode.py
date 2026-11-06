"""Run from checkout: python -m examples.training.agent_episode OUTPUT.json.

No external LLM, training framework, API key or remote runner is required.
"""

import argparse
import json
from pathlib import Path

from atomic_model_training.episode import run_episode
from atomic_pwr_agent import Episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    episode = Episode(max_steps=3, seed=17)
    transitions = []
    for commands in ({"throttle": 0.95}, {}, {}):
        observation = episode.observe()
        transitions.append(
            dict(
                before=observation,
                commands=commands,
                result=episode.step(commands, observation_id=observation["observation_id"]),
            )
        )
        if len(transitions) == 1:
            episode = Episode.restore(episode.checkpoint())
    case = episode.training_case(case_id="agent_example", family="load_change")
    replay = run_episode(case)
    if replay["checkpoint"] != episode.checkpoint()["payload"]["core"]:
        raise RuntimeError("Campaign replay differs")
    payload = dict(
        schema_version=1,
        transitions=transitions,
        case=case,
        checkpoint=episode.checkpoint(),
        campaign_replay_equal=True,
        validated=False,
        plant_referenced=False,
    )
    # Never overwrite a previous trajectory.
    with args.output.open("x") as stream:
        json.dump(payload, stream, allow_nan=False, indent=2)
    print(f"Saved {len(transitions)} transitions; campaign replay equal: {args.output}")


if __name__ == "__main__":
    main()
