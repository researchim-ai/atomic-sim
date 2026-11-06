"""Verify an installed wheel outside checkout using real worker processes."""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected_source")
    parser.add_argument("cases")
    parser.add_argument("directory")
    parser.add_argument("output")
    args = parser.parse_args()
    import atomic_model_training
    from atomic_model_training.episode import identity
    from atomic_model_training.runner import Campaign, envelope
    from atomic_model_training.spec import atomic_json, read_json
    from atomic_model_training.tiers import control_replay

    installed = Path(atomic_model_training.__file__).resolve()
    if not installed.is_relative_to(Path(sys.prefix).resolve()):
        raise ValueError("Imported runtime outside installed environment")
    if identity() != read_json(args.expected_source):
        raise ValueError("Installed complete source identity differs")
    cases = read_json(args.cases)
    campaign = Campaign(cases, args.directory, workers=2).run()
    resumed = Campaign(cases, args.directory, workers=1).run(resume=True)
    control = control_replay(cases[1])
    if not campaign["completed"] or resumed != campaign or not control["accepted"]:
        raise ValueError("Installed campaign/replay/tier qualification failed")
    atomic_json(
        args.output,
        envelope(
            dict(
                source=identity(),
                installed_root=str(installed.parent),
                campaign_completed=True,
                completed_resume_equal=True,
                core_control_replay=control,
                result_checkpoint_sha256=[
                    r["receipt"]["checkpoint"]["sha256"] for r in campaign["results"]
                ],
                validated=False,
                plant_referenced=False,
            )
        ),
    )
    print(json.dumps(dict(installed=True, campaign_completed=True, resume_equal=True)))


if __name__ == "__main__":
    main()
