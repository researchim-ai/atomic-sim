"""Run from repository root: python -m examples.closed_loop_eos_pi."""

import json

from scripts.closed_loop_eos_pi import ClosedLoopEOSPI
from scripts.eos_pi_references import request


def main():
    episode = ClosedLoopEOSPI(request(mode="filtered"))
    result = episode.run(10, expected_revision=episode.revision)
    print(
        json.dumps(
            dict(
                controller=result["controller"],
                history=result["history"],
                plant=result["plant"],
                work=result["work"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
