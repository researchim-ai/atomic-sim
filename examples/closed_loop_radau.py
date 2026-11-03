"""Persist controller memory, owned plant and RNG, then continue exactly."""

import json

from scripts.closed_loop_radau import ClosedLoopRadau
from scripts.closed_loop_references import cases


def main():
    request = dict(cases())["idle_seed42"]
    model = ClosedLoopRadau(request)
    model.run(17, expected_revision=0)
    saved = json.loads(json.dumps(model.checkpoint()))
    restored = ClosedLoopRadau(request)
    restored.restore(saved)
    assert model.run(23, expected_revision=1) == restored.run(23, expected_revision=1)
    state = model.snapshot()
    print(
        dict(
            time_s=state["plant"]["time_s"],
            ticks=state["tick_count"],
            episode_revision=state["revision"],
            plant_revision=state["plant"]["revision"],
            memory=state["decision"],
            rng_sha256=state["rng_sha256"],
            exact_local_feedback_restart=True,
        )
    )


if __name__ == "__main__":
    main()
