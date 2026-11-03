"""Repeated heating/cooling modes with checkpoint on the first transition."""

import json

from scripts.hysteresis_references import relay
from scripts.hysteretic_radau import HystereticRadauGraph


def main():
    program = relay()
    model = HystereticRadauGraph(program)
    first = model.advance(120.0, stop_at_transition=True)
    restored = HystereticRadauGraph(program)
    restored.restore(model.checkpoint())
    final = restored.advance(120.0 - first["time_s"])
    print(
        json.dumps(
            dict(
                time_s=final["time_s"],
                mode=final["mode"],
                transition_count=final["transition_count"],
                transition_ids=sorted({r["guard_id"] for r in final["history"]}),
                root_checkpoint_mode=first["mode"],
                temperature_k=final["engine"]["state"]["temperatures_k"]["a"],
                energy_residual_j=final["engine"]["state"]["energy_residual_j"],
            )
        )
    )


if __name__ == "__main__":
    main()
