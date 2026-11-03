"""One-shot heating/cooling guards with checkpoint directly at the first root."""

import json

from scripts.state_event_radau import StateEventGraph
from scripts.state_event_references import cases


def main():
    request = dict(cases())["heat_then_cool"]
    first = StateEventGraph(request)
    stopped = first.advance(20.0, stop_at_event=True)
    restored = StateEventGraph(request)
    restored.restore(first.checkpoint())
    final = restored.advance(20.0 - stopped["time_s"])
    print(
        json.dumps(
            dict(
                stopped_at_event=stopped["stopped_at_event"],
                root_stop_time_s=stopped["time_s"],
                final_time_s=final["time_s"],
                fired_ids=[r["guard_id"] for r in final["fired"]],
                event_times_s=[r["time_s"] for r in final["fired"]],
                temperature_k=final["model"]["engine"]["state"]["temperatures_k"]["a"],
                energy_residual_j=final["model"]["engine"]["state"]["energy_residual_j"],
            )
        )
    )


if __name__ == "__main__":
    main()
