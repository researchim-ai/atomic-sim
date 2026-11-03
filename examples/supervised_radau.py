"""Calendar pause, online acknowledgement and exact local restart."""

from scripts.hysteresis_references import commands, relay
from scripts.supervised_radau import SupervisedRadauGraph


def main():
    controller = relay()
    controller["modes"].append(dict(id="hold", commands=commands(0), transitions=[]))
    request = dict(
        format="atomic-supervised-mode-graph",
        schema_version=1,
        controller=controller,
        schedule=[dict(id="pause", time_s=20, to="hold"), dict(id="resume", time_s=30, to="cool")],
        online_modes=["heat", "cool", "hold"],
    )
    model = SupervisedRadauGraph(request)
    model.advance(2)
    model.step("heat", 10, expected_revision=model.revision, command_id="operator:ack-1")
    model.advance(28)
    checkpoint = model.checkpoint()
    restored = SupervisedRadauGraph(request)
    restored.restore(checkpoint)
    assert model.advance(5) == restored.advance(5)
    state = model.snapshot()
    print(
        dict(
            time_s=state["time_s"],
            mode=state["mode"],
            revision=state["revision"],
            events=state["event_count"],
            calendar_cursor=state["calendar_cursor"],
            exact_local_restart=True,
        )
    )


if __name__ == "__main__":
    main()
