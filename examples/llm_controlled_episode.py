"""Real local JSONL client for the controlled PWR, without an external LLM.

Run: python -m examples.llm_controlled_episode
Replies have deadlines; the example does not write files or contact providers.
"""

import json
import queue
import subprocess
import sys
import threading


def main():
    process = subprocess.Popen(
        [sys.executable, "-m", "atomic_agent_tools"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )
    replies = queue.Queue()

    def read():
        for line in process.stdout:
            replies.put(line)
        replies.put(None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    counter = 0

    def call(tool, arguments):
        nonlocal counter
        counter += 1
        request_id = f"r{counter}"
        process.stdin.write(
            json.dumps(dict(request_id=request_id, tool=tool, arguments=arguments), allow_nan=False)
            + "\n"
        )
        process.stdin.flush()
        try:
            line = replies.get(timeout=60)
        except queue.Empty as exc:
            raise RuntimeError("Worker reply deadline exceeded") from exc
        if line is None:
            raise RuntimeError("Worker closed before reply")
        response = json.loads(line)
        if response["request_id"] != request_id or not response["ok"]:
            raise RuntimeError(response)
        return response["result"]

    def guard(view):
        return dict(
            session_id=view["session_id"],
            expected_revision=view["revision"],
            observation_id=view["observation_id"],
        )

    try:
        initial = call(
            "create_controlled_episode", dict(session_id="pwr", config=dict(max_steps=2))
        )
        saved = call("save_checkpoint", guard(initial))
        first = call("step_controlled_episode", dict(**guard(initial), action={"throttle": 0.95}))
        restored = call(
            "restore_checkpoint", dict(**guard(first), checkpoint_id=saved["checkpoint_id"])
        )
        repeated = call(
            "step_controlled_episode", dict(**guard(restored), action={"throttle": 0.95})
        )
        if first["observation"] != repeated["observation"]:
            raise RuntimeError("Restored trajectory differs")
        final = call("step_controlled_episode", dict(**guard(repeated), action={}))
        exported = call(
            "export_controlled_case", dict(**guard(final), case_id="llm_load", family="load_change")
        )
        call("close_episode", guard(final))
        call("forget_checkpoint", dict(checkpoint_id=saved["checkpoint_id"]))
        process.stdin.close()
        process.wait(timeout=60)
        if process.returncode != 0:
            raise RuntimeError("Worker failed")
        print(
            json.dumps(
                dict(
                    exact_checkpoint_replay=True,
                    model=final["model"],
                    time_s=final["time_s"],
                    case=exported["case"],
                ),
                indent=2,
            )
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        reader.join(timeout=10)
        process.stdout.close()
        if not process.stdin.closed:
            process.stdin.close()


if __name__ == "__main__":
    main()
