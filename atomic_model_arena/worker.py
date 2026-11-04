"""Private fresh interpreter: pin CPU before importing numerical packages."""

import json
import os
import sys


def main():
    request = json.loads(sys.stdin.buffer.readline(256 * 1024 + 1))
    if (
        set(request) != {"schema_version", "pack", "commands", "output", "cpus"}
        or type(request["schema_version"]) is not int
        or request["schema_version"] != 1
        or not isinstance(request["cpus"], list)
        or len(request["cpus"]) != 1
    ):
        raise ValueError("Arena worker request")
    os.sched_setaffinity(0, request["cpus"])
    from atomic_model_arena.runner import execute

    print(
        json.dumps(
            execute(request["pack"], request["commands"], request["output"], request["cpus"]),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
