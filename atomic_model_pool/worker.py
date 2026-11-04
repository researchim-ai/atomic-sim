"""Private bounded mailbox worker. No user code or executable is loaded."""

import gc
import sys
import time

from atomic_model_bench.worker import run
from atomic_model_pool.identity import identity
from atomic_model_workspace.common import atomic_json, digest, keys, loads, read


def main():
    started = time.perf_counter()
    initial = identity()
    startup_wall_s = time.perf_counter() - started
    while True:
        line = sys.stdin.buffer.readline(8193)
        if not line:
            return
        if len(line) > 8192 or not line.endswith(b"\n"):
            raise ValueError("Pool mailbox size limit")
        command = loads(line)
        keys(command, ("request", "result", "token"))
        if identity() != initial:
            raise ValueError("Pool source changed during worker lifetime")
        request = read(command["request"])
        if command["token"] != digest(request):
            raise ValueError("Pool task token mismatch")
        # run() creates a brand-new EOSEpisode and local history on every call.
        result = run(request)
        atomic_json(
            command["result"],
            dict(
                schema_version=1,
                token=command["token"],
                pool=initial,
                startup_wall_s=startup_wall_s,
                result=result,
            ),
        )
        del request, result, command
        gc.collect()


if __name__ == "__main__":
    main()
