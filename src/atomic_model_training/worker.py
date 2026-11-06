"""Linux subprocess endpoint: JSON only, no imported user policy code."""

import json
import resource
import sys

from .episode import run_episode
from .spec import MAX_INPUT, exact, integer
from .template import Template


def send(value):
    print(json.dumps(value, allow_nan=False, separators=(",", ":")), flush=True)


def main():
    memory_mb = integer(int(sys.argv[1]), 128, 8192)
    cpu_s = integer(int(sys.argv[2]), 1, 300)
    # RSS is additionally watched by the parent; AS bounds virtual allocations.
    address_cap = (memory_mb * 4 + 1024) * 1024**2
    resource.setrlimit(resource.RLIMIT_AS, (address_cap, address_cap))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s * 4 + 10, cpu_s * 4 + 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_INPUT, MAX_INPUT))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    template = Template()
    for _ in range(4):
        raw = sys.stdin.buffer.readline(MAX_INPUT + 1)
        if not raw:
            return
        if len(raw) > MAX_INPUT:
            raise ValueError("Request byte budget exceeded")
        try:
            request = json.loads(raw)
            exact(request, {"case", "saved"})
            result = run_episode(
                request["case"],
                template=template,
                saved=request["saved"],
                emit=lambda r: send(dict(kind="progress", receipt=r)),
            )
            send(dict(kind="result", receipt=result))
        except Exception as exc:
            send(dict(kind="failure", error=type(exc).__name__ + ": " + str(exc)[:1000]))


if __name__ == "__main__":
    main()
