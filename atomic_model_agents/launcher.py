"""Private Linux exec trampoline; no shell, parent-death signal before agent exec."""

import ctypes
import json
import os
import signal
import sys


def main():
    parent = int(sys.argv[1])
    command = json.loads(sys.argv[2])
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl")
    if os.getppid() != parent:
        raise SystemExit(3)
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
