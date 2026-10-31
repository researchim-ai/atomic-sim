"""Process transport for trusted Python policies, not an OS security sandbox."""

import json
import os
import selectors
import subprocess
import tempfile
import time
from pathlib import Path


class PolicyError(RuntimeError):
    pass


class PolicyProcess:
    def __init__(self, command, timeout_s=20):
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(s, str) for s in command)
        ):
            raise ValueError("Policy command must be an argv list, never a shell string")
        self.timeout = timeout_s
        self.directory = tempfile.TemporaryDirectory(prefix="atomic-policy-")
        self.errors = tempfile.TemporaryFile()
        env = {
            "PATH": os.defpath,
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.errors,
            cwd=self.directory.name,
            env=env,
            bufsize=0,
        )
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def request(self, payload):
        try:
            self.process.stdin.write((json.dumps(payload, allow_nan=False) + "\n").encode())
            self.process.stdin.flush()
            deadline = time.monotonic() + self.timeout
            line = b""
            while b"\n" not in line:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self.selector.select(remaining):
                    raise PolicyError("Policy response deadline exceeded")
                chunk = os.read(self.process.stdout.fileno(), 4096)
                if not chunk:
                    raise PolicyError("Policy exited without a response")
                line += chunk
                if len(line) > 65536:
                    raise PolicyError("Policy response exceeds size limit")
            if line.count(b"\n") != 1 or not line.endswith(b"\n"):
                raise PolicyError("Unexpected extra policy response")
            result = json.loads(line)
            if not isinstance(result, dict):
                raise PolicyError("Policy response must be an object")
            json.dumps(result, allow_nan=False)
            return result
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise PolicyError("Invalid policy response") from exc

    def reset(self, target, dt):
        response = self.request(dict(schema_version=1, op="reset", target=target, dt_s=dt))
        if response != {"ready": True}:
            raise PolicyError("Invalid policy reset acknowledgement")

    def act(self, observation, measurement):
        response = self.request(
            dict(
                schema_version=1,
                op="act",
                observation=observation.tolist(),
                measurement=measurement,
            )
        )
        if set(response) != {"action", "diagnosis"} or response["diagnosis"] not in (
            "healthy",
            "sensor_fault",
        ):
            raise PolicyError("Invalid policy action schema")
        return response

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.selector.close()
        self.process.stdin.close()
        self.process.stdout.close()
        self.errors.close()
        self.directory.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
