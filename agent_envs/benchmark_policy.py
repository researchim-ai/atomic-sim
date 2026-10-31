"""Trusted policy worker: JSON-lines protocol, no simulator state in requests."""

import argparse
import json
import sys

from control.regulation import PID


class BaselinePolicy:
    def __init__(self, kind, parameters=None):
        if kind not in ("hold", "pid", "rule"):
            raise ValueError("Unknown baseline")
        self.kind = kind
        self.parameters = parameters or {}

    def reset(self, target, dt):
        self.target, self.dt = target, dt
        self.pid = PID(**self.parameters) if self.kind == "pid" else None

    def act(self, observation, measurement):
        good = measurement["valid"][0] and not measurement["saturated"][0]
        error = float(observation[0]) - self.target
        diagnosis = "healthy" if good or self.kind == "hold" else "sensor_fault"
        if self.kind == "hold" or not good:
            command = 0.0
        elif self.kind == "pid":
            command = self.pid.step(error, self.dt)
        else:
            command = 1e-5 if error > 1e-4 else -1e-5 if error < -1e-4 else 0.0
        return dict(action=[float(command)], diagnosis=diagnosis)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["hold", "pid", "rule"], required=True)
    p.add_argument("--parameters-json", default="{}")
    args = p.parse_args()
    policy = BaselinePolicy(args.kind, json.loads(args.parameters_json))
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("schema_version") != 1:
            raise ValueError("Unknown policy protocol")
        if request["op"] == "reset":
            if set(request) != {"schema_version", "op", "target", "dt_s"}:
                raise ValueError("Unexpected policy reset fields")
            policy.reset(request["target"], request["dt_s"])
            result = {"ready": True}
        elif request["op"] == "act":
            if set(request) != {"schema_version", "op", "observation", "measurement"}:
                raise ValueError("Unexpected policy request fields")
            result = policy.act(request["observation"], request["measurement"])
        else:
            raise ValueError("Unknown policy operation")
        sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
