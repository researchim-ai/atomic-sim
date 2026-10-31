"""Generate, tune, run and independently rebuild a public PWR AI benchmark."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from agent_envs import EnvConfig, PWRPlantEnv
from agent_envs.benchmark_metrics import metrics, summarize
from agent_envs.benchmark_pack import digest, generate_pack, validate_pack
from agent_envs.policy_process import PolicyError, PolicyProcess
from sdk.experiment import _source_fingerprint


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def truth(env):
    observed = env.plant.observation()
    return dict(
        power_fraction=observed["thermal_power_w"] / env.plant.fission_power_nom,
        fuel_temperature_k=observed["fuel_temperature_k"],
        primary_pressure_pa=observed["primary_pressure_pa"],
        scram_latched=observed["scram_latched"],
    )


def policy_command(kind, parameters=None):
    return [
        sys.executable,
        "-m",
        "agent_envs.benchmark_policy",
        "--kind",
        kind,
        "--parameters-json",
        json.dumps(parameters or {}, allow_nan=False),
    ]


def run_episode(case, worker):
    env = PWRPlantEnv(config=EnvConfig.from_dict(case["config"]))
    obs, _ = env.reset()
    raw = dict(
        schema_version=1,
        case=case,
        initial_truth=truth(env),
        steps=[],
        reason="running",
        label_origin="synthetic scenario configuration, not expert causal annotation",
    )
    start = time.monotonic()
    try:
        worker.reset(env.config.target_power_fraction, env.config.action_dt_s)
        while not env._done:
            answer = worker.act(obs, env.observation_metadata())
            try:
                env._action(answer["action"])
            except ValueError as exc:
                raise PolicyError("Policy action violates declared schema/range") from exc
            obs, reward, terminated, truncated, info = env.step(answer["action"])
            raw["steps"].append(
                dict(
                    time_s=env.plant.time_s,
                    observation=obs.tolist(),
                    measurement=env.observation_metadata(),
                    action=answer["action"],
                    diagnosis=answer["diagnosis"],
                    truth=truth(env),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    reason=info["reason"],
                )
            )
            raw["reason"] = info["reason"]
    except PolicyError as exc:
        raw["reason"] = "policy_error"
        raw["policy_error"] = str(exc)
    raw["wall_time_s"] = time.monotonic() - start
    raw["checkpoint"] = env.export_episode()
    raw["prescribed_events"] = list(env._event_log)
    return raw


def run_cases(cases, command, output, policy_name):
    worker = None
    records = []
    try:
        for case in cases:
            if worker is None:
                worker = PolicyProcess(command)
            raw = run_episode(case, worker)
            raw["policy"] = policy_name
            raw["policy_command"] = command
            write(output / (case["id"] + ".json"), raw)
            records.append(
                dict(
                    policy=policy_name,
                    case_id=case["id"],
                    family=case["family"],
                    metrics=metrics(raw),
                )
            )
            if raw["reason"] == "policy_error":
                worker.close()
                worker = None
    finally:
        if worker is not None:
            worker.close()
    return records


def tune(pack, output):
    """Select on train only, then report unchanged selected parameters on dev."""
    output.mkdir()
    candidates = [
        dict(kp=kp, ki=0.0001, kd=0.0, output_limit=0.0005) for kp in (0.001, 0.003, 0.01)
    ]
    scores = []
    for i, parameters in enumerate(candidates):
        folder = output / f"candidate-{i}"
        folder.mkdir()
        records = run_cases(
            pack["splits"]["train"], policy_command("pid", parameters), folder, "pid"
        )
        result = summarize(records)["policies"]["pid"]
        scores.append(dict(parameters=parameters, train=result))
    best = min(
        range(len(scores)),
        key=lambda i: (
            -scores[i]["train"]["completed"],
            scores[i]["train"]["mean_iae_completed_s"]
            if scores[i]["train"]["mean_iae_completed_s"] is not None
            else float("inf"),
        ),
    )
    folder = output / "dev"
    folder.mkdir()
    dev = run_cases(pack["splits"]["dev"], policy_command("pid", candidates[best]), folder, "pid")
    result = dict(
        schema_version=1,
        selection="train completion first, then mean IAE; dev reporting only",
        training_digest=digest(pack["splits"]["train"]),
        development_digest=digest(pack["splits"]["dev"]),
        candidates=scores,
        selected_parameters=candidates[best],
        dev=summarize(dev)["policies"]["pid"],
        source_sha256=_source_fingerprint(),
        test_used=False,
    )
    write(output / "tuning.json", result)
    return result


def verify_tuning(pack, tuning):
    if (
        tuning.get("test_used") is not False
        or tuning["training_digest"] != digest(pack["splits"]["train"])
        or tuning["development_digest"] != digest(pack["splits"]["dev"])
    ):
        raise ValueError("Tuning provenance does not match train/dev")
    if tuning["selected_parameters"] not in [c["parameters"] for c in tuning["candidates"]]:
        raise ValueError("Selected PID absent from training candidates")
    if tuning["source_sha256"] != _source_fingerprint():
        raise ValueError("Tuning/runtime source mismatch")
    candidates = tuning["candidates"]
    if not candidates or any(
        c["train"]["episodes"] != len(pack["splits"]["train"]) for c in candidates
    ):
        raise ValueError("Tuning training coverage mismatch")
    best = min(
        candidates,
        key=lambda c: (
            -c["train"]["completed"],
            c["train"]["mean_iae_completed_s"]
            if c["train"]["mean_iae_completed_s"] is not None
            else float("inf"),
        ),
    )
    if best["parameters"] != tuning["selected_parameters"]:
        raise ValueError("Selected PID does not match training objective")


def campaign(pack, tuning, output, split="test", external_command=None):
    verify_tuning(pack, tuning)
    output.mkdir()
    write(output / "pack.json", pack)
    write(output / "tuning.json", tuning)
    metadata = dict(
        schema_version=1,
        source_sha256=_source_fingerprint(),
        pack_sha256=digest(pack),
        split=split,
        public_corpus=pack["public_corpus"],
        blind_test=pack["blind_test"],
        protocol_scope="Evaluator in parent; trusted policy subprocess, no OS filesystem sandbox",
        expected_cases=[c["id"] for c in pack["splits"][split]],
        policy_names=["external"] if external_command else ["hold", "pid", "rule"],
    )
    write(output / "campaign.json", metadata)
    records = []
    commands = (
        {"external": external_command}
        if external_command
        else {
            kind: policy_command(kind, tuning["selected_parameters"] if kind == "pid" else None)
            for kind in ("hold", "pid", "rule")
        }
    )
    for name, command in commands.items():
        folder = output / name
        folder.mkdir()
        records.extend(run_cases(pack["splits"][split], command, folder, name))
    report = summarize(records)
    report.update(
        public_corpus=pack["public_corpus"],
        blind_test=pack["blind_test"],
        split=split,
        source_sha256=metadata["source_sha256"],
        recorded_episodes=len(records),
        all_episodes_completed=all(r["metrics"]["completed"] for r in records),
    )
    write(output / "report.json", report)
    hashes = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.rglob("*.json")
    }
    write(output / "manifest.json", dict(schema_version=1, sha256=hashes))
    return report


def rebuild(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    actual = {
        str(p.relative_to(folder)) for p in folder.rglob("*.json") if p.name != "manifest.json"
    }
    if actual != set(manifest["sha256"]):
        raise ValueError("Campaign file set mismatch")
    for name, value in manifest["sha256"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != value:
            raise ValueError("Campaign checksum mismatch")
    metadata = json.loads((folder / "campaign.json").read_text())
    if (
        not metadata["policy_names"]
        or len(set(metadata["policy_names"])) != len(metadata["policy_names"])
        or set(metadata["policy_names"]) - {"hold", "pid", "rule", "external"}
    ):
        raise ValueError("Invalid campaign policy namespace")
    pack = validate_pack(json.loads((folder / "pack.json").read_text()))
    if digest(pack) != metadata["pack_sha256"]:
        raise ValueError("Pack fingerprint mismatch")
    expected = {c["id"]: c for c in pack["splits"][metadata["split"]]}
    if set(metadata["expected_cases"]) != set(expected):
        raise ValueError("Unexpected campaign coverage")
    records = []
    for policy in metadata["policy_names"]:
        found = list((folder / policy).glob("*.json"))
        if {p.stem for p in found} != set(expected):
            raise ValueError("Missing episode records")
        for path in sorted(found):
            raw = json.loads(path.read_text())
            if raw["case"] != expected[path.stem] or raw["policy"] != policy:
                raise ValueError("Episode case binding mismatch")
            records.append(
                dict(
                    policy=policy,
                    case_id=path.stem,
                    family=raw["case"]["family"],
                    metrics=metrics(raw),
                )
            )
    report = summarize(records)
    report.update(
        public_corpus=pack["public_corpus"],
        blind_test=pack["blind_test"],
        split=metadata["split"],
        source_sha256=metadata["source_sha256"],
        recorded_episodes=len(records),
        all_episodes_completed=all(r["metrics"]["completed"] for r in records),
    )
    # Order-independent aggregate sums can differ by roundoff; use canonical
    # case ordering in run and rebuild to make the machine report reproducible.
    original = json.loads((folder / "report.json").read_text())
    if digest(report) != digest(original):
        raise ValueError("Recomputed report differs from stored report")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="op", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("tune")
    p.add_argument("pack", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("pack", type=Path)
    p.add_argument("tuning", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "dev", "test"], default="test")
    p.add_argument("--policy-command-json")
    p = sub.add_parser("report")
    p.add_argument("campaign", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.op == "generate":
        write(args.output, generate_pack())
        return 0
    if args.op == "report":
        write(args.output, rebuild(args.campaign))
        return 0
    pack = validate_pack(json.loads(args.pack.read_text()))
    if args.op == "tune":
        result = tune(pack, args.output)
    else:
        result = campaign(
            pack,
            json.loads(args.tuning.read_text()),
            args.output,
            args.split,
            json.loads(args.policy_command_json) if args.policy_command_json else None,
        )
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
