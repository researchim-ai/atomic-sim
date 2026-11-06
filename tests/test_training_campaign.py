"""Actual-core and adversarial tests for the bounded training runtime."""

import copy
import json
import os
import time

import pytest

from atomic_model_controlled_plant.session import Session
from atomic_model_training.episode import run_episode
from atomic_model_training.runner import Campaign, envelope, unpack
from atomic_model_training.spec import (
    admit,
    admit_campaign,
    atomic_json,
    episode_seed,
    promotion,
    read_json,
)
from atomic_model_training.template import Template


def case(name="nominal", commands=None, steps=2, split="validation", tier="training"):
    return dict(
        id=name,
        family=name,
        split=split,
        seed=17,
        tier=tier,
        steps=[dict(duration_s=0.1, commands=commands or {}) for _ in range(steps)],
    )


@pytest.mark.parametrize(
    "commands",
    [{}, {"throttle": 0.9}, {"scram": True}, {"motor_off": True}, {"feed_bias_kg_s": -10}],
)
def test_pristine_template_exact_full_core_checkpoint(commands):
    template = Template()
    clone, cold = template.new(), Session()
    for _ in range(2):
        clone.advance(0.1, commands=commands)
        cold.advance(0.1, commands=commands)
        assert clone.checkpoint() == cold.checkpoint()
    assert template.new().checkpoint() == Session().checkpoint()


def test_dirty_template_rejected():
    template = Template()
    template._pristine.advance(0.1)
    with pytest.raises(ValueError, match="pristine"):
        template.new()


def test_resume_replays_exact_accepted_prefix():
    spec = case(steps=3)
    progress = []
    full = run_episode(spec, emit=progress.append)
    restored = run_episode(spec, saved=progress[1])
    assert restored == full
    broken = copy.deepcopy(progress[1])
    broken["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="mismatch"):
        run_episode(spec, saved=broken)
    broken = copy.deepcopy(spec)
    broken["steps"][0]["commands"] = {"throttle": 0.9}
    with pytest.raises(ValueError):
        run_episode(broken, saved=progress[1])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(policy="import os; os.system('false')"),
        lambda s: s.update(id="../escape"),
        lambda s: s.update(seed=True),
        lambda s: s.update(tier="surrogate"),
        lambda s: s["steps"][0].update(duration_s=float("nan")),
        lambda s: s["steps"][0].update(commands={"exec": "danger"}),
        lambda s: s["steps"][0].update(commands={"throttle": 10}),
    ],
)
def test_strict_policy_admission(mutation):
    value = case()
    mutation(value)
    with pytest.raises(ValueError):
        admit(value)


def test_family_and_curve_split_leakage():
    a, b = case("a", split="calibration"), case("b")
    with pytest.raises(ValueError, match="leakage"):
        admit_campaign([a, b])
    b["steps"][0]["commands"] = {"throttle": 0.9}
    b["family"] = "a"
    with pytest.raises(ValueError, match="leakage"):
        admit_campaign([a, b])
    b["family"] = "b"
    assert len(admit_campaign([a, b])) == 2


def test_seed_order_independent_and_promotion_boundaries():
    a = case()
    assert episode_seed(42, a) == episode_seed(42, copy.deepcopy(a))
    assert episode_seed(43, a) != episode_seed(42, a)
    for kind in ("numerical", "intercode", "experimental"):
        for status in ("validated", "plant_referenced"):
            with pytest.raises(ValueError):
                promotion(kind, status)
    assert not promotion("numerical", "verified")["validated"]


def test_regular_bounded_no_symlink_no_fifo(tmp_path):
    p = tmp_path / "input.json"
    p.write_text('{"value": NaN}')
    with pytest.raises(ValueError):
        read_json(p)
    p.write_text("[1,2]")
    assert read_json(p) == [1, 2]
    with pytest.raises(ValueError):
        read_json(p, limit=2)
    link = tmp_path / "link"
    link.symlink_to(p)
    with pytest.raises(OSError):
        read_json(link)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError):
        read_json(fifo)
    with pytest.raises(ValueError):
        read_json("/dev/zero")


def test_corrupt_artifact_refused(tmp_path):
    p = tmp_path / "a.json"
    atomic_json(p, envelope(dict(answer=42)))
    assert unpack(p) == dict(answer=42)
    obj = json.loads(p.read_text())
    obj["payload"]["answer"] = 43
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="Corrupt"):
        unpack(p)


def test_sync_async_same_core_and_completed_resume(tmp_path):
    cases = [case("nominal"), case("load", {"throttle": 0.9})]
    sync = Campaign(cases, tmp_path / "sync", workers=2, root_seed=12).run()
    async_job = Campaign(cases, tmp_path / "async", workers=2, root_seed=12)
    asynchronous = async_job.start().result(timeout=30)
    assert sync["completed"] and asynchronous["completed"]
    for a, b in zip(sync["results"], asynchronous["results"]):
        assert a["receipt"] == b["receipt"]
        assert a["seed"] == b["seed"]
    resumed = Campaign(cases, tmp_path / "sync", root_seed=12).run(resume=True)
    assert resumed == sync
    with pytest.raises(ValueError):
        Campaign(cases, tmp_path / "sync", root_seed=13).run(resume=True)


def test_deadline_and_retry_preserve_campaign(tmp_path):
    cases = [case("a"), case("b", {"throttle": 0.9})]
    report = Campaign(cases, tmp_path, deadline_s=0.001).run()
    assert not report["completed"]
    assert all(r["status"] == "deadline" for r in report["results"])
    resumed = Campaign(cases, tmp_path, deadline_s=30).run(resume=True)
    assert resumed["completed"]


def test_cancel_during_real_core_then_resume(tmp_path):
    cases = [case("a", steps=128), case("b", {"throttle": 0.9})]
    job = Campaign(cases, tmp_path)
    future = job.start()
    until = time.monotonic() + 20
    progress_path = tmp_path / "a.progress.json"
    while time.monotonic() < until:
        if (
            progress_path.exists()
            and unpack(progress_path)["checkpoint"]["payload"]["revision"] >= 1
        ):
            break
        time.sleep(0.02)
    else:
        pytest.fail("No actual accepted-step progress")
    job.cancel()
    report = future.result(timeout=10)
    assert not report["completed"]
    assert report["results"][0]["status"] == "cancelled"
    assert unpack(progress_path)["checkpoint"]["payload"]["revision"] >= 1
    resumed = Campaign(cases, tmp_path, deadline_s=60).run(resume=True)
    assert resumed["completed"]


def test_dead_worker_does_not_destroy_next_episode(tmp_path, monkeypatch):
    from atomic_model_training import runner

    original = runner.Campaign._start_worker
    calls = []

    def start(self):
        process = original(self)
        if not calls:
            process.kill()
        calls.append(process.pid)
        return process

    monkeypatch.setattr(runner.Campaign, "_start_worker", start)
    report = Campaign([case("a"), case("b")], tmp_path).run()
    assert report["results"][0]["status"] == "worker_failed"
    assert report["results"][1]["status"] == "completed"
    assert len(calls) == 2


def test_output_clock_cannot_change_partition():
    a = case("a", steps=2, split="calibration")
    b = case("b", steps=3)
    b["steps"][0]["duration_s"] = 0.2
    with pytest.raises(ValueError, match="leakage"):
        admit_campaign([a, b])


def test_separate_evidence_register_and_real_replay(tmp_path):
    from atomic_model_training.evidence import EvidenceStore

    store = EvidenceStore(tmp_path)
    receipt = run_episode(case(steps=1))
    numerical = store.record("numerical", receipt)
    experimental = store.record("experimental", dict(source="unqualified experiment"))
    intercode = store.record("intercode", dict(source="unqualified other code"))
    assert numerical.parent != experimental.parent != intercode.parent
    assert (
        store.promote(numerical, "verified")["replayed_checkpoint_sha256"]
        == receipt["checkpoint"]["sha256"]
    )
    for path in (numerical, experimental, intercode):
        with pytest.raises(ValueError):
            store.promote(path, "validated")
    for path in (experimental, intercode):
        with pytest.raises(ValueError):
            store.promote(path, "verified")


def test_tier_actual_core_control_replay():
    from atomic_model_training.tiers import control_replay

    result = control_replay(case("load", {"throttle": 0.9}))
    assert result["accepted"] and not result["surrogate"]
    assert result["state_scaled_error"] < 1e-3


def test_cache_is_bounded_and_restores_original_exact_result():
    from atomic_model_training.cache import CachedVolume

    template = Template()
    session = template.new()
    volume = next(iter(session.plant.network.volumes.values()))
    assert isinstance(volume, CachedVolume)
    energy = volume.internal_energy_j
    original = volume.sync()
    for k in range(140):
        volume.internal_energy_j = energy + k * 10
        volume.sync()
    assert len(volume._exact_cache) <= 128
    volume.internal_energy_j = energy
    assert volume.sync() == original
    assert not getattr(template._pristine.plant.network.volumes[volume.name], "_exact_cache", {})


def _hostile_test_process(code):
    import subprocess
    import sys

    return subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        bufsize=0,
    )


@pytest.mark.parametrize("resource_kind", ["memory", "cpu"])
def test_resource_failure_isolated_from_next_episode(tmp_path, monkeypatch, resource_kind):
    original = Campaign._start_worker
    calls = []

    def start(self):
        if not calls:
            code = (
                "import time; a=bytearray(200*1024**2); time.sleep(20)"
                if resource_kind == "memory"
                else "while True: pass"
            )
            process = _hostile_test_process(code)
        else:
            process = original(self)
        calls.append(process.pid)
        return process

    monkeypatch.setattr(Campaign, "_start_worker", start)
    report = Campaign([case("a"), case("b")], tmp_path, memory_mb=128, cpu_s=1, deadline_s=20).run()
    assert report["results"][0]["status"] == resource_kind + "_limit"
    assert report["results"][1]["status"] == "completed"


def test_deadline_also_bounds_blocked_request_pipe(tmp_path):
    process = _hostile_test_process("import time; time.sleep(20)")
    job = Campaign([case()], tmp_path, deadline_s=0.05)
    try:
        result = job._episode(process, case(), {"padding": "x" * (1024**2)})
        assert result["status"] == "deadline"
    finally:
        from atomic_model_training.runner import stop

        stop(process)


def test_cross_episode_cache_never_changes_science():
    template = Template()
    programs = [{}, {"throttle": 0.9}, {"motor_off": True}, {}, {"scram": True}]
    for index, commands in enumerate(programs):
        spec = case(f"case-{index}", commands)
        accelerated = run_episode(spec, template=template)
        baseline = run_episode(spec)
        assert accelerated == baseline


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"tier":"training","tier":"reference"}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_json(path)


def test_admission_does_not_mutate_caller():
    spec = case("a", {"throttle": 1})
    original = copy.deepcopy(spec)
    normalized = admit(spec)
    assert spec == original
    assert type(normalized["steps"][0]["commands"]["throttle"]) is float


def test_state_units_follow_real_core_layout():
    receipt = run_episode(case(steps=1))
    assert len(receipt["state"]) == len(receipt["state_schema"])
    assert receipt["state_schema"][10] == dict(index=10, name="fuel_temperature", unit="K")
    assert receipt["state_schema"][11]["unit"] == "rad/s"
    assert receipt["state_schema"][13]["unit"] == "kg"
