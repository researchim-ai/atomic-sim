"""Install a wheel in an isolated temporary venv and exercise its real resources.

Dependencies are inherited from the current environment; the package itself is
installed from the wheel with no index access. Run after `python -m build`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import venv
import zipfile
from pathlib import Path

SMOKE = r"""
import json
from pathlib import Path
import subprocess
import sys
from data.catalog import assert_bundled_checksum
from scenarios.session import bundled_scenario
from sdk.api import create_session, export_snapshot, restore_snapshot
from sdk.experiment import run_experiment

expected = Path(sys.argv[1]).resolve()
import sdk
assert expected in Path(sdk.__file__).resolve().parents, sdk.__file__
assert assert_bundled_checksum("project-open-two-group")["bundled"]
spec = bundled_scenario("vver_load_rejection.yaml")
spec = {key: value for key, value in spec.items() if not key.startswith("_")}
scenario = Path("smoke.yaml")
import yaml
scenario.write_text(yaml.safe_dump(spec), encoding="utf-8")
executable = Path(sys.executable).with_name("atomic-experiment")
subprocess.run([str(executable), str(scenario), "--output", "run"], check=True)
result = json.loads(Path("run/report.json").read_text())
assert result["status"] == "completed"
assert result["completed_time_s"] == spec["duration_s"]
central = bundled_scenario("vver_nominal_hour_central.yaml")
central = {key: value for key, value in central.items() if not key.startswith("_")}
central["duration_s"] = 0.4
central["step_s"] = 0.2
scenario.write_text(yaml.safe_dump(central), encoding="utf-8")
subprocess.run([str(executable), str(scenario), "--output", "central_run"], check=True)
assert json.loads(Path("central_run/report.json").read_text())["numerical_acceptance"]
radau = bundled_scenario("vver_radau_throttle_120s.yaml")
radau = {key: value for key, value in radau.items() if not key.startswith("_")}
radau.update(duration_s=0.4, step_s=0.2, timeline=[{"t": 0.1, "throttle": 0.98}],
             domain_boundary_tolerance_s=1e-4)
scenario.write_text(yaml.safe_dump(radau), encoding="utf-8")
subprocess.run([str(executable), str(scenario), "--output", "radau_run"], check=True)
assert json.loads(Path("radau_run/report.json").read_text())["numerical_acceptance"]
scram = bundled_scenario("vver_scram_60s.yaml")
scram = {key: value for key, value in scram.items() if not key.startswith("_")}
scram.update(duration_s=4.0, step_s=1.0, timeline=[{"t": 0.0, "scram": True}])
scenario.write_text(yaml.safe_dump(scram), encoding="utf-8")
subprocess.run([str(executable), str(scenario), "--output", "scram_run"], check=True)
scram_result = json.loads(Path("scram_run/report.json").read_text())
assert scram_result["numerical_acceptance"]
assert scram_result["extrema"]["rod_insertion"]["max"] == 1.0
stops = [event for event in scram_result["events"] if event["kind"] == "rod_stop"]
assert len(stops) == 1
assert stops[0]["position"] == 1.0
assert abs(stops[0]["t"] - 3.75) <= 1e-12
from physics.materials.iapws import properties_pt
hot = properties_pt(15.5976954793e6, 617.8110989055)
payload = export_snapshot(create_session())
mass = hot.rho_kg_m3 * 8.0
payload["state"][16:19] = [mass, mass * hot.u_j_kg, mass * 0.001]
cold = create_session()
restore_snapshot(cold, payload)
assert export_snapshot(cold) == payload
from core.status import ModelLimitError
from sdk.boundary import advance_until_limit
probe = create_session()
raw_rhs = probe.derivatives
def limited_rhs(t, y):
    if t > 0.04:
        raise ModelLimitError("wheel probe", criterion="time", node="synthetic",
                              observed=t, limit=0.04)
    return raw_rhs(t, y)
probe.derivatives = limited_rhs
bounded = advance_until_limit(probe, 0.1, time_tolerance_s=1e-5,
                              method="Radau", jacobian_relative_step=1e-5)
assert bounded.boundary["interval_width_s"] <= 1e-5
assert 0.04 - 1e-5 <= probe.time_s <= 0.04
assert not bounded.boundary["physical_crossing_localized"]
assert bundled_scenario("vver_boundary_1s.yaml")["domain_boundary_tolerance_s"] == 1e-4
channel_spec = bundled_scenario("isobaric_decay_1800s.yaml")
channel_spec = {key: value for key, value in channel_spec.items() if not key.startswith("_")}
channel_spec.update(duration_s=2.0, step_s=0.5)
scenario.write_text(yaml.safe_dump(channel_spec), encoding="utf-8")
channel_executable = Path(sys.executable).with_name("atomic-channel-experiment")
subprocess.run([str(channel_executable), str(scenario), "--output", "channel_run"], check=True)
channel_report = json.loads(Path("channel_run/report.json").read_text())
assert channel_report["numerical_acceptance"]
assert channel_report["extrema"]["outlet_quality"]["max"] > 0.0
from physics.hydraulics.isobaric import IsobaricChannel
channel_snapshot = json.loads(Path("channel_run/checkpoint.json").read_text())
assert IsobaricChannel.from_snapshot(channel_snapshot).snapshot() == channel_snapshot
wall_spec = bundled_scenario("isobaric_wall_1800s.yaml")
wall_spec = {key: value for key, value in wall_spec.items() if not key.startswith("_")}
wall_spec.update(duration_s=2.0, step_s=0.5)
wall_spec["wall"]["power_off_time_s"] = 0.7
scenario.write_text(yaml.safe_dump(wall_spec), encoding="utf-8")
subprocess.run([str(channel_executable), str(scenario), "--output", "wall_run"], check=True)
wall_report = json.loads(Path("wall_run/report.json").read_text())
assert wall_report["numerical_acceptance"]
assert wall_report["events"] == [{"kind": "heater_off", "t": 0.7}]
from physics.hydraulics.wall_channel import WallChannel
wall_snapshot = json.loads(Path("wall_run/checkpoint.json").read_text())
assert WallChannel.from_snapshot(wall_snapshot).snapshot() == wall_snapshot
dynamic_spec = bundled_scenario("pressure_wall_1800s.yaml")
dynamic_spec = {key: value for key, value in dynamic_spec.items() if not key.startswith("_")}
dynamic_spec.update(duration_s=0.1, step_s=0.05)
scenario.write_text(yaml.safe_dump(dynamic_spec), encoding="utf-8")
dynamic_executable = Path(sys.executable).with_name("atomic-dynamic-experiment")
subprocess.run([str(dynamic_executable), str(scenario), "--output", "dynamic_run"], check=True)
dynamic_report = json.loads(Path("dynamic_run/report.json").read_text())
assert dynamic_report["numerical_acceptance"]
assert dynamic_report["extrema"]["inlet_mass_flow_kg_s"]["max"] > 0
from physics.hydraulics.dynamic_channel import DynamicWallChannel
dynamic_snapshot = json.loads(Path("dynamic_run/checkpoint.json").read_text())
assert DynamicWallChannel.from_snapshot(dynamic_snapshot).snapshot() == dynamic_snapshot
pipe_spec = bundled_scenario("pipe_geometry_120s.yaml")
pipe_spec = {key: value for key, value in pipe_spec.items() if not key.startswith("_")}
pipe_spec.update(duration_s=0.05, step_s=0.025)
scenario.write_text(yaml.safe_dump(pipe_spec), encoding="utf-8")
subprocess.run([str(dynamic_executable), str(scenario), "--output", "pipe_run"], check=True)
assert json.loads(Path("pipe_run/report.json").read_text())["numerical_acceptance"]
pipe_snapshot = json.loads(Path("pipe_run/checkpoint.json").read_text())
pipe_model = DynamicWallChannel.from_snapshot(pipe_snapshot)
assert pipe_model.snapshot() == pipe_snapshot
assert pipe_model.config.junction_lengths_m == (0.375, 0.75, 0.75, 0.75, 0.375)
assert abs(sum(j.length_m for j in pipe_model.network.junctions.values()) - 3.0) < 1e-12
plant = create_session()
plant.set_sink_temperature_k(310.0)
restored = create_session()
restore_snapshot(restored, export_snapshot(plant))
assert restored.t_sink == 310.0
print(json.dumps({"wheel_smoke": "passed", "model": plant.passport.model_id,
                  "duration_s": result["completed_time_s"]}))
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        required = {
            "data/sources.json",
            "scenarios/vver_load_rejection.yaml",
            "validation/run_manifest.json",
            "sdk/experiment.py",
            "core/jacobian.py",
            "core/motion.py",
            "sdk/boundary.py",
            "sdk/channel_experiment.py",
            "physics/hydraulics/isobaric.py",
            "physics/hydraulics/wall_channel.py",
            "physics/hydraulics/dynamic_channel.py",
            "sdk/dynamic_experiment.py",
            "scenarios/pressure_wall_1800s.yaml",
            "scenarios/pipe_geometry_120s.yaml",
            "scenarios/vver_nominal_hour_central.yaml",
            "scenarios/vver_radau_throttle_120s.yaml",
            "scenarios/vver_radau_throttle_120s_resolved.yaml",
            "scenarios/vver_radau_throttle_120s_resolved_base.yaml",
            "scenarios/vver_scram_60s.yaml",
            "scenarios/vver_protected_coastdown_120s.yaml",
            "scenarios/vver_boundary_1s.yaml",
            "scenarios/isobaric_decay_1800s.yaml",
            "scenarios/isobaric_wall_1800s.yaml",
        }
        missing = required - set(archive.namelist())
        if missing:
            raise RuntimeError("Missing wheel resources: %s" % sorted(missing))
    with tempfile.TemporaryDirectory(prefix="atomic-wheel-check-") as temporary:
        folder = Path(temporary)
        env = folder / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env)
        python = env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        clean_env = dict(os.environ)
        clean_env.pop("PYTHONPATH", None)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--ignore-installed",
                str(wheel),
            ],
            check=True,
            cwd=folder,
            env=clean_env,
        )
        site = subprocess.check_output(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            text=True,
            cwd=folder,
            env=clean_env,
        ).strip()
        subprocess.run([str(python), "-c", SMOKE, site], check=True, cwd=folder, env=clean_env)
    print(json.dumps({"verified_wheel": str(wheel)}))


if __name__ == "__main__":
    main()
