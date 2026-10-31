"""Install a wheel in an isolated temporary venv and exercise its real resources.

Dependencies are inherited from the current environment; the package itself is
installed from the wheel with no index access. Run after `python -m build`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import venv
import zipfile
from pathlib import Path

SMOKE = r"""
import json
import numpy as np
from pathlib import Path
import subprocess
import sys
from data.catalog import assert_bundled_checksum
from scenarios.session import bundled_scenario
from sdk.api import create_session, export_snapshot, restore_snapshot
from sdk.experiment import run_experiment, _source_fingerprint

expected = Path(sys.argv[1]).resolve()
import sdk
assert expected in Path(sdk.__file__).resolve().parents, sdk.__file__
assert _source_fingerprint() == sys.argv[2], "Installed runtime fingerprint differs from checkout"
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
profile_spec = bundled_scenario("pipe_wall_pulse_n4.yaml")
profile_spec = {key: value for key, value in profile_spec.items() if not key.startswith("_")}
profile_spec.update(duration_s=0.02, step_s=0.01)
scenario.write_text(yaml.safe_dump(profile_spec), encoding="utf-8")
subprocess.run([str(dynamic_executable), str(scenario), "--output", "profile_run"], check=True)
profile_report = json.loads(Path("profile_run/report.json").read_text())
assert profile_report["numerical_acceptance"]
profile_snapshot = json.loads(Path("profile_run/checkpoint.json").read_text())
profile_model = DynamicWallChannel.from_snapshot(profile_snapshot)
assert profile_model.snapshot() == profile_snapshot
assert profile_model.config.initial_pressure_profile_pa is not None
assert profile_model.initial_energy_j > pipe_model.initial_energy_j
muscl_spec = bundled_scenario("pipe_muscl_pulse_8s.yaml")
muscl_spec = {key: value for key, value in muscl_spec.items() if not key.startswith("_")}
muscl_spec.update(duration_s=0.01, step_s=0.005)
scenario.write_text(yaml.safe_dump(muscl_spec), encoding="utf-8")
subprocess.run([str(dynamic_executable), str(scenario), "--output", "muscl_run"], check=True)
muscl_report = json.loads(Path("muscl_run/report.json").read_text())
assert muscl_report["numerical_acceptance"]
assert muscl_report["solver_diagnostics"]["accepted_rhs_evaluations"] > 0
assert muscl_report["solver_diagnostics"]["accepted_output_intervals"] == 2
muscl_snapshot = json.loads(Path("muscl_run/checkpoint.json").read_text())
muscl_model = DynamicWallChannel.from_snapshot(muscl_snapshot)
assert muscl_model.config.enthalpy_transport == "muscl_mc"
assert muscl_model.snapshot() == muscl_snapshot
analytic_spec = bundled_scenario("batch11_muscl_n4_analytic.yaml")
analytic_spec = {key: value for key, value in analytic_spec.items() if not key.startswith("_")}
analytic_spec.update(duration_s=0.01, step_s=0.005)
scenario.write_text(yaml.safe_dump(analytic_spec), encoding="utf-8")
subprocess.run([str(dynamic_executable), str(scenario), "--output", "analytic_run"], check=True)
analytic_report = json.loads(Path("analytic_run/report.json").read_text())
assert analytic_report["numerical_acceptance"]
assert analytic_report["solver_diagnostics"]["analytic_jacobian_successes"] > 0
assert analytic_report["solver_diagnostics"]["last_analytic_jacobian_nnz"] > 0
analytic_snapshot = json.loads(Path("analytic_run/checkpoint.json").read_text())
assert DynamicWallChannel.from_snapshot(analytic_snapshot).snapshot() == analytic_snapshot
total_spec = bundled_scenario("batch12_total_wall_8s.yaml")
total_spec = {key: value for key, value in total_spec.items() if not key.startswith("_")}
total_spec.update(duration_s=0.02, step_s=0.01)
scenario.write_text(yaml.safe_dump(total_spec), encoding="utf-8")
subprocess.run([str(dynamic_executable), str(scenario), "--output", "total_run"], check=True)
total_report = json.loads(Path("total_run/report.json").read_text())
assert total_report["numerical_acceptance"]
assert total_report["budget_checks"]["energy_absolute_j"]
assert total_report["budget_checks"]["mechanical_absolute_j"]
total_snapshot = json.loads(Path("total_run/checkpoint.json").read_text())
assert total_snapshot["schema_version"] == 2
assert total_snapshot["model_id"] == "compatible-total-energy-liquid-channel"
assert DynamicWallChannel.from_snapshot(total_snapshot).snapshot() == total_snapshot
reverse_spec = json.loads(json.dumps(total_spec))
hydraulics = reverse_spec["hydraulics"]
hydraulics["left_pressure_pa"], hydraulics["right_pressure_pa"] = (
    hydraulics["right_pressure_pa"], hydraulics["left_pressure_pa"]
)
hydraulics["initial_flow_kg_s"] *= -1
for field in ("initial_pressure_profile_pa", "initial_temperature_profile_k",
              "initial_wall_temperature_profile_k"):
    hydraulics[field].reverse()
json_scenario = Path("reverse.JSON")
json_scenario.write_text(json.dumps(reverse_spec))
subprocess.run([str(dynamic_executable), str(json_scenario), "--output", "reverse_run"], check=True)
reverse_report = json.loads(Path("reverse_run/report.json").read_text())
assert reverse_report["numerical_acceptance"]
assert json.loads(Path("reverse_run/checkpoint.json").read_text())["config"]["initial_flow_kg_s"] < 0
from physics.hydraulics.dynamic_channel import DynamicConfig
from physics.hydraulics.total_energy_initialization import equilibrated_total_config
muscl_input = dict(reverse_spec["hydraulics"])
for field in ("initial_pressure_profile_pa", "initial_temperature_profile_k",
              "initial_wall_temperature_profile_k"):
    muscl_input.pop(field)
muscl_input.update(enthalpy_transport="muscl_mc", heat_w=3000.0)
muscl_equilibrium = equilibrated_total_config(DynamicConfig(**muscl_input), wall_offset_k=2.0)
reverse_spec["hydraulics"] = muscl_equilibrium.as_dict()
json_scenario.write_text(json.dumps(reverse_spec))
subprocess.run([str(dynamic_executable), str(json_scenario), "--output", "muscl_total_run"], check=True)
assert json.loads(Path("muscl_total_run/report.json").read_text())["numerical_acceptance"]
from agent_envs import EnvConfig, PWRPlantEnv
env = PWRPlantEnv(config=EnvConfig(action_dt_s=0.01, episode_horizon_s=0.025, sensor_noise=0.01))
env.reset(seed=7)
env.step([0.001])
episode = env.export_episode()
Path("episode.json").write_text(json.dumps(episode))
agent_executable = Path(sys.executable).with_name("atomic-agent-replay")
subprocess.run([str(agent_executable), "episode.json", "--action-json", "[0.0]",
                "--output", "continued_episode.json"], check=True)
continued = json.loads(Path("continued_episode.json").read_text())
expected = env.step([0.0])
assert np.allclose(continued["observation"], expected[0], rtol=0, atol=1e-6)
assert abs(continued["reward"] - expected[1]) < 1e-12
assert continued["checkpoint"]["rng"] == env.export_episode()["rng"]
assert PWRPlantEnv.from_episode(continued["checkpoint"]).export_episode() == continued["checkpoint"]
benchmark_executable = Path(sys.executable).with_name("atomic-agent-benchmark")
subprocess.run([str(benchmark_executable), "generate", "--output", "benchmark_pack.json"], check=True)
from agent_envs.benchmark import policy_command, run_episode
from agent_envs.benchmark_metrics import metrics
from agent_envs.benchmark_pack import validate_pack
from agent_envs.policy_process import PolicyProcess
pack = validate_pack(json.loads(Path("benchmark_pack.json").read_text()))
assert len(pack["splits"]["test"]) == 80
with PolicyProcess(policy_command("hold")) as worker:
    raw_episode = run_episode(pack["splits"]["train"][0], worker)
assert metrics(raw_episode)["completed"]
from physics.hydraulics.euler_channel import EulerConfig, EulerLiquidChannel
euler_spec = dict(schema_version=1, engine="conservative-euler-liquid-channel", name="installed_euler",
                  duration_s=0.02, step_s=0.01,
                  channel=EulerConfig(cells=3, boundary="periodic", initial_velocity_m_s=2.0,
                                      total_ua_w_k=0.0).as_dict())
Path("euler.json").write_text(json.dumps(euler_spec))
euler_executable = Path(sys.executable).with_name("atomic-euler-experiment")
subprocess.run([str(euler_executable), "euler.json", "--output", "euler_run"], check=True)
euler_report = json.loads(Path("euler_run/report.json").read_text())
assert euler_report["numerical_acceptance"]
assert euler_report["budget_checks"]["momentum_absolute_kg_m_s"]
euler_snapshot = json.loads(Path("euler_run/checkpoint.json").read_text())
assert euler_snapshot["schema_version"] == 3
assert EulerLiquidChannel.from_snapshot(euler_snapshot).snapshot() == euler_snapshot
euler_spec["engine"] = "contact-euler-liquid-channel"
euler_spec["channel"]["numerical_flux"] = "hllc"
Path("contact.json").write_text(json.dumps(euler_spec))
subprocess.run([str(euler_executable), "contact.json", "--output", "contact_run"], check=True)
assert json.loads(Path("contact_run/report.json").read_text())["numerical_acceptance"]
contact_snapshot = json.loads(Path("contact_run/checkpoint.json").read_text())
assert contact_snapshot["schema_version"] == 4
contact = EulerLiquidChannel.from_snapshot(contact_snapshot)
assert contact.snapshot() == contact_snapshot
assert contact.observation()["right_port_x_m"] == contact.config.total_length_m
euler_spec.update(duration_s=0.001, step_s=0.001)
euler_spec["solver"] = {"atol_scale": 0.1}
euler_spec["channel"] = EulerConfig(cells=512, numerical_flux="hllc", boundary="periodic",
                                    initial_velocity_m_s=1.0, total_ua_w_k=0,
                                    total_loss_coefficient=0).as_dict()
Path("contact_refined.json").write_text(json.dumps(euler_spec))
subprocess.run([str(euler_executable), "contact_refined.json", "--output", "contact_refined_run"], check=True)
assert json.loads(Path("contact_refined_run/report.json").read_text())["numerical_acceptance"]
refined = json.loads(Path("contact_refined_run/checkpoint.json").read_text())
assert refined["config"]["cells"] == 512
assert json.loads(Path("contact_refined_run/input.json").read_text())["solver"]["atol_scale"] == 0.1
assert EulerLiquidChannel.from_snapshot(refined).snapshot() == refined
from physics.hydraulics.euler_initialization import steady_contact_config
from sdk.euler_qualification import EulerQualification
assert callable(steady_contact_config) and callable(EulerQualification)
from importlib.resources import files
from components.linear_thermal_network import LinearThermalNetwork
network_spec = json.loads(files("scenarios").joinpath("ai05_thermal_two_capacity.json").read_text())
Path("network.json").write_text(json.dumps(network_spec))
network_executable = Path(sys.executable).with_name("atomic-network-experiment")
subprocess.run([str(network_executable), "network.json", "--output", "network_run"], check=True)
network_report = json.loads(Path("network_run/report.json").read_text())
assert network_report["numerical_acceptance"] and network_report["completed_time_s"] == 1000
network_checkpoint = json.loads(Path("network_run/checkpoint.json").read_text())
network = LinearThermalNetwork.from_snapshot(network_checkpoint)
assert network.snapshot() == network_checkpoint
assert abs(network.advance(10)["energy_residual_j"]) <= 1e-5
from physics.hydraulics.circulation_loop import CirculationLoop
circulation_spec = json.loads(files("scenarios").joinpath("ai05_circulation_reversal.json").read_text())
Path("circulation.json").write_text(json.dumps(circulation_spec))
circulation_executable = Path(sys.executable).with_name("atomic-circulation-experiment")
subprocess.run([str(circulation_executable), "circulation.json", "--output", "circulation_run"], check=True)
circulation_report = json.loads(Path("circulation_run/report.json").read_text())
assert circulation_report["numerical_acceptance"] and circulation_report["completed_time_s"] == 10
circulation_checkpoint = json.loads(Path("circulation_run/checkpoint.json").read_text())
circulation = CirculationLoop.from_snapshot(circulation_checkpoint)
assert circulation.snapshot() == circulation_checkpoint
assert abs(circulation.advance(1)["energy_residual_j"]) <= 1e-5
from physics.hydraulics.pump_loop import PumpCirculationLoop
pump_spec = json.loads(files("scenarios").joinpath("ai05_displacement_pump_trip.json").read_text())
Path("pump.json").write_text(json.dumps(pump_spec))
pump_executable = Path(sys.executable).with_name("atomic-pump-experiment")
subprocess.run([str(pump_executable), "pump.json", "--output", "pump_run"], check=True)
pump_report = json.loads(Path("pump_run/report.json").read_text())
assert pump_report["numerical_acceptance"] and pump_report["completed_time_s"] == 20
assert pump_report["accepted_commands"] == 1
pump_checkpoint = json.loads(Path("pump_run/checkpoint.json").read_text())
pump = PumpCirculationLoop.from_snapshot(pump_checkpoint)
assert pump.snapshot() == pump_checkpoint
assert abs(pump.advance(1)["total_energy_residual_j"]) <= 1e-5
from physics.hydraulics.valved_pump_loop import ValvedPumpLoop
valve_spec = json.loads(files("scenarios").joinpath("ai05_valve_close_reopen.json").read_text())
Path("valve.json").write_text(json.dumps(valve_spec))
valve_executable = Path(sys.executable).with_name("atomic-valve-experiment")
subprocess.run([str(valve_executable), "valve.json", "--output", "valve_run"], check=True)
valve_report = json.loads(Path("valve_run/report.json").read_text())
assert valve_report["numerical_acceptance"] and valve_report["completed_time_s"] == 20
assert valve_report["accepted_commands"] == 3
valve_checkpoint = json.loads(Path("valve_run/checkpoint.json").read_text())
valve = ValvedPumpLoop.from_snapshot(valve_checkpoint)
assert valve.snapshot() == valve_checkpoint
assert abs(valve.advance(1)["total_energy_residual_j"]) <= 1e-5
from physics.hydraulics.gas_storage_pair import GasStoragePair
storage_spec = json.loads(files("scenarios").joinpath("ai05_storage_isolation.json").read_text())
Path("storage.json").write_text(json.dumps(storage_spec))
storage_executable = Path(sys.executable).with_name("atomic-storage-experiment")
subprocess.run([str(storage_executable), "storage.json", "--output", "storage_run"], check=True)
storage_report = json.loads(Path("storage_run/report.json").read_text())
assert storage_report["numerical_acceptance"] and storage_report["completed_time_s"] == 20
assert storage_report["accepted_commands"] == 2
storage_checkpoint = json.loads(Path("storage_run/checkpoint.json").read_text())
storage = GasStoragePair.from_snapshot(storage_checkpoint)
assert storage.snapshot() == storage_checkpoint
assert abs(storage.advance(1)["energy_residual_j"]) <= 1e-5
from physics.hydraulics.inertial_storage_pair import InertialStoragePair
inertial_spec = json.loads(files("scenarios").joinpath("ai05_inertial_storage_damping.json").read_text())
Path("inertial.json").write_text(json.dumps(inertial_spec))
inertial_executable = Path(sys.executable).with_name("atomic-inertial-storage-experiment")
subprocess.run([str(inertial_executable), "inertial.json", "--output", "inertial_run"], check=True)
inertial_report = json.loads(Path("inertial_run/report.json").read_text())
assert inertial_report["numerical_acceptance"] and inertial_report["completed_time_s"] == 20
assert inertial_report["accepted_commands"] == 2
inertial_checkpoint = json.loads(Path("inertial_run/checkpoint.json").read_text())
inertial = InertialStoragePair.from_snapshot(inertial_checkpoint)
assert inertial.snapshot() == inertial_checkpoint
assert abs(inertial.advance(1)["energy_residual_j"]) <= 1e-5
from physics.hydraulics.pump_storage_pair import PumpStoragePair
coupled_spec = json.loads(files("scenarios").joinpath("ai05_pump_storage_trip.json").read_text())
Path("coupled.json").write_text(json.dumps(coupled_spec))
coupled_executable = Path(sys.executable).with_name("atomic-pump-storage-experiment")
subprocess.run([str(coupled_executable), "coupled.json", "--output", "coupled_run"], check=True)
coupled_report = json.loads(Path("coupled_run/report.json").read_text())
assert coupled_report["numerical_acceptance"] and coupled_report["completed_time_s"] == 20
assert coupled_report["accepted_commands"] == 2
coupled_checkpoint = json.loads(Path("coupled_run/checkpoint.json").read_text())
coupled = PumpStoragePair.from_snapshot(coupled_checkpoint)
assert coupled.snapshot() == coupled_checkpoint
assert abs(coupled.advance(1)["energy_residual_j"]) <= 1e-5
from physics.hydraulics.check_pump_storage import CheckPumpStorage
check_spec = json.loads(files("scenarios").joinpath("ai05_check_valve_restart.json").read_text())
Path("check.json").write_text(json.dumps(check_spec))
check_executable = Path(sys.executable).with_name("atomic-check-storage-experiment")
subprocess.run([str(check_executable), "check.json", "--output", "check_run"], check=True)
check_report = json.loads(Path("check_run/report.json").read_text())
assert check_report["numerical_acceptance"] and check_report["completed_time_s"] == 20
assert check_report["accepted_commands"] == 2
assert any(e["kind"] == "check_closed" for e in check_report["events"])
check_checkpoint = json.loads(Path("check_run/checkpoint.json").read_text())
check = CheckPumpStorage.from_snapshot(check_checkpoint)
assert check.snapshot() == check_checkpoint
assert abs(check.advance(1)["energy_residual_j"]) <= 1e-5






if sys.argv[3]:
    gate = EulerQualification(sys.argv[3])
    admitted = 0
    for entry in gate.matrix["entries"]:
        for cells in entry.get("qualified_cells", []):
            gate.verify_request(entry["id"], cells, **gate.approved_input(entry["id"], cells))
            admitted += 1
    assert admitted > 0
    print(json.dumps({"installed_qualified_fixtures": admitted}))
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
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        checkout = Path(__file__).resolve().parent.parent
        digest = hashlib.sha256()
        for package in (
            "sdk",
            "core",
            "plants",
            "physics",
            "components",
            "control",
            "instrumentation",
            "agent_envs",
            "data",
            "scenarios",
        ):
            for path in sorted((checkout / package).rglob("*")):
                if path.is_file() and path.suffix in (".py", ".json", ".yaml"):
                    relative = path.relative_to(checkout)
                    name = str(relative)
                    if (
                        relative.as_posix() not in archive.namelist()
                        or archive.read(relative.as_posix()) != path.read_bytes()
                    ):
                        raise RuntimeError("Wheel runtime differs from checkout: " + name)
                    digest.update(name.encode())
                    digest.update(b"\0")
                    digest.update(path.read_bytes())
        expected_source = digest.hexdigest()
        required = {
            "data/sources.json",
            "scenarios/vver_load_rejection.yaml",
            "validation/run_manifest.json",
            "sdk/experiment.py",
            "agent_envs/config.py",
            "agent_envs/episode_validation.py",
            "agent_envs/replay.py",
            "agent_envs/benchmark.py",
            "agent_envs/benchmark_pack.py",
            "agent_envs/benchmark_metrics.py",
            "agent_envs/benchmark_policy.py",
            "agent_envs/policy_process.py",
            "physics/hydraulics/euler_channel.py",
            "physics/hydraulics/hllc.py",
            "physics/hydraulics/euler_initialization.py",
            "sdk/euler_qualification.py",
            "core/network.py",
            "components/linear_thermal_network.py",
            "sdk/network_experiment.py",
            "physics/hydraulics/circulation_loop.py",
            "sdk/circulation_experiment.py",
            "scenarios/ai05_circulation_reversal.json",
            "physics/hydraulics/pump_loop.py",
            "sdk/pump_experiment.py",
            "scenarios/ai05_displacement_pump_trip.json",
            "physics/hydraulics/valved_pump_loop.py",
            "scenarios/ai05_valve_close_reopen.json",
            "physics/hydraulics/gas_storage_pair.py",
            "sdk/storage_experiment.py",
            "scenarios/ai05_storage_isolation.json",
            "physics/hydraulics/inertial_storage_pair.py",
            "scenarios/ai05_inertial_storage_damping.json",
            "physics/hydraulics/pump_storage_pair.py",
            "sdk/pump_storage_experiment.py",
            "scenarios/ai05_pump_storage_trip.json",
            "physics/hydraulics/check_pump_storage.py",
            "sdk/check_storage_experiment.py",
            "scenarios/ai05_check_valve_restart.json",
            "scenarios/ai05_thermal_two_capacity.json",
            "scenarios/ai05_thermal_star.json",
            "sdk/euler_experiment.py",
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
            "scenarios/pipe_wall_pulse_n4.yaml",
            "physics/hydraulics/initialization.py",
            "physics/hydraulics/reconstruction.py",
            "scenarios/pipe_muscl_pulse_8s.yaml",
            "physics/hydraulics/liquid_derivatives.py",
            "physics/hydraulics/channel_jacobian.py",
            "scenarios/batch11_muscl_n4_analytic.yaml",
            "physics/hydraulics/mechanical_energy.py",
            "physics/hydraulics/total_energy_initialization.py",
            "physics/hydraulics/muscl_initialization.py",
            "scenarios/batch12_total_wall_8s.yaml",
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
        subprocess.run(
            [
                str(python),
                "-c",
                SMOKE,
                site,
                expected_source,
                str(args.qualification.resolve()) if args.qualification else "",
            ],
            check=True,
            cwd=folder,
            env=clean_env,
        )
    print(json.dumps({"verified_wheel": str(wheel)}))


if __name__ == "__main__":
    main()
