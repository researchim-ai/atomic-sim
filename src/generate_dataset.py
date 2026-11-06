#!/usr/bin/env python3
"""
Генератор датасетов для обучения LLM/RL моделей
Создает большие объемы размеченных траекторий симуляции
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

from core.status import DomainError, SolverError
from reactor import ReactorSimulator
from sdk.api import create_session, step_session


def generate_vver_trajectory(seed: int, duration_s: float, step_s: float, scenario_type: str) -> list:
    """Траектория паспорта ВВЭР. Поля, которых в сосредоточенной модели нет, не заполняются."""
    rng = np.random.default_rng(seed)
    plant = create_session("vver")
    nominal = plant.fission_power_nom
    records = []
    time_s = 0.0
    step_index = 0
    while time_s < duration_s - 1.0e-12:
        commands = {}
        if scenario_type == "rod":
            commands["rod_command"] = float(rng.uniform(-0.05, 0.05))
        elif scenario_type == "trip" and time_s >= 0.5 * duration_s:
            commands["throttle"] = 0.0
        elif scenario_type == "coastdown" and time_s >= duration_s / 3.0:
            commands["motor_off"] = True
        out_of_domain = False
        message = ""
        try:
            observed = step_session(plant, step_s, commands)
        except (DomainError, SolverError) as exc:
            out_of_domain = True
            message = str(exc)
            observed = plant.observation()
        records.append(
            {
                "step": step_index,
                "time": observed["time_s"],
                "obs": {
                    "power_MW": observed["thermal_power_w"] / 1.0e6,
                    "T_fuel_C": observed["fuel_temperature_k"] - 273.15,
                    "rod_position_pct": observed["rod_insertion"] * 100.0,
                    "flow_rate": observed["mass_flow_kg_s"],
                    "steam_flow_kg_s": observed["steam_flow_kg_s"],
                    "sg_pressure_MPa": observed["sg_pressure_pa"] / 1.0e6,
                    "electric_MW": observed["electrical_power_w"] / 1.0e6,
                },
                "action": commands,
                "labels": {
                    "safe": not out_of_domain,
                    "out_of_domain": out_of_domain,
                    "message": message,
                    "scram_latched": observed["scram_latched"],
                },
                "meta": {
                    "scenario_type": scenario_type,
                    "seed": int(seed),
                    "core": "plants.vver.VVERPlant",
                    "core_version": plant.passport.version,
                    "model_id": plant.passport.model_id,
                    "nominal_fission_MW": nominal / 1.0e6,
                },
            }
        )
        if out_of_domain:
            break
        time_s += step_s
        step_index += 1
    return records


class DatasetGenerator:
    """Генератор датасетов из симуляций реактора"""

    def __init__(self, output_dir="./datasets", seed=None):
        """
        Args:
            output_dir: директория для сохранения датасетов
            seed: seed для воспроизводимости
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed = seed if seed is not None else np.random.randint(0, 2**31)
        np.random.seed(self.seed)

        # Метаданные датасета
        self.metadata = {
            "generator_version": "2.0.0",
            "created_at": datetime.now().isoformat(),
            "seed": self.seed,
        }

    def generate_scenario(self, scenario_type="random", duration=100.0, scenario_index=0):
        """Сгенерировать один сценарий со своим RNG."""
        scenario_seed = int((self.seed + scenario_index) % (2**31))
        rng = np.random.default_rng(scenario_seed)
        sim = ReactorSimulator()
        sim.reset()

        trajectory = []
        steps = int(duration / sim.dt)

        if scenario_type == "random":
            # Случайные действия
            for i in tqdm(range(steps), desc="Random scenario", leave=False):
                action = int(rng.choice([-1, 0, 1]))
                state = sim.step(manual_rod_direction=action)

                if i % 10 == 0:  # Сэмплинг каждые 10 шагов
                    trajectory.append(self._state_to_record(state, action, i))

                if not state["safe"]:
                    break

        elif scenario_type == "auto_control":
            # Автоматическое управление
            target_power = 80.0 + float(rng.random()) * 40.0  # 80-120 МВт
            sim.set_auto_control(enabled=True, target_power=target_power)

            for i in tqdm(range(steps), desc=f"Auto control ({target_power:.0f} МВт)", leave=False):
                state = sim.step()

                if i % 10 == 0:
                    trajectory.append(self._state_to_record(state, None, i))

                if not state["safe"]:
                    break

        elif scenario_type == "ramp":
            # Плавное изменение мощности
            start_power = 100.0
            end_power = 50.0 + float(rng.random()) * 100.0

            for i in tqdm(
                range(steps), desc=f"Ramp {start_power:.0f}→{end_power:.0f} МВт", leave=False
            ):
                # Линейная интерполяция целевой мощности
                progress = i / steps
                target = start_power + (end_power - start_power) * progress

                sim.set_auto_control(enabled=True, target_power=target)
                state = sim.step()

                if i % 10 == 0:
                    trajectory.append(self._state_to_record(state, None, i))

                if not state["safe"]:
                    break

        elif scenario_type == "emergency":
            # Аварийная ситуация: потеря охлаждения
            # Разгон до высокой мощности
            sim.set_auto_control(enabled=True, target_power=150.0)
            for i in tqdm(range(steps // 3), desc="Emergency: ramp up", leave=False):
                state = sim.step()
                if i % 10 == 0:
                    trajectory.append(self._state_to_record(state, None, i))

            # Потеря охлаждения
            sim.set_coolant_flow(0.3)
            sim.set_auto_control(enabled=False)

            for i in tqdm(range(steps // 3, steps), desc="Emergency: loss of cooling", leave=False):
                state = sim.step(manual_rod_direction=0)

                if i % 10 == 0:
                    trajectory.append(self._state_to_record(state, 0, i))

                # SCRAM при критической температуре
                if state["T_fuel"] > 900.0:
                    sim.scram()
                    trajectory.append(self._state_to_record(state, "SCRAM", i))

                if not state["safe"]:
                    break

        return trajectory

    def _state_to_record(self, state, action, step):
        """Преобразовать состояние в запись датасета"""
        record = {
            "step": step,
            "time": state["time"],
            "obs": {
                "power_MW": state["power"],
                "T_fuel_C": state["T_fuel"],
                "T_clad_C": state["T_clad"],
                "T_coolant_C": state["T_coolant"],
                "rod_reactivity_beta": state["rod_reactivity"],
                "rod_position_pct": state["avg_rod_position"] * 100,
                "flow_rate": state["flow_rate"],
            },
            "action": action,
            "labels": {
                "safe": state["safe"],
                "fuel_overheat": state.get("fuel_overheat", False),
                "coolant_boiling": state.get("coolant_boiling", False),
            },
        }
        return record

    def generate_batch(self, n_scenarios=100, scenario_types=None, duration=100.0):
        """
        Сгенерировать батч сценариев

        Args:
            n_scenarios: количество сценариев
            scenario_types: список типов сценариев
            duration: длительность каждого сценария

        Returns:
            all_trajectories: список всех траекторий
        """
        if scenario_types is None:
            scenario_types = ["random", "auto_control", "ramp", "emergency"]

        all_trajectories = []

        for i in tqdm(range(n_scenarios), desc="Generating scenarios"):
            scenario_type = np.random.choice(scenario_types)
            trajectory = self.generate_scenario(scenario_type, duration, scenario_index=i)

            # Добавляем метаданные сценария
            scenario_meta = {
                "scenario_id": i,
                "scenario_type": scenario_type,
                "seed": int((self.seed + i) % (2**31)),
                "core": "reactor.ReactorSimulator",
                "core_version": "2.0.0",
            }

            for record in trajectory:
                record["meta"] = scenario_meta

            all_trajectories.extend(trajectory)

        return all_trajectories

    def save_dataset(self, trajectories, name="reactor_dataset", format="parquet"):
        """
        Сохранить датасет

        Args:
            trajectories: список траекторий
            name: имя датасета
            format: формат ('parquet', 'jsonl', 'csv')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}"

        # Преобразовать в DataFrame
        records = []
        for traj in trajectories:
            flat_record = {
                "scenario_id": traj["meta"]["scenario_id"],
                "scenario_type": traj["meta"]["scenario_type"],
                "step": traj["step"],
                "time": traj["time"],
                "power_MW": traj["obs"]["power_MW"],
                "T_fuel_C": traj["obs"]["T_fuel_C"],
                "T_clad_C": traj["obs"]["T_clad_C"],
                "T_coolant_C": traj["obs"]["T_coolant_C"],
                "rod_reactivity_beta": traj["obs"]["rod_reactivity_beta"],
                "rod_position_pct": traj["obs"]["rod_position_pct"],
                "flow_rate": traj["obs"]["flow_rate"],
                "action": str(traj["action"]),
                "safe": traj["labels"]["safe"],
                "fuel_overheat": traj["labels"]["fuel_overheat"],
                "coolant_boiling": traj["labels"]["coolant_boiling"],
            }
            records.append(flat_record)

        df = pd.DataFrame(records)

        # Сохранить в выбранном формате
        if format == "parquet":
            filepath = self.output_dir / f"{filename}.parquet"
            df.to_parquet(filepath, index=False)
        elif format == "jsonl":
            filepath = self.output_dir / f"{filename}.jsonl"
            df.to_json(filepath, orient="records", lines=True)
        elif format == "csv":
            filepath = self.output_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)

        # Сохранить метаданные
        meta_filepath = self.output_dir / f"{filename}_metadata.json"
        metadata = {
            **self.metadata,
            "n_records": len(records),
            "n_scenarios": len(set(r["scenario_id"] for r in records)),
            "format": format,
            "filepath": str(filepath),
        }

        with open(meta_filepath, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Датасет сохранен:")
        print(f"  Файл: {filepath}")
        print(f"  Записей: {len(records)}")
        print(f"  Сценариев: {metadata['n_scenarios']}")
        print(f"  Метаданные: {meta_filepath}")

        return filepath


def main():
    parser = argparse.ArgumentParser(description="Генератор датасетов симулятора реактора")
    parser.add_argument("--n-scenarios", type=int, default=10, help="Количество сценариев")
    parser.add_argument(
        "--duration", type=float, default=100.0, help="Длительность каждого сценария (сек)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./datasets", help="Директория для сохранения"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="parquet",
        choices=["parquet", "jsonl", "csv"],
        help="Формат файла",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--name", type=str, default="reactor_dataset", help="Имя датасета")
    parser.add_argument("--engine", choices=["vver", "legacy"], default="vver")
    parser.add_argument("--step", type=float, default=0.5, help="Шаг ядра ВВЭР, с")

    args = parser.parse_args()

    print("=" * 70)
    print("ГЕНЕРАТОР ДАТАСЕТОВ СИМУЛЯТОРА РЕАКТОРА")
    print("=" * 70)
    print(f"\nПараметры:")
    print(f"  Сценариев: {args.n_scenarios}")
    print(f"  Длительность: {args.duration} сек")
    print(f"  Формат: {args.format}")
    print(f"  Seed: {args.seed}")

    if args.engine == "vver":
        seed0 = 0 if args.seed is None else int(args.seed)
        kinds = ["rod", "trip", "coastdown"]
        trajectories = []
        for index in range(args.n_scenarios):
            kind = kinds[index % len(kinds)]
            trajectories.extend(
                generate_vver_trajectory(seed0 + index, args.duration, args.step, kind)
            )
        destination = Path(args.output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        outfile = destination / f"{args.name}_vver.jsonl"
        with outfile.open("w", encoding="utf-8") as handle:
            for row in trajectories:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nЯдро: plants.vver.VVERPlant")
        print(f"  Записей: {len(trajectories)}")
        print(f"  Файл: {outfile}")
        return

    generator = DatasetGenerator(output_dir=args.output_dir, seed=args.seed)
    print(f"\nГенерация legacy ReactorSimulator...")
    trajectories = generator.generate_batch(n_scenarios=args.n_scenarios, duration=args.duration)
    generator.save_dataset(trajectories, name=args.name, format=args.format)

    print("\n✓ Генерация завершена!")


if __name__ == "__main__":
    main()
