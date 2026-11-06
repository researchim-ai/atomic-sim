"""Measured rarefaction vs bulk EOS screen; no threshold chosen after seeing data."""

import csv
import io
from pathlib import Path

from atomic_model_hem_qualified import eos

from .importer import finite, sha

PIN = "8d74f91af068fc597e263a3a7b742f769e2750b6221229e40758d5eefe6de263"
DATA = Path(__file__).parent / "data" / "alamgir"


def source_identity():
    return {p.name: sha(p.read_bytes()) for p in sorted(DATA.iterdir()) if p.is_file()}


def screen():
    raw = (DATA / "table1.csv").read_bytes()
    if sha(raw) != PIN:
        raise ValueError("Published acoustic data changed")
    observations = list(csv.DictReader(io.StringIO(raw.decode())))
    if [r["run"] for r in observations] != ["HW-18", "HW-17", "HW-20", "HW-27"]:
        raise ValueError("Published run selection changed")
    rows = []
    for r in observations:
        p = finite(r["initial_pressure_atm"]) * 101325
        measured = finite(r["rarefaction_speed_m_per_s"])
        for sensor in ("PT1", "PT3"):
            t = finite(r["initial_temperature_" + sensor + "_degC"]) + 273.15

            def speed(pp, tt):
                state = eos.pt(pp, tt)
                if state.region != 1:
                    raise ValueError("Acoustic screen requires subcooled liquid")
                return eos.derivatives(state)["equilibrium_sound_speed_squared_m2_s2"] ** 0.5

            predicted = speed(p, t)
            samples = [
                dict(pressure_pa=pp, temperature_k=tt, bulk_sound_speed_m_s=speed(pp, tt))
                for pp in (p * 0.98, p * 1.02)
                for tt in (t - 0.6, t + 0.6)
            ]
            rows.append(
                dict(
                    run=r["run"],
                    sensor=sensor,
                    pressure_pa=p,
                    temperature_k=t,
                    measured_run_rarefaction_speed_m_s=measured,
                    bulk_sound_speed_m_s=predicted,
                    residual_m_s=predicted - measured,
                    relative_residual=predicted / measured - 1,
                    input_sensitivity_samples=samples,
                )
            )
    return dict(
        source_url="https://engines.egr.uh.edu/sites/engines/files/talks/ExperimentalStudyOfDepressurization.pdf",
        source_table="Table1 printed435 / PDF page3",
        data_sha256=PIN,
        independent_experiments=4,
        local_eos_estimates=8,
        rows=rows,
        maximum_absolute_relative_residual=max(abs(r["relative_residual"]) for r in rows),
        comparison_performed=True,
        acceptance_threshold=None,
        physical_acceptance=False,
        wave_speed_measurement_uncertainty=None,
        interpretation="Descriptive bulk-EOS/pipe-rarefaction screen; repeated run speeds are not independent sensor measurements. Four p/T samples are sensitivity, not a rigorous uncertainty envelope or confidence interval. Pipe compliance and axial profiles unmodeled.",
    )
