"""Independent closed-form homogeneous reservoir with a source switch."""

import argparse
import json
import math
from pathlib import Path

from physics.hydraulics.isobaric import ChannelConfig
from physics.hydraulics.wall_channel import WallChannel, WallConfig
from physics.materials.iapws import saturation_properties


def verify():
    liquid, vapor = saturation_properties(7e6)
    h0 = float(liquid.h_j_kg + 0.1 * (vapor.h_j_kg - liquid.h_j_kg))
    v0 = float(liquid.v_m3_kg + 0.1 * (vapor.v_m3_kg - liquid.v_m3_kg))
    b = float((vapor.v_m3_kg - liquid.v_m3_kg) / (vapor.h_j_kg - liquid.h_j_kg))
    # P=UA*50 until .7 s, then exponential wall cooling with tau=C/UA=10 s.
    transferred = 5000 * 0.7 + 1000 * 50 * (1 - math.exp(-19.3 / 10))
    volume = 0.01
    v = v0 * math.exp(b * transferred / volume)
    exact = dict(
        enthalpy_j_kg=h0 + (v - v0) / b,
        wall_temperature_k=float(liquid.T_k) + 50 * math.exp(-19.3 / 10),
        stored_mass_kg=volume / v,
    )
    cases = []
    for rtol in (1e-3, 1e-5, 1e-7):
        c = WallChannel(
            WallConfig(
                ChannelConfig(1, volume, 7e6, h0, 0.0, 5000.0),
                1000.0,
                100.0,
                (float(liquid.T_k) + 50,),
                0.7,
            )
        )
        c.enable_continuation(20.0)
        row = c.advance(20.0, rtol=rtol, method="Radau", max_step_s=20.0)
        cases.append(
            dict(
                rtol=rtol,
                enthalpy_error_j_kg=abs(row["outlet_enthalpy_j_kg"] - exact["enthalpy_j_kg"]),
                wall_temperature_error_k=abs(
                    row["max_wall_temperature_k"] - exact["wall_temperature_k"]
                ),
                mass_error_kg=abs(row["stored_mass_kg"] - exact["stored_mass_kg"]),
                energy_residual_j=float(abs(row["energy_residual_j"])),
            )
        )
    errors = ("enthalpy_error_j_kg", "wall_temperature_error_k", "mass_error_kg")
    decreasing = all(cases[i + 1][key] < cases[i][key] for i in range(2) for key in errors)
    fine = cases[-1]
    return dict(
        kind="numerical_verification",
        validated=False,
        plant_referenced=False,
        scope="Synthetic constant-pressure homogeneous boiling reservoir, prescribed C and UA",
        exact=exact,
        cases=cases,
        passed=decreasing
        and fine["enthalpy_error_j_kg"] < 0.02
        and fine["wall_temperature_error_k"] < 2e-6
        and fine["energy_residual_j"] < 0.1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
