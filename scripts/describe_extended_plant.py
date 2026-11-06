"""Export the exact initialized reduced configuration and parameter provenance."""

import argparse
import json
from importlib.resources import files
from pathlib import Path

from atomic_model_plant_extension.initialization import solve
from atomic_model_plant_extension.pwr import SourceBoundPWR
from atomic_model_plant_extension.session import identity


def describe():
    p = SourceBoundPWR()
    initialization = solve(p)
    legacy = json.loads(
        files("validation").joinpath("ai07_lumped_configuration_v1.json").read_text()
    )
    return dict(
        schema_version=1,
        configuration_id="pwr-reduced-seven-bank-v1",
        source=identity(),
        model_id=p.passport.model_id,
        model_version=p.passport.version,
        parameter_provenance="Mixed-provenance reduced research model; not a BEAVRS or VVER reference plant",
        nominal={k: float(getattr(p, k)) for k in legacy["nominal"]},
        volumes={
            n: {
                k: float(getattr(v, k))
                for k in [
                    "volume_m3",
                    "elevation_m",
                    "mass_kg",
                    "internal_energy_j",
                    "pressure_pa",
                    "temperature_k",
                ]
            }
            for n, v in p.network.volumes.items()
        },
        junctions={n: vars(j) for n, j in p.network.junctions.items()},
        absorber=p.absorber_source,
        initialization=initialization,
        feed_density_ref_kg_m3=p.feed_density_ref,
        primary_link_density_refs_kg_m3=p.link_density_ref,
        protection="SCRAM latches rod insertion and effective turbine throttle0; operator request is retained",
        approximations=list(p.passport.approximations),
        validated=False,
        plant_referenced=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with a.output.open("x") as f:
        json.dump(describe(), f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
