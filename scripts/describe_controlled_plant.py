"""Export exact controlled-plant parameters, units-bearing stocks and sources."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from atomic_model_controlled_plant import Session
from atomic_model_controlled_plant.session import identity


def describe(mode):
    session = Session(initialization=mode)
    p = session.plant
    scalar = {
        key: value
        for key, value in vars(p).items()
        if not key.startswith("_") and type(value) in (str, bool, int, float)
    }
    return dict(
        schema_version=1,
        configuration_id="pwr-reduced-flow-controlled-" + mode + "-v1",
        source=identity(),
        session_configuration=session.config,
        model_id=p.passport.model_id,
        model_version=p.passport.version,
        phenomenon=p.passport.phenomenon,
        approximations=list(p.passport.approximations),
        parameter_provenance="Mixed-provenance reduced research configuration, not a plant reference",
        scalar_parameters_and_state=scalar,
        kinetics=vars(p.kinetics),
        decay_heat=vars(p.decay),
        volumes={
            name: {
                key: getattr(v, key)
                for key in [
                    "volume_m3",
                    "elevation_m",
                    "mass_kg",
                    "internal_energy_j",
                    "boron_kg",
                    "pressure_pa",
                    "temperature_k",
                ]
            }
            for name, v in p.network.volumes.items()
        },
        junctions={name: vars(link) for name, link in p.network.junctions.items()},
        sg_wall=vars(p.sg_wall),
        condenser=vars(p.condenser),
        absorber_source=p.absorber_source,
        feed_controller=p.feed_controller,
        maximum_feed_head_pa=p.maximum_feed_head_pa,
        feed_reference_density_kg_m3=p.feed_density_ref,
        primary_reference_densities_kg_m3=p.link_density_ref,
        initialization=session.initialization,
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        validated=False,
        plant_referenced=False,
    )


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initialization", choices=["nominal", "natural"], default="nominal")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    r = describe(a.initialization)
    with a.output.open("x") as f:
        json.dump(r, f, indent=2, default=serial, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
