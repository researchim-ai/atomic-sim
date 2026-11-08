"""Source-bound closure passports and sampled applicability for qualified HEM inputs."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from atomic_model_hem_mixture.friction import SOURCE as FRICTION_SOURCE
from atomic_model_hem_qualified import closures, eos
from atomic_model_hem_qualified.cases import matrix, program
from atomic_model_hem_qualified.model import QualifiedHEMNetwork, signature
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json, read_json
from atomic_model_workspace.common import digest


def passports():
    return dict(
        eos=dict(source="https://iapws.org/technical-guidance/release/IF97-Rev",
                 pressure_pa=[eos.P_MIN, eos.P_MAX], temperature_k=[eos.T_MIN, eos.T_MAX],
                 regions=[1, 2, 4], scope="Adapter subset, not the entire IF97 domain"),
        friction=dict(source=FRICTION_SOURCE, models=list(closures.MODELS),
                      darcy_laws=["churchill", "colebrook"], diameter_m=[0.005, 0.3],
                      relative_roughness=[0, 0.05], required_flags=list(closures.FLAGS),
                      uncertainty="Model-choice spread is diagnostic; not a calibrated probability distribution"),
        boiling=dict(source=closures.CHEN_SOURCE, variant="chen-inl-documented-hfg025",
                     quality=[0, 0.1], diameter_m=[0.005, 0.0375], heated_length_m=[0.2, 6],
                     wall_superheat_k=dict(lower_exclusive=0, upper_inclusive=10),
                     liquid_reynolds=[1e4, 5e6], liquid_prandtl=[0.7, 100],
                     hydrostatic_to_pressure_ratio_upper_exclusive=0.01,
                     wall_temperature_k_upper_exclusive=647.096,
                     chf_fraction_upper_exclusive=0.8, required_flags=list(closures.HEAT_FLAGS),
                     uncertainty="Not calibrated; no confidence interval assigned"),
        chf_table1995=dict(source=closures.CHF_SOURCE, diameter_m=0.008,
                           disconnected_boxes=[dict(pressure_pa=[3e5, 5e5], mass_flux_kg_m2_s=[1000, 1500], quality=[0, 0.05]),
                                               dict(pressure_pa=[15e6, 16e6], mass_flux_kg_m2_s=[1000, 2000], quality=[0, 0.1])],
                           uncertainty="Tabulated correlation values; no measurement uncertainty assigned"),
        chf_biasi_inl=dict(source=closures.CHEN_SOURCE, pressure_pa=[0.27e6, 14e6],
                          mass_flux_kg_m2_s=[100, 600], quality=[0, 0.1],
                          diameter_m=[0.005, 0.0375], heated_length_m=[0.2, 6],
                          uncertainty="Restricted documented range; no confidence interval assigned"),
        excluded=["post-CHF", "critical nozzle flow", "slip/two-fluid", "metastable phase delay", "condensation heat-transfer correlation"],
    )


def sample(model):
    _, _, _, velocities, states, wall_temperatures, _ = model.primitive(model._y)
    friction, boiling = [], []
    for edge, indices in zip(model.edge_specs, model.edge_indices, strict=True):
        law = model.friction_laws.get(edge["id"])
        if law is None:
            continue
        for index in indices:
            values = closures.friction(states[index], float(velocities[index]), edge["diameter_m"], edge["roughness_m"], **law)
            alternate = dict(law, viscosity_model="cicchitti" if law["viscosity_model"] == "mcadams" else "mcadams")
            other = closures.friction(states[index], float(velocities[index]), edge["diameter_m"], edge["roughness_m"], **alternate)
            friction.append(dict(cell=model.names[index], selected=law, values=values,
                                 alternative_viscosity_model=alternate["viscosity_model"],
                                 absolute_force_density_spread_n_m3=abs(values["force_density_n_m3"] - other["force_density_n_m3"])))
    for j, law in enumerate(model.boiling_laws):
        if law is None:
            continue
        index = model.ports[j]
        edge = next(edge for edge, indices in zip(model.edge_specs, model.edge_indices, strict=True) if index in indices)
        state = states[index]
        G = state.density_kg_m3 * float(velocities[index])
        value = closures.boiling(state, G, edge["diameter_m"], float(wall_temperatures[j]), edge["length_m"], **law)
        boiling.append(dict(wall=model.wall_ids[j], cell=model.names[index], selected=law,
                            pressure_pa=state.pressure_pa, quality=state.quality, mass_flux_kg_m2_s=G,
                            wall_superheat_k=float(wall_temperatures[j] - state.temperature_k),
                            chf_fraction_slack=law["maximum_chf_fraction"] - value.chf_fraction,
                            values=asdict(value)))
    return dict(time_s=float(model._time), friction=friction, boiling=boiling,
                pressure_range_pa=[min(s.pressure_pa for s in states), max(s.pressure_pa for s in states)],
                temperature_range_k=[min(s.temperature_k for s in states), max(s.temperature_k for s in states)])


def audit(request, calls):
    if not isinstance(calls, list) or not 1 <= len(calls) <= 100:
        raise ValueError("Audit program requires 1..100 calls")
    source = signature()
    model = QualifiedHEMNetwork(request)
    samples = [sample(model)]
    for call in calls:
        if set(call) != {"duration_s", "command"} or call["command"] is not None:
            raise ValueError("Audit supports duration-only calls; no implicit control rewrite")
        model.advance(call["duration_s"])
        samples.append(sample(model))
    return dict(schema_version=1, source=source, driver_sha256=sha(__file__),
                request_sha256=digest(request), program=calls, passports=passports(),
                runtime_contract=model.contract(), sampled_applicability=samples,
                source_unchanged=source == signature(),
                scope="Accepted model stages enforce runtime domains; report summaries sample only requested times",
                uncertainty_status="unqualified; model-choice spread is not statistical uncertainty",
                validated=False, plant_referenced=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="JSON with request and program; omitted: eight built-in cases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.input:
        spec = read_json(args.input)
        if set(spec) != {"request", "program"}:
            raise ValueError("Expected request/program")
        reports = {"input": audit(spec["request"], spec["program"])}
    else:
        reports = {name: audit(request, program(name)) for name, request in matrix()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, seal(dict(cases=reports, validated=False, plant_referenced=False)))
    print(json.dumps(dict(cases=len(reports), source_unchanged=all(r["source_unchanged"] for r in reports.values()))))
    return 0 if all(r["source_unchanged"] for r in reports.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
