"""Conservative steady rod/channel/neutronics fixed point with transactional state."""

from copy import deepcopy

import numpy as np

from atomic_model_spatial import Model, eigenvalue
from atomic_model_training.spec import integer
from physics.thermal.fuel_rod import FuelRod

from .feedback import FeedbackTable
from .provenance import seal, unseal


def overlaps(source_faces, target_faces):
    a = np.asarray(source_faces, float)
    b = np.asarray(target_faces, float)
    if (
        a.ndim != 1
        or b.ndim != 1
        or min(len(a), len(b)) < 2
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or np.any(np.diff(a) <= 0)
        or np.any(np.diff(b) <= 0)
        or a[0] != b[0]
        or a[-1] != b[-1]
    ):
        raise ValueError("Conservative maps require matching, increasing physical domains")
    return np.maximum(
        0, np.minimum(b[1:, None], a[None, 1:]) - np.maximum(b[:-1, None], a[None, :-1])
    )


def map_power(source_faces, target_faces, powers):
    matrix = overlaps(source_faces, target_faces)
    q = np.asarray(powers, float)
    if q.shape != (matrix.shape[1],) or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Invalid cell-integrated power")
    return matrix @ (q / np.diff(source_faces))


def map_temperature(source_faces, target_faces, temperature):
    matrix = overlaps(source_faces, target_faces)
    t = np.asarray(temperature, float)
    if t.shape != (matrix.shape[1],) or not np.isfinite(t).all() or np.any(t <= 0):
        raise ValueError("Invalid cell temperatures")
    return matrix @ t / np.diff(target_faces)


def rod(length):
    return FuelRod(
        0.004,
        0.0041,
        0.00475,
        12,
        3,
        float(length),
        1,
        10970.0,
        330.0,
        6500.0,
        330.0,
        3.0,
        16.0,
        8000.0,
        20000.0,
    )


def thermal(channel_faces, powers, *, inlet_k=560.0, mass_flow_kg_s=0.3, cp_j_kg_k=4200.0):
    q = np.asarray(powers, float)
    faces = np.asarray(channel_faces, float)
    if (
        q.ndim != 1
        or faces.ndim != 1
        or len(faces) != len(q) + 1
        or not np.isfinite(faces).all()
        or not all(
            type(v) in (int, float) and np.isfinite(v) for v in (inlet_k, mass_flow_kg_s, cp_j_kg_k)
        )
        or np.any(np.diff(faces) <= 0)
        or np.any(q < 0)
        or not np.isfinite(q).all()
        or min(inlet_k, mass_flow_kg_s, cp_j_kg_k) <= 0
    ):
        raise ValueError("Invalid thermal channel")
    outlet = inlet_k + np.cumsum(q) / (mass_flow_kg_s * cp_j_kg_k)
    entrance = np.concatenate(([inlet_k], outlet[:-1]))
    coolant = (entrance + outlet) / 2
    fuel = []
    boundary = []
    profiles = []
    for length, heat, tc in zip(np.diff(faces), q, coolant):
        r = rod(length)
        generation = heat / sum(r.volumes()[: r.n_fuel])
        temperatures = r.steady_state(generation, tc)
        profiles.append(temperatures.tolist())
        fuel.append(float(np.average(temperatures[: r.n_fuel], weights=r.volumes()[: r.n_fuel])))
        boundary.append(r.boundary_heat_w(temperatures, tc))
    defect = abs(sum(boundary) - sum(q)) / max(1.0, sum(q))
    channel_defect = abs(mass_flow_kg_s * cp_j_kg_k * (outlet[-1] - inlet_k) - sum(q)) / max(
        1.0, sum(q)
    )
    if max(defect, channel_defect) > 1e-9:
        raise ValueError("Thermal energy balance failed")
    return dict(
        fuel_temperature_k=fuel,
        coolant_temperature_k=coolant.tolist(),
        channel_outlet_k=outlet.tolist(),
        rod_temperature_k=profiles,
        boundary_heat_w=boundary,
        energy_relative=max(defect, channel_defect),
    )


def solve(
    config,
    table_specification,
    *,
    total_power_w=5000.0,
    channel_faces=None,
    max_iterations=200,
    temperature_tolerance_k=1e-5,
    power_tolerance=1e-6,
    inlet_k=560.0,
    mass_flow_kg_s=0.3,
    heat_response_per_m=None,
):
    model = Model(config)
    table = FeedbackTable(table_specification)
    if heat_response_per_m is None:
        if not config["provenance"].startswith("Synthetic"):
            raise ValueError("Evaluated coupling requires explicit kappa-fission heat response")
        heat_response = model.nu_fission_per_m
    else:
        heat_response = np.asarray(heat_response_per_m, float)
        if (
            heat_response.shape != (model.g, model.n)
            or not np.isfinite(heat_response).all()
            or np.any(heat_response < 0)
            or not np.any(heat_response > 0)
        ):
            raise ValueError("Invalid heat-response data")
    integer(max_iterations, 1, 1000)
    if (
        type(total_power_w) not in (float, int)
        or not np.isfinite(total_power_w)
        or total_power_w <= 0
    ):
        raise ValueError("Positive finite thermal power required")
    if not 0 < temperature_tolerance_k <= 1e-5 or not 0 < power_tolerance <= 1e-6:
        raise ValueError("Coupling tolerances may not be relaxed")
    faces = model.faces if channel_faces is None else np.asarray(channel_faces, float)
    overlaps(model.faces, faces)
    fuel = np.full(model.n, inlet_k)
    cool = np.full(model.n, inlet_k)
    previous = np.full(model.n, total_power_w / model.n)
    records = []
    for iteration in range(1, max_iterations + 1):
        multiplier = np.array(
            [
                table(
                    fuel_temperature_k=tf,
                    coolant_temperature_k=tc,
                    density_kg_m3=750.0,
                    boron_ppm=0.0,
                    composition_fraction=0.0,
                )
                for tf, tc in zip(fuel, cool)
            ]
        )
        c = deepcopy(config)
        c["absorption_per_m"] = (model.absorption_per_m * multiplier).tolist()
        eig = eigenvalue(Model(c))
        weights = np.sum(heat_response * np.array(eig["flux"]), axis=0) * model.width
        powers = weights / weights.sum() * total_power_w
        target = map_power(model.faces, faces, powers)
        temperatures = thermal(faces, target, inlet_k=inlet_k, mass_flow_kg_s=mass_flow_kg_s)
        updated = map_temperature(faces, model.faces, temperatures["fuel_temperature_k"])
        coolant = map_temperature(faces, model.faces, temperatures["coolant_temperature_k"])
        temperature_error = float(max(np.max(abs(updated - fuel)), np.max(abs(coolant - cool))))
        power_error = float(np.max(abs(powers - previous)) / total_power_w)
        records.append(
            dict(
                iteration=iteration,
                temperature_error_k=temperature_error,
                power_relative=power_error,
            )
        )
        if temperature_error <= temperature_tolerance_k and power_error <= power_tolerance:
            return dict(
                k=eig["k"],
                flux=eig["flux"],
                cell_power_w=powers.tolist(),
                fuel_temperature_k=updated.tolist(),
                coolant_temperature_k=coolant.tolist(),
                thermal=temperatures,
                iterations=records,
                feedback_sha256=table.sha256,
                total_power_w=total_power_w,
                fidelity="steady-1d-diffusion-radial-rod-constant-cp-channel",
                nuclear_family=table.specification["family"],
                power_shape_assumption=(
                    "constant energy per produced neutron; synthetic coupling fixture"
                    if heat_response_per_m is None
                    else "explicit supplied relative heat-response field"
                ),
                validated=False,
                plant_referenced=False,
            )
        fuel = 0.5 * fuel + 0.5 * updated
        cool = 0.5 * cool + 0.5 * coolant
        previous = powers
    raise ValueError("Coupled fixed point did not converge; no state accepted")


class SpatialSession:
    def __init__(self, config, feedback):
        Model(config)
        FeedbackTable(feedback)
        self.config = deepcopy(config)
        self.feedback = deepcopy(feedback)
        self.history = []
        self.revision = 0

    def advance(self, power_w, *, expected_revision):
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise ValueError("Stale spatial revision")
        result = solve(self.config, self.feedback, total_power_w=power_w)
        self.history.append(dict(power_w=power_w, result=result))
        self.revision += 1
        return result

    def checkpoint(self):
        from .provenance import identity

        return seal(
            dict(
                config=self.config,
                feedback=self.feedback,
                history=self.history,
                source=identity(),
                revision=self.revision,
            )
        )

    @classmethod
    def restore(cls, saved):
        from .provenance import identity

        p = unseal(saved)
        if (
            p["source"] != identity()
            or p["revision"] != len(p["history"])
            or len(p["history"]) > 256
        ):
            raise ValueError("Spatial restart source/history mismatch")
        session = cls(p["config"], p["feedback"])
        for entry in p["history"]:
            if (
                session.advance(entry["power_w"], expected_revision=session.revision)
                != entry["result"]
            ):
                raise ValueError("Spatial restart replay mismatch")
        if session.checkpoint() != saved:
            raise ValueError("Spatial restart differs")
        return session
