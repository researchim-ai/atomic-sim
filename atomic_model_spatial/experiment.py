"""Strict JSON requests and portable source-bound result envelopes."""

import numpy as np

from .diagnostics import face_currents
from .model import Model
from .solvers import eigenvalue, fixed_source
from .transient import checkpoint, restore, start, step
from .workflow import identity


def execute(request, saved=None):
    if not isinstance(request, dict) or request.get("schema_version") != 1:
        raise ValueError("Unsupported request schema")
    mode = request.get("mode")
    fields = dict(
        eigen={"schema_version", "mode", "model"},
        fixed_source={"schema_version", "mode", "model", "source"},
        transient={
            "schema_version",
            "mode",
            "model",
            "initial_flux",
            "initial_precursors",
            "steps",
        },
    )
    if mode not in fields or set(request) != fields[mode]:
        raise ValueError("Unexpected request fields or mode")
    model = Model(request["model"])
    if mode == "eigen":
        if saved is not None:
            raise ValueError("Restart is only supported for transients")
        result = eigenvalue(model)
    elif mode == "fixed_source":
        if saved is not None:
            raise ValueError("Restart is only supported for transients")
        result = fixed_source(model, request["source"])
    else:
        schedule = request["steps"]
        if not isinstance(schedule, list) or not 1 <= len(schedule) <= 10000:
            raise ValueError("Require 1..10000 explicit source/timestep records")
        if saved is None:
            state = start(model, request["initial_flux"], request["initial_precursors"])
        else:
            if request["initial_flux"] is not None or request["initial_precursors"] is not None:
                raise ValueError("Restart request must use null initial populations")
            state = restore(model, saved)
        for record in schedule:
            if not isinstance(record, dict) or set(record) != {"dt_s", "source"}:
                raise ValueError("Malformed schedule entry")
            state = step(model, state, record["dt_s"], record["source"])
        result = dict(
            checkpoint=checkpoint(model, state),
            time_s=state.time_s,
            flux=np.array(state.flux).tolist(),
            precursors=np.array(state.precursors).tolist(),
        )
    result["face_currents"] = face_currents(model, result["flux"]).tolist()
    result["face_current_sign"] = (
        "positive towards increasing x; reflecting=0, zero_flux at physical face"
    )
    return dict(
        schema_version=1,
        source=identity(),
        mode=mode,
        result=result,
        input=request,
        validated=False,
        plant_referenced=False,
        fidelity="1D slab diffusion; supplied macroscopic data, no thermal feedback",
        units=dict(
            length="m",
            diffusion="m",
            macroscopic_cross_section="1/m",
            speed="m/s",
            time="s",
            decay="1/s",
            source="neutron/(m^3 s)",
            flux="source-normalized" if mode == "eigen" else "neutron/(m^2 s)",
            face_current="source-normalized" if mode == "eigen" else "neutron/(m^2 s)",
            precursors="1/m^3",
            geometry_area="1 m^2",
        ),
    )
