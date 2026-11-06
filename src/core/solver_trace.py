"""Validate and aggregate returned solver meshes, without claiming hidden RHS counts."""

import math

from core.network import NetworkError, number

FIELDS = {
    "schema_version",
    "backend",
    "method",
    "start_time_s",
    "end_time_s",
    "max_step_s",
    "rtol",
    "atol_scale",
    "scipy_nfev",
    "scipy_njev",
    "scipy_nlu",
    "mesh_times_s",
}


def summarize_solver_trace(records, *, start_time_s, end_time_s):
    if not isinstance(records, list):
        raise NetworkError("Solver trace list required")
    start = number(start_time_s, "trace start")
    end = number(end_time_s, "trace end")
    if start < 0 or end < start:
        raise NetworkError("Reversed solver trace clock")
    summary = dict(
        available=bool(records),
        segments=len(records),
        scipy_segments=0,
        reported_mesh_steps=0,
        max_reported_step_s=0.0,
        scipy_nfev=0,
        scipy_njev=0,
        scipy_nlu=0,
    )
    previous = start
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != FIELDS
            or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or record["method"] not in ("Radau", "BDF")
            or record["backend"] not in ("scipy.solve_ivp", "boundary_hold")
        ):
            raise NetworkError("Unknown solver-trace schema/backend/method")
        a = number(record["start_time_s"], "segment start")
        b = number(record["end_time_s"], "segment end")
        step = number(record["max_step_s"], "solver cap", positive=True)
        rtol = number(record["rtol"], "solver rtol", positive=True)
        atol = number(record["atol_scale"], "solver atol scale", positive=True)
        if not 1e-12 <= rtol <= 1e-2 or not 1e-6 <= atol <= 1:
            raise NetworkError("Invalid solver-trace tolerance")
        if not math.isclose(a, previous, rel_tol=0, abs_tol=1e-10) or b < a or b > end + 1e-10:
            raise NetworkError("Solver-trace gaps, overlap or reversed segment")
        times = record["mesh_times_s"]
        if not isinstance(times, list) or not times:
            raise NetworkError("Nonempty returned mesh required")
        times = [number(t, "mesh time") for t in times]
        if times[0] != a or times[-1] != b:
            raise NetworkError("Returned mesh endpoints differ from segment")
        widths = [y - x for x, y in zip(times, times[1:])]
        if any(dt <= 0 or dt > step + 1e-10 for dt in widths):
            raise NetworkError("Returned mesh is unordered or exceeds configured cap")
        for key in ("scipy_nfev", "scipy_njev", "scipy_nlu"):
            if type(record[key]) is not int or record[key] < 0:
                raise NetworkError("Nonnegative integer SciPy counter required")
            summary[key] += record[key]
        if record["backend"] == "boundary_hold":
            if (
                a != b
                or len(times) != 1
                or any(record[k] for k in ("scipy_nfev", "scipy_njev", "scipy_nlu"))
            ):
                raise NetworkError("Boundary hold cannot claim solver work or time advance")
        elif b <= a or not widths or record["scipy_nfev"] == 0:
            raise NetworkError("Solver segment must report integration work")
        else:
            summary["scipy_segments"] += 1
        summary["reported_mesh_steps"] += len(widths)
        summary["max_reported_step_s"] = max(
            summary["max_reported_step_s"], max(widths, default=0.0)
        )
        previous = b
    if records and not math.isclose(previous, end, rel_tol=0, abs_tol=1e-10):
        raise NetworkError("Incomplete returned solver mesh coverage")
    return summary
