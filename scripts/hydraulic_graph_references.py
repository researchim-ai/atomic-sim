"""Independent mixed pressure/flow equations, without the production inverse law."""

import numpy as np
from scipy.optimize import root


def reference(spec):
    nodes, edges = spec["nodes"], spec["edges"]
    indices = {n["id"]: i for i, n in enumerate(nodes)}
    free = [i for i, n in enumerate(nodes) if "pressure_pa" not in n]
    pressures = np.array([n.get("pressure_pa", 0) for n in nodes], dtype=float)
    anchor = next(n["pressure_pa"] for n in nodes if "pressure_pa" in n)
    incidence = np.zeros((len(nodes), len(edges)))
    constants, linear, quadratic = [], [], []
    for j, edge in enumerate(edges):
        source, target = indices[edge["from"]], indices[edge["to"]]
        incidence[source, j], incidence[target, j] = 1, -1
        constants.append(
            edge["pressure_gain_pa"]
            - spec["density_kg_m3"]
            * spec["gravity_m_s2"]
            * (nodes[target]["elevation_m"] - nodes[source]["elevation_m"])
        )
        linear.append(edge["linear_pa_s_kg"])
        quadratic.append(edge["quadratic_pa_s2_kg2"])
    linear, quadratic = np.array(linear), np.array(quadratic)
    inject = np.array([nodes[i]["injection_kg_s"] for i in free])
    k = len(free)

    def residual(x):
        p = pressures.copy()
        p[free] = anchor + x[:k] * 1e5
        q = x[k:] * 10
        return np.r_[
            (incidence[free] @ q - inject) / 10,
            (incidence.T @ p + constants - linear * q - quadratic * q * np.abs(q)) / 1e5,
        ]

    def jacobian(x):
        return np.block(
            [
                [np.zeros((k, k)), incidence[free]],
                [incidence[free].T, np.diag(-(linear + 2 * quadratic * np.abs(x[k:] * 10)) * 1e-4)],
            ]
        )

    solved = root(
        residual,
        np.zeros(k + len(edges)),
        jac=jacobian,
        method="hybr",
        options=dict(xtol=1e-11, maxfev=10000),
    )
    # MINPACK's step-size stop is not a dimensional acceptance criterion.
    # Polish the full mixed system (no eliminated edge inverse) to roundoff.
    for _ in range(8):
        errors = residual(solved.x)
        if (
            np.max(np.abs(errors[:k]), initial=0.0) * 10 <= 1e-10
            and np.max(np.abs(errors[k:])) * 1e5 <= 1e-7
        ):
            break
        solved.x += np.linalg.solve(jacobian(solved.x), -errors)
    errors = residual(solved.x)
    # Some exact roots end with MINPACK's no-progress flag; admission depends on
    # the original dimensional residuals, not on that flag alone.
    mass_error = float(np.max(np.abs(errors[:k])) * 10) if k else 0.0
    pressure_error = float(np.max(np.abs(errors[k:])) * 1e5)
    if mass_error > 1e-10 or pressure_error > 1e-7 or not np.all(np.isfinite(solved.x)):
        raise ValueError(
            f"Independent mixed-system reference failed: mass={mass_error}, pressure={pressure_error}"
        )
    pressures[free] = anchor + solved.x[:k] * 1e5
    return dict(
        pressures_pa={n["id"]: float(pressures[i]) for i, n in enumerate(nodes)},
        flows_kg_s={e["id"]: float(solved.x[k + i] * 10) for i, e in enumerate(edges)},
        mass_residual_kg_s=mass_error,
        pressure_residual_pa=pressure_error,
        solver_success=bool(solved.success),
    )


def cases():
    """Deterministic manufactured solutions plus simple exact limiting cases."""
    for size in (3, 8, 16, 32, 64, 128):
        for mode in ("linear", "mixed", "near_zero"):
            rng = np.random.default_rng(701 + size)
            rho, gravity = 997.0, 9.80665
            pressure = 1e6 + rng.uniform(-5000, 5000, size)
            heights = rng.uniform(-5, 5, size)
            pairs = [(i, i + 1) for i in range(size - 1)]
            pairs += [(i, i + 2) for i in range(size - 2)]
            pairs += [(size - 1, 0)]
            balances = np.zeros(size)
            edges, expected = [], {}
            for j, (source, target) in enumerate(pairs):
                flow = float(rng.uniform(-20, 20))
                if mode == "near_zero" and j % 3 == 0:
                    flow = 0.0 if j % 2 == 0 else 1e-10
                linear = float(rng.uniform(100, 500))
                quadratic = 0.0 if mode == "linear" else float(rng.uniform(1, 20))
                head = (
                    linear * flow
                    + quadratic * flow * abs(flow)
                    - (pressure[source] - pressure[target])
                    + rho * gravity * (heights[target] - heights[source])
                )
                edges.append(
                    dict(
                        id="edge" + str(j),
                        **{"from": "node" + str(source), "to": "node" + str(target)},
                        linear_pa_s_kg=linear,
                        quadratic_pa_s2_kg2=quadratic,
                        pressure_gain_pa=float(head),
                    )
                )
                balances[source] += flow
                balances[target] -= flow
                expected["edge" + str(j)] = flow
            nodes = []
            for i in range(size):
                node = dict(id="node" + str(i), elevation_m=float(heights[i]))
                if i in (0, size // 2, size - 1):
                    node["pressure_pa"] = float(pressure[i])
                else:
                    node.update(
                        injection_kg_s=float(balances[i]),
                        initial_pressure_pa=float(rng.uniform(1e4, 2e7)),
                    )
                nodes.append(node)
            yield (
                str(size) + "_" + mode,
                dict(
                    format="atomic-hydraulic-steady",
                    schema_version=1,
                    density_kg_m3=rho,
                    gravity_m_s2=gravity,
                    nodes=nodes,
                    edges=edges,
                ),
                dict(
                    pressures_pa={n["id"]: float(pressure[i]) for i, n in enumerate(nodes)},
                    flows_kg_s=expected,
                ),
            )
