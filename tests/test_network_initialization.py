"""Physical residual witnesses, independent branches and rejected certificates."""

import copy
import json

import numpy as np
import pytest

from atomic_model_initialization import audit_certificate, initialize
from atomic_model_initialization.__main__ import main
from atomic_model_initialization.qualification import qualify
from atomic_model_initialization.steady import Equations
from atomic_model_workspace.common import digest
from scripts.buoyant_graph_references import IndependentBuoyant, loop


@pytest.mark.parametrize(
    "nodes,mirror,branch",
    [(6, False, False), (6, True, False), (8, False, False), (6, False, True)],
)
def test_dimensional_balances_and_independent_stationary_branch(nodes, mirror, branch):
    request = loop(nodes=nodes, mirror=mirror, branch=branch)
    before = copy.deepcopy(request)
    guess = -0.13 if mirror else 0.13
    saved = initialize(request, flow_guess_kg_s=guess)
    assert request == before
    report = saved["payload"]
    reference = IndependentBuoyant(request).stationary(guess)
    for field, atol in [("flows_kg_s", 1e-7), ("temperatures_k", 1e-6), ("pressures_pa", 1e-4)]:
        for key, value in report["state"][field].items():
            assert value == pytest.approx(reference[field][key], abs=atol)
    assert report["jacobian_rank"] == report["unknown_count"]
    assert report["hidden_warmup_steps"] == 0
    assert report["initialization_time_s"] == 0
    assert report["validated"] is False
    assert audit_certificate(json.loads(json.dumps(saved))).snapshot()["state"]["time_s"] == 0


@pytest.mark.parametrize(
    "guess,temp",
    [(float("nan"), 310), (float("inf"), 310), (0.13, float("nan")), (1e10, 310), (0.13, 10000)],
)
def test_invalid_guess_without_projection(guess, temp):
    with pytest.raises(ValueError):
        initialize(loop(), flow_guess_kg_s=guess, temperature_guess_k=temp)


def test_missing_thermal_constraint_is_rejected():
    with pytest.raises(ValueError, match="not locally unique"):
        initialize(loop(heated=False), flow_guess_kg_s=0, temperature_guess_k=300)


@pytest.mark.parametrize(
    "change",
    [
        "heat",
        "topology",
        "temperature",
        "state",
        "residual",
        "maxima",
        "gauge",
        "rank",
        "warmup",
        "status",
        "source",
        "criteria",
    ],
)
def test_rehashed_certificate_tampering(change):
    saved = initialize(loop())
    corrupt = copy.deepcopy(saved)
    p = corrupt["payload"]
    if change == "heat":
        p["initialized_request"]["base"]["thermal"]["nodes"]["n0"]["heat_power_w"] *= 1.01
    elif change == "topology":
        p["initialized_request"]["base"]["hydraulic"]["edges"][0]["linear_pa_s_kg"] *= 1.01
    elif change == "temperature":
        p["initialized_request"]["base"]["thermal"]["nodes"]["n0"]["initial_temperature_k"] += 1
    elif change == "state":
        p["state"]["flows_kg_s"]["e0"] += 0.01
    elif change == "residual":
        p["residuals"][0]["value"] = 1
    elif change == "maxima":
        p["max_thermal_residual_w"] = 1000
    elif change == "gauge":
        p["pressure_gauge"]["n0"] += 1
    elif change == "rank":
        p["jacobian_rank"] -= 1
    elif change == "warmup":
        p["hidden_warmup_steps"] = 1
    elif change == "status":
        p["validated"] = True
    elif change == "criteria":
        p["criteria"]["momentum_pa"] = 1000
    else:
        p["source"]["python"] = "wrong"
    corrupt["sha256"] = digest(p)
    with pytest.raises(ValueError):
        audit_certificate(corrupt)


def test_pure_residual_and_negative_probe_does_not_mutate_equations():
    eq = Equations(loop())
    before = copy.deepcopy(eq.request)
    x = (np.r_[np.full(eq.m, 0.13), eq.pbase[eq.free], np.full(eq.n, 310.0)] - eq.origin) / eq.scale
    expected = eq.raw(x)
    bad = x.copy()
    bad[-1] = 1e10
    with pytest.raises(ValueError):
        eq.raw(bad)
    np.testing.assert_array_equal(eq.raw(x), expected)
    assert eq.request == before


def test_cli_and_strict_no_overwrite(tmp_path):
    inp = tmp_path / "request.json"
    out = tmp_path / "result.json"
    inp.write_text(json.dumps(loop()))
    assert main(["initialize", str(inp), "--output", str(out)]) == 0
    assert main(["audit", str(out)]) == 0
    with pytest.raises(SystemExit):
        main(["initialize", str(inp), "--output", str(out)])


def test_qualification_handoff_restarts_and_rollback():
    result = qualify()
    assert len(result["cases"]) == 4
    for case in result["cases"]:
        assert case["rollback_equal"]
        assert all(case["restart_equal"])
        assert len(case["hold_and_restart"]) == 2
