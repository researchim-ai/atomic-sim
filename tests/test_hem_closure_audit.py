"""Machine-readable closure provenance, runtime domains and diagnostic spread."""

from copy import deepcopy

import pytest

from atomic_model_hem_qualified.cases import baseline, boiling_case
from scripts.audit_hem_closures import audit, passports


def test_passports_preserve_disconnected_chf_domain_and_unknown_uncertainty():
    p = passports()
    assert len(p["chf_table1995"]["disconnected_boxes"]) == 2
    assert p["chf_table1995"]["diameter_m"] == 0.008
    assert p["chf_biasi_inl"]["mass_flux_kg_m2_s"] == [100, 600]
    assert "post-CHF" in p["excluded"]
    assert "no confidence" in p["boiling"]["uncertainty"]


def test_actual_friction_diagnostic_is_source_bound_and_does_not_mutate_input():
    request = baseline(0.5)
    original = deepcopy(request)
    result = audit(request, [dict(duration_s=2e-5, command=None)])
    assert request == original
    assert result["source_unchanged"] and not result["validated"]
    assert result["sampled_applicability"][-1]["time_s"] == 2e-5
    friction = result["sampled_applicability"][0]["friction"]
    assert friction and all(r["absolute_force_density_spread_n_m3"] > 0 for r in friction)
    assert all(r["values"]["entropy_production_w_m3_k"] >= 0 for r in friction)
    assert "not statistical" in result["uncertainty_status"]


def test_actual_boiling_passport_reports_positive_chf_slack():
    result = audit(boiling_case(1e6, 500.0, "biasi_inl"), [dict(duration_s=2e-6, command=None)])
    boiling = result["sampled_applicability"][0]["boiling"][0]
    assert boiling["chf_fraction_slack"] > 0
    assert boiling["values"]["heat_flux_w_m2"] > 0
    assert not boiling["values"]["validated"]


@pytest.mark.parametrize("change", ["scope", "table-gap", "post-chf"])
def test_unsupported_closures_refused_instead_of_extrapolated(change):
    request = boiling_case(1e6, 500.0, "biasi_inl")
    wall = request["model"]["walls"][0]
    if change == "scope": wall["boiling"]["scope"]["vertical_upflow"] = False
    elif change == "table-gap": wall["boiling"]["chf_model"] = "table1995"
    else: wall["boiling"]["maximum_chf_fraction"] = 1e-8
    with pytest.raises(ValueError): audit(request, [dict(duration_s=2e-6, command=None)])


@pytest.mark.parametrize("calls", [[], [dict(duration_s=1e-5, command={})], [dict(duration_s=1e-5)]])
def test_ambiguous_audit_programs_refused(calls):
    with pytest.raises(ValueError): audit(baseline(), calls)
