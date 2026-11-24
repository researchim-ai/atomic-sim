from types import SimpleNamespace
import pytest

from scripts import qualify_boron_regenerator as qualification


@pytest.mark.parametrize('cases',[{},dict(dilution={},boration={}),dict(dilution={},unknown={})])
def test_incomplete_or_unknown_case_matrix_is_refused(cases):
    with pytest.raises(ValueError):qualification.assess(cases)


def test_failed_independent_reference_cannot_be_used_as_evidence(monkeypatch):
    monkeypatch.setattr(qualification,'solve_bvp',lambda *args,**kwargs:SimpleNamespace(success=False))
    data=dict(hot_inlet_temperature_k=560.,cold_inlet_temperature_k=350.,ua_w_k=200000.,
        hot_cp_j_kg_k=5200.,cold_cp_j_kg_k=4180.)
    with pytest.raises(ValueError,match='did not converge'):qualification.capacity_reference(data,10.)


@pytest.mark.parametrize('budget',[True,1,10001])
def test_reference_node_budget_is_explicit_and_bounded(budget):
    with pytest.raises(ValueError,match='budget'):qualification.capacity_reference({},10.,max_nodes=budget)
