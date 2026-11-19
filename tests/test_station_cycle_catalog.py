import pytest
from atomic_model_station_hybrid_heat import catalog
from atomic_model_station_hybrid_heat.catalog import CycleCase,run_cycle_case,run_cycle_catalog


@pytest.mark.parametrize('name',['','../escape','a/b','has space','x'*65])
def test_bad_identifiers_are_rejected(name):
    with pytest.raises(ValueError):CycleCase(name)


@pytest.mark.parametrize('kwargs',[dict(area_m2=True),dict(area_m2=float('nan')),dict(stroke_time_s=.01)])
def test_case_parameters_use_backend_domain(kwargs):
    with pytest.raises(ValueError):CycleCase('case',**kwargs)


def test_duplicate_names_and_physical_variants_are_rejected_before_execution(monkeypatch):
    monkeypatch.setattr(catalog,'run_cycle_case',lambda *a,**k:pytest.fail('must not execute'))
    with pytest.raises(ValueError,match='identifier'):run_cycle_catalog([CycleCase('same'),CycleCase('same',area_m2=.029)])
    with pytest.raises(ValueError,match='physical'):run_cycle_catalog([CycleCase('a'),CycleCase('b')])


def test_failed_attempt_remains_in_denominator_and_later_case_executes(monkeypatch):
    seen=[]
    def run(case,**kwargs):
        seen.append(case.name)
        return dict(case=case.name,passed=case.name=='second',failure=None if case.name=='second' else dict(type='numerical_refusal'))
    monkeypatch.setattr(catalog,'run_cycle_case',run)
    r=run_cycle_catalog([CycleCase('first'),CycleCase('second',area_m2=.029)])
    assert seen==['first','second'] and len(r['results'])==2
    assert r['total_cases']==2 and r['passed_cases']==1 and r['failed_cases']==1
    assert r['success_fraction']==.5 and not r['passed'] and r['all_attempts_retained']


def test_initialization_failure_is_reported_without_fake_success(monkeypatch):
    def fail():raise RuntimeError('cannot initialize')
    monkeypatch.setattr(catalog,'prepare_relief_cycle',fail)
    result=run_cycle_case(CycleCase('case'))
    assert not result['passed'] and result['proof'] is None
    assert result['failure']['type']=='RuntimeError'


def test_source_change_invalidates_catalog(monkeypatch):
    versions=iter([1,2]);monkeypatch.setattr(catalog,'identity',lambda:next(versions))
    monkeypatch.setattr(catalog,'run_cycle_case',lambda *a,**k:dict(passed=True))
    with pytest.raises(ValueError,match='source changed'):run_cycle_catalog([CycleCase('case')])
