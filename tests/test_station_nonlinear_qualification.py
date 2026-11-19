import pytest
from scripts import qualify_station_nonlinear as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    p=tmp_path_factory.mktemp('nonlinear-dynamics')/'source';study.run(p);return p


def test_dynamic_nonlinear_pressure_energy_and_scope(source):
    r=read(source/'report.json');ref=read(source/'refinement.json')
    assert r['passed'] and r['retained_nonlinear_midpoint_qualified'] and r['trajectory_cases']==5
    assert r['all_native_restarts_exact'] and r['iteration_exhaustion_atomic']
    assert r['minimum_time_order']>1.99 and r['adaptive_reference_agreement']<1e-10
    assert not r['spatial_mode_convergence_qualified'] and not r['strong_wall_momentum_qualified'] and not r['station_usable']
    assert r['closed_tasks']==[] and ref['stationary_swirl_velocity_exact']
    for row in read(source/'cases.json').values():assert all(row['checks'].values())


def test_all_four_scientific_json_repeat_exactly(source,tmp_path):
    assert study.run(tmp_path/'repeat',source)['passed']
    assert {p.name for p in source.glob('*.json')}=={'protocol.json','cases.json','refinement.json','report.json'}


def test_reported_unconverged_iteration_cannot_be_accepted(tmp_path,monkeypatch):
    original=study.NonlinearSession.advance
    def wrong(self,*args,**kwargs):
        row,fields=original(self,*args,**kwargs);row['nonlinear_relative_residual']=1.;return row,fields
    monkeypatch.setattr(study.NonlinearSession,'advance',wrong)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'rejected')
    assert not read(tmp_path/'rejected/report.json')['passed']
