import pytest
from scripts import qualify_station_regular as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    p=tmp_path_factory.mktemp('regular')/'source';study.run(p);return p


def test_linear_regular_stokes_scope_and_all_dynamic_gates(source):
    r=read(source/'report.json')
    assert r['passed'] and r['linear_single_mode_stokes_qualified'] and r['cases']==12
    assert r['all_native_restarts_exact'] and r['minimum_time_order']>1.99 and r['finest_bessel_error']<1e-10
    for case in read(source/'dynamics.json').values():assert all(case['checks'].values())
    assert not r['physical_pressure_reconstructed'] and not r['nonlinear_advection_supported']
    assert not r['station_usable'] and not r['native_openpipeflow_backend_changed'] and r['closed_tasks']==[]


def test_all_four_scientific_files_repeat_exactly(source,tmp_path):
    assert study.run(tmp_path/'repeat',source)['passed']
    assert {p.name for p in source.glob('*.json')}=={'protocol.json','dynamics.json','refinement.json','report.json'}


def test_a_reported_boundary_failure_cannot_receive_acceptance(tmp_path,monkeypatch):
    original=study.StokesSession.advance
    def wrong(self,*args,**kwargs):
        result=original(self,*args,**kwargs);result['wall_maximum']=1.;return result
    monkeypatch.setattr(study.StokesSession,'advance',wrong)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'rejected')
    assert not read(tmp_path/'rejected/report.json')['passed']
