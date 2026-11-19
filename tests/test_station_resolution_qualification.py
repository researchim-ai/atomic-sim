import pytest
from scripts import qualify_station_resolution as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    p=tmp_path_factory.mktemp('resolution')/'source';study.run(p);return p


def test_execution_and_conditional_physical_gates_are_not_conflated(source):
    r=read(source/'report.json');c=read(source/'comparison.json')
    assert r['passed'] and r['cases']==9 and r['maximum_initial_velocity_error']<1e-10
    assert r['fourier_conditional_gate_passed']
    assert not r['basis_gate_passed'] and not r['temporal_gate_passed'] and not r['finest_wall_gate_passed']
    assert not r['full_spatial_convergence_qualified'] and not r['station_usable'] and r['closed_tasks']==[]
    assert c['fourier'][0]['total_rms']>100*c['fourier'][0]['common_rms']
    assert all(v['omitted_modes_included'] for v in c['fourier'])


def test_four_scientific_files_repeat_exactly(source,tmp_path):
    assert study.run(tmp_path/'repeat',source)['passed']
    assert {p.name for p in source.glob('*.json')}=={'protocol.json','cases.json','comparison.json','report.json'}


def test_inconsistent_initial_state_cannot_receive_execution_acceptance(tmp_path,monkeypatch):
    original=study.compare_states
    def wrong(*args,**kwargs):
        r=original(*args,**kwargs)
        if kwargs['first_time']==0:r['total_rms']=1.
        return r
    monkeypatch.setattr(study,'compare_states',wrong)
    with pytest.raises(ValueError,match='initial-state qualification failed'):study.run(tmp_path/'rejected')
    assert not read(tmp_path/'rejected/report.json')['passed']
