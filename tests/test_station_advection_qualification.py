import pytest
from scripts import qualify_station_advection as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    p=tmp_path_factory.mktemp('advection')/'source';study.run(p);return p


def test_kernels_energy_alias_refusal_and_nonlinear_scope(source):
    r=read(source/'report.json');physics=read(source/'physics.json')
    assert r['passed'] and r['nonlinear_spatial_kernel_qualified'] and r['cases']==6
    assert r['maximum_fft_coefficient_error']<1e-11 and r['maximum_energy_power']<1e-11
    assert r['alias_default_refusal'] and r['alias_coefficient_error']>1e-6 and r['alias_energy_power']>1e-9
    assert not r['nonlinear_time_integrator_qualified'] and not r['station_usable'] and r['closed_tasks']==[]
    for row in read(source/'cases.json').values():
        assert all(row['checks'].values()) and row['maximum_modal_transfer']>1e-12
    assert physics['centrifugal_swirl']['pressure_error']<1e-10


def test_all_four_scientific_json_repeat_exactly(source,tmp_path):
    assert study.run(tmp_path/'repeat',source)['passed']
    assert {p.name for p in source.glob('*.json')}=={'protocol.json','cases.json','physics.json','report.json'}


def test_false_fft_metadata_cannot_hide_corrupted_coefficients(tmp_path,monkeypatch):
    original=study.fft_advection
    def wrong(*args,**kwargs):
        result,proof=original(*args,**kwargs)
        # Keep conjugate symmetry, so the coefficient comparison must catch it.
        result={key:value+1 for key,value in result.items()}
        proof['alias_free_for_requested_targets']=True
        return result,proof
    monkeypatch.setattr(study,'fft_advection',wrong)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'rejected')
    assert not read(tmp_path/'rejected/report.json')['passed']
