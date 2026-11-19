import numpy as np
import pytest
from atomic_model_station_stokes_pressure import PressureField
from scripts import qualify_station_stokes_pressure as study
from scripts.qualify_station_inventory import read


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    p=tmp_path_factory.mktemp('physical-pressure')/'source';study.run(p);return p


def test_physical_weak_pressure_and_coupled_gates(source):
    r=read(source/'report.json')
    assert r['passed'] and r['linear_weak_physical_pressure_qualified'] and r['cases']==12
    assert r['base_v72_dynamics_exact'] and r['all_restarts_exact']
    assert not r['strong_wall_momentum_qualified'] and not r['nonlinear_pressure_qualified']
    assert not r['station_usable'] and r['closed_tasks']==[]
    for row in read(source/'cases.json').values():assert all(row['checks'].values())


def test_scientific_json_repeat_exactly(source,tmp_path):
    assert study.run(tmp_path/'repeat',source)['passed']
    assert {p.name for p in source.glob('*.json')}=={'protocol.json','cases.json','report.json'}


def test_wrong_pressure_field_cannot_pass_on_reported_linear_residual_alone(tmp_path,monkeypatch):
    original=study.recover_pressure
    def wrong(model,reaction):
        field,receipt=original(model,reaction)
        return PressureField(model,field.coefficients+.01),receipt
    monkeypatch.setattr(study,'recover_pressure',wrong)
    with pytest.raises(ValueError,match='qualification failed'):study.run(tmp_path/'rejected')
    assert not read(tmp_path/'rejected/report.json')['passed']
