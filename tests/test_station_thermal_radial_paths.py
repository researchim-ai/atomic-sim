"""Independent shell axes, checkpoint preservation and fail-closed path evidence."""
import json
import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from atomic_model_station_thermal_cases import thermal_catalog,build_thermal_case
from atomic_model_station_thermal_path import ThermalTrajectory
from atomic_model_station_thermal_resolved import resolved_velocity_mode,compare_thermal
from scripts import qualify_station_thermal_radial_paths as driver
from scripts.qualify_station_inventory import read


@pytest.mark.parametrize('name,velocity_order,temperature_order',[
    ('shell_velocity26',26,24),('shell_temperature32',24,32),
])
def test_shell_axis_changes_only_declared_modes_and_preserves_physical_seed(name,velocity_order,temperature_order):
    case=thermal_catalog()[1]
    base=build_thermal_case(case)
    refined=driver.build_variant(case,name)
    assert refined.keys==base.keys
    assert refined.viscosity==base.viscosity and refined.diffusivity==base.diffusivity
    assert refined.buoyancy==base.buoyancy and refined.sources==base.sources
    for v,t in zip(refined.models,refined.scalars):
        shell=max(abs(v.k),v.m)==4
        assert v.order==(velocity_order if shell else 32)
        assert t.order==(temperature_order if shell else 40)
    a=ThermalTrajectory(base).endpoint()
    b=ThermalTrajectory(refined).endpoint()
    x,w=leggauss(160)
    assert driver.below(compare_thermal(a,b,np.sqrt((x+1)/2),w/4),1e-10)


@pytest.mark.parametrize('name',['shell_velocity26','shell_temperature32'])
def test_shell_spaces_survive_json_checkpoint_roundtrip(name):
    path=ThermalTrajectory(driver.build_variant(thermal_catalog()[0],name))
    envelope=json.loads(json.dumps(path.checkpoint()))
    restored=ThermalTrajectory.restore(envelope)
    assert restored.checkpoint()==envelope
    for (m,q),(n,p) in zip(path.endpoint().velocity,restored.endpoint().velocity):
        assert m.fingerprint==n.fingerprint
        np.testing.assert_array_equal(q,p)
    for (m,q),(n,p) in zip(path.endpoint().temperature,restored.endpoint().temperature):
        assert m.fingerprint==n.fingerprint
        np.testing.assert_array_equal(q,p)


def test_unqualified_velocity_order_is_still_refused():
    with pytest.raises(ValueError,match='constraint/coercivity'):
        resolved_velocity_mode(0,4,.628319,order=28,quadrature=128)


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError,match='Declared radial'):
        driver.build_variant(thermal_catalog()[0],'invented')


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,1e-7])
def test_invalid_or_threshold_norm_does_not_pass(value):
    assert not driver.below(dict(velocity_l2=value,temperature_l2=0.),1e-7)


def test_factory_refusals_do_not_omit_matrix_variants(monkeypatch):
    def refusal(*args,**kwargs):
        raise ValueError('test refusal')
    monkeypatch.setattr(driver,'build_variant',refusal)
    row=driver.qualify_case(thermal_catalog()[0])
    assert not row['accepted'] and not any(row['checks'].values())
    assert set(row['stages'])==set(driver.VARIANTS)
    assert all(s['reason']=='test refusal' for s in row['stages'].values())
    assert row['comparisons']=={}


def test_failed_case_remains_in_final_receipt(tmp_path,monkeypatch):
    def row(case):
        return dict(case=case.specification(),accepted=case.name!='negative_b',
                    stages={'main':dict(completed=False,records=[])})
    monkeypatch.setattr(driver,'qualify_case',row)
    report=driver.run(tmp_path/'study')
    assert not report['passed'] and report['refused_cases']==['negative_b']
    assert report['completed_paths']==report['completed_epochs']==0
    assert len(report['case_evidence'])==5
    assert read(tmp_path/'study/report.json')==report
    assert not report['continuous_radial_error_bound']
    assert not report['station_usable'] and report['closed_tasks']==[]


def test_source_change_prevents_final_receipt(tmp_path,monkeypatch):
    versions=iter([{'v':1},{'v':2}])
    monkeypatch.setattr(driver,'identity',lambda:next(versions))
    monkeypatch.setattr(driver,'qualify_case',lambda c:dict(case=c.specification(),accepted=True))
    with pytest.raises(ValueError,match='source changed'):
        driver.run(tmp_path/'study')
    assert not (tmp_path/'study/report.json').exists()


def test_output_directory_must_be_fresh(tmp_path):
    with pytest.raises(ValueError,match='Fresh ordinary'):
        driver.run(tmp_path)
