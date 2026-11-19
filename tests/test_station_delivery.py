from copy import deepcopy
import json
import pytest
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_training.spec import read_json
from scripts import verify_station_install as driver

@pytest.fixture
def nominal():
    role=dict(duration_s=3600.,window_count=601,completed=True,passed=True,source_unchanged=True,
        all_600s_windows_ready=True,physical_audits_passed=True)
    return dict(format='station-nominal-requalification/v105',report=dict(
        source=driver.nominal.identity(),driver_sha256=sha(driver.nominal_delivery.__file__),
        passed=True,failure=None,mathematical_files_compared=18,
        checks={k:True for k in ('source_inputs_unchanged','wheel_unchanged','all_mathematical_json_exact','native_replay','installed_fine','source_unchanged')},
        source_report=dict(passed=True,source_unchanged=True,duration_s=3600.,coarse=deepcopy(role),fine=deepcopy(role))),
        installed_native=dict(duration_s=3600.,completed=True,prefix_exact=True,full_suffix_and_final_exact=True,source_unchanged=True))


def test_full_current_hour_can_be_reused_but_not_reported_as_new(nominal):
    result=driver.accept_nominal(seal(nominal))
    assert result['reused_existing_execution'] and result['new_hour_executed'] is False
    assert result['duration_s']==3600 and result['mathematical_files_exact']==18

@pytest.mark.parametrize('change',[
 lambda p:p.update(format='other'),
 lambda p:p['report'].update(source={}),
 lambda p:p['report'].update(driver_sha256='0'*64),
 lambda p:p['report'].update(passed=False),
 lambda p:p['report'].update(failure={'message':'partial'}),
 lambda p:p['report']['checks'].pop('native_replay'),
 lambda p:p['report']['checks'].update(installed_fine=False),
 lambda p:p['report'].update(mathematical_files_compared=17),
 lambda p:p['report']['source_report'].update(duration_s=3595.),
 lambda p:p['report']['source_report']['coarse'].update(window_count=600),
 lambda p:p['report']['source_report']['fine'].update(completed=False),
 lambda p:p['installed_native'].update(full_suffix_and_final_exact=False),
 lambda p:p['installed_native'].update(duration_s=1800.),
 lambda p:p.pop('installed_native'),
])
def test_stale_or_partial_nominal_cannot_close_delivery(nominal,change):
    change(nominal)
    with pytest.raises(ValueError):driver.accept_nominal(seal(nominal))


def test_tampered_receipt_refused(nominal):
    envelope=seal(nominal);envelope['payload']['report']['passed']=False
    with pytest.raises(ValueError):driver.accept_nominal(envelope)


def test_installed_requirement_refuses_checkout_before_simulation(tmp_path):
    with pytest.raises(ValueError,match='installed wheel'):
        driver.run(tmp_path/'result.json',require_installed=True)
    assert not (tmp_path/'result.json').exists()


def test_native_probe_retains_expected_refusal_and_midpoint_replay(tmp_path,monkeypatch):
    monkeypatch.setattr(driver.precision,'CASES',{k:v for k,v in driver.precision.CASES.items() if k in ('hold','invalid-command')})
    result=driver.run(tmp_path/'result.json')
    assert result['passed'] and all(result['checks'].values())
    assert result['cases']['invalid-command']['failure']['code']=='invalid_action'
    assert result['midpoint_replay']['transitions_exact'] and result['midpoint_replay']['checkpoint_exact']
    assert result['nominal'] is None and not result['full_r1_accepted']
    persisted=unseal(read_json(tmp_path/'result.json'))
    assert persisted['scientific']['passed'] and persisted['execution']['installed_imports_required'] is False
    with pytest.raises(ValueError):driver.run(tmp_path/'result.json')


def test_unexpected_case_failure_prevents_acceptance_but_is_retained(tmp_path,monkeypatch):
    monkeypatch.setattr(driver.precision,'CASES',{'hold':[{}]})
    monkeypatch.setattr(driver.precision,'attempt',lambda *a:dict(scientific=dict(
        restore_exact=False,completed=False,failure=dict(code='unexpected',phase='advance'))))
    monkeypatch.setattr(driver,'midpoint_replay',lambda:dict(transitions_exact=True,checkpoint_exact=True))
    monkeypatch.setattr(driver,'demo',lambda:{})
    monkeypatch.setattr(driver.viewer,'load',lambda value:value)
    with pytest.raises(ValueError,match='results retained'):driver.run(tmp_path/'failed.json')
    result=unseal(read_json(tmp_path/'failed.json'))['scientific'];assert not result['passed']
    assert result['cases']['hold']['failure']['code']=='unexpected'


def test_midpoint_exception_is_retained_and_cannot_pass(tmp_path,monkeypatch):
    monkeypatch.setattr(driver.precision,'CASES',{})
    def broken():raise RuntimeError('restart failed')
    monkeypatch.setattr(driver,'midpoint_replay',broken)
    monkeypatch.setattr(driver,'demo',lambda:{})
    monkeypatch.setattr(driver.viewer,'load',lambda value:value)
    with pytest.raises(ValueError):driver.run(tmp_path/'failed.json')
    result=unseal(read_json(tmp_path/'failed.json'))['scientific']
    assert not result['passed'] and result['failures'][0]['phase']=='midpoint-replay'


def test_source_freeze_covers_packaging_and_resources_but_ignores_caches(tmp_path):
    from scripts.verify_local_library import source_manifest
    (tmp_path/'src/model').mkdir(parents=True);(tmp_path/'src/model/a.py').write_text('x=1\n')
    (tmp_path/'data').mkdir();(tmp_path/'data/case.json').write_text('{}')
    (tmp_path/'pyproject.toml').write_text('[build-system]\n')
    initial=source_manifest(tmp_path)
    (tmp_path/'src/model/__pycache__').mkdir();(tmp_path/'src/model/__pycache__/a.pyc').write_bytes(b'cache')
    (tmp_path/'src/test.egg-info').mkdir();(tmp_path/'src/test.egg-info/PKG-INFO').write_text('generated')
    assert source_manifest(tmp_path)==initial
    (tmp_path/'data/case.json').write_text('{"changed":true}')
    assert source_manifest(tmp_path)!=initial
    (tmp_path/'data/case.json').write_text('{}')
    (tmp_path/'pyproject.toml').write_text('[project]\n')
    assert source_manifest(tmp_path)!=initial
