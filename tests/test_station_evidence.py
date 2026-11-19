from copy import deepcopy
import json
import pytest
from atomic_model_station_evidence import candidate_registry
from atomic_model_station_evidence import model
from atomic_model_station_evidence.__main__ import main


def test_offline_registry_verifies_all_bundled_artifacts_without_promotion():
    r=candidate_registry()
    assert r['total_candidates']==3 and not r['experimental_validation'] and not r['full_r2_accepted']
    assert {c['id'] for c in r['candidates']}=={'beavrs-cycle1-hzp','nist-tn2294-friction','motel-2025-integral'}
    for c in r['candidates']:
        assert c['gaps'] and c['source']['version'] and c['observables']
        assert not c['admitted_for_station_validation'] and not c['experimental_validation']
        assert not c['uncertainty']['complete_joint_budget']
        assert all(a['integrity_verified'] for a in c['artifacts'])


@pytest.mark.parametrize('change',['duplicate','missing_units','missing_gaps','forged_admission','missing_uncertainty','license_conflation'])
def test_incomplete_or_forged_metadata_is_refused(monkeypatch,change):
    p=deepcopy(model._manifest());c=p['candidates'][0]
    if change=='duplicate':p['candidates'].append(deepcopy(c))
    if change=='missing_units':del c['observables'][0]['unit']
    if change=='missing_gaps':c['gaps']=[]
    if change=='forged_admission':c['admitted_for_station_validation']=True
    if change=='missing_uncertainty':del c['uncertainty']['statement']
    if change=='license_conflation':c['rights']={'license':'open'}
    monkeypatch.setattr(model,'_manifest',lambda:p)
    with pytest.raises(ValueError):candidate_registry()


@pytest.mark.parametrize('path',['../secret','/tmp/secret','data//source.pdf','data/./source.pdf','C:secret'])
def test_nonportable_artifact_paths_are_rejected(monkeypatch,path):
    p=deepcopy(model._manifest());p['candidates'][0]['artifacts'][0]['path']=path
    monkeypatch.setattr(model,'_manifest',lambda:p)
    with pytest.raises(ValueError,match='outside allowed'):candidate_registry()


def test_changed_artifact_checksum_is_refused(monkeypatch):
    p=deepcopy(model._manifest());p['candidates'][0]['artifacts'][0]['sha256']='0'*64
    monkeypatch.setattr(model,'_manifest',lambda:p)
    with pytest.raises(ValueError,match='artifact changed'):candidate_registry()


def test_returned_record_mutation_does_not_change_next_read():
    first=candidate_registry();first['candidates'][0]['rights']['raw_measurements_bundled']=True
    assert not candidate_registry()['candidates'][0]['rights']['raw_measurements_bundled']


def test_cli_new_output_and_existing_result_preservation(tmp_path,monkeypatch,capsys):
    path=tmp_path/'registry.json';monkeypatch.setattr('sys.argv',['registry','--output',str(path)])
    main();stdout=json.loads(capsys.readouterr().out)
    assert json.loads(path.read_text())==stdout
    before=path.read_bytes()
    with pytest.raises(SystemExit):main()
    assert path.read_bytes()==before
