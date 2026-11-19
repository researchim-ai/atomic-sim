from copy import deepcopy
from pathlib import Path
import pytest
from atomic_model_training.spec import read_json,atomic_json
from atomic_model_multiphysics.provenance import seal,unseal
from scripts.qualify_station_shutdown import identity
from tests import _station_history as history


def sample():return {'model':{'plant':{'files':{'water.py':history.OLD_WATER,'other.py':'unchanged'}}}}


def test_unchanged_sources_need_no_migration():
    assert history.assert_source_compatible({'a':{'b':'same'}},{'a':{'b':'same'}})==[]


def test_only_actual_evidenced_cache_evolution_is_acknowledged():
    old=unseal(read_json(history.ROOT/'docs/benchmarks/station_low_flow_v63.json'))['source']
    original=deepcopy(old);paths=history.assert_source_compatible(old,identity())
    assert len(paths)==1 and paths[0].endswith('/plant/files/water.py')
    assert old==original # no rewriting of historical provenance

@pytest.mark.parametrize('kind',['new_water_version','unrelated_file','added_key'])
def test_other_source_changes_still_fail(kind):
    old=sample();current=deepcopy(old)
    if kind=='new_water_version':current['model']['plant']['files']['water.py']='0'*64
    if kind=='unrelated_file':current['model']['plant']['files']['other.py']='changed'
    if kind=='added_key':current['model']['extra']='new'
    with pytest.raises(AssertionError):history.assert_source_compatible(old,current)


def test_missing_proof_cannot_authorize_cache_exception(tmp_path,monkeypatch):
    old=sample();current=deepcopy(old);current['model']['plant']['files']['water.py']=history.CACHED_WATER
    monkeypatch.setattr(history,'ROOT',tmp_path)
    with pytest.raises(OSError):history.assert_source_compatible(old,current)


def test_resealed_but_failed_cache_proof_is_not_accepted(tmp_path,monkeypatch):
    original=unseal(read_json(history.ROOT/'docs/benchmarks/station_water_cache_v104.json'))
    original['attempts'][0]['checks']['exact_scientific']=False
    target=tmp_path/'docs/benchmarks';target.mkdir(parents=True)
    atomic_json(target/'station_water_cache_v104.json',seal(original))
    old=sample();current=deepcopy(old);current['model']['plant']['files']['water.py']=history.CACHED_WATER
    monkeypatch.setattr(history,'ROOT',tmp_path)
    with pytest.raises(AssertionError):history.assert_source_compatible(old,current)


def test_evidence_freeze_rejects_changes_but_allows_new_receipts(tmp_path):
    from scripts.verify_local_library import evidence_manifest,evidence_unchanged
    folder=tmp_path/'docs/benchmarks';folder.mkdir(parents=True)
    old=folder/'old.json';old.write_text('{}')
    expected=evidence_manifest(tmp_path)
    (folder/'new.json').write_text('{"new":true}')
    assert evidence_unchanged(expected,tmp_path)
    old.write_text('{"changed":true}')
    assert not evidence_unchanged(expected,tmp_path)
    old.unlink();assert not evidence_unchanged(expected,tmp_path)
