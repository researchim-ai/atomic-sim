from copy import deepcopy
import json
import pytest

from atomic_model_multiphysics.provenance import seal,sha
from atomic_model_training.spec import atomic_json
from atomic_model_station_campaign.engine import signature
from scripts import qualify_station_nominal as nominal
from scripts.qualify_station_hourly_replay import physical_events
from scripts.run_station_hourly_program import generate_reference


def archive(tmp_path,*,foreign=False):
    coarse=tmp_path/'coarse';coarse.mkdir();hashes={}
    for n in nominal.BLOCKS:
        path=coarse/(str(n)+'.json')
        atomic_json(path,seal(dict(source=signature(),trace=[dict(time_s=5.)])))
        hashes['artifacts/station-nominal-exploration-v1/'+path.name]=sha(path)
    baseline=tmp_path/'baseline.json'
    atomic_json(baseline,seal(dict(completed=True,completed_duration_s=3600.,formal_acceptance=False,
        source={'foreign':True} if foreign else signature(),evidence_sha256=hashes)))
    return coarse,baseline


def test_metadata_cannot_promote_partial_coarse_archive_to_full_hour(tmp_path):
    coarse,baseline=archive(tmp_path);out=tmp_path/'out';out.mkdir()
    with pytest.raises(ValueError,match='Complete 3600s'):
        nominal.materialize(out,None,coarse,baseline)


def test_coarse_archive_content_hash_is_checked_before_numerical_work(tmp_path):
    coarse,baseline=archive(tmp_path);(coarse/'120.json').write_text('{}')
    out=tmp_path/'out';out.mkdir()
    with pytest.raises(ValueError,match='hash differs'):nominal.materialize(out,None,coarse,baseline)


def test_foreign_model_archive_is_not_used_as_current_baseline(tmp_path):
    coarse,baseline=archive(tmp_path,foreign=True);out=tmp_path/'out';out.mkdir()
    with pytest.raises(ValueError,match='unchanged'):nominal.materialize(out,None,coarse,baseline)


def test_fresh_generator_preserves_failure_instead_of_fabricating_reference(tmp_path,monkeypatch):
    from atomic_model_station_faults import FaultSession
    def reject(*args,**kwargs):raise ValueError('injected coarse-domain refusal')
    monkeypatch.setattr(FaultSession,'step',reject)
    out=tmp_path/'generated'
    with pytest.raises(ValueError,match='did not complete'):generate_reference(out)
    data=nominal.read(out/'failure.json')
    assert not data['completed'] and data['completed_time_s']==0.
    assert data['accepted_partial_rows']==[] and not (out/'baseline.json').exists()


def test_event_classification_keeps_physical_switches_and_excludes_actual_kinematic_labels():
    physical=[dict(kind=k,time_s=1.) for k in ['steam_valve_arrived','rod_arrived','bypass_position_arrival','feed_check_closed','bypass_supply_close']]
    state=dict(plant=dict(payload=dict(snapshot=dict(events=physical))),events=[dict(kind='protection_latched',time_s=1.)])
    checkpoint=dict(payload=dict(control=dict(payload=dict(state=state))))
    assert {r['kind'] for r in physical_events(checkpoint)}=={'feed_check_closed','bypass_supply_close','protection_latched'}
