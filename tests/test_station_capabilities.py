from copy import deepcopy
from dataclasses import asdict
import json
import pytest
from atomic_model_station_env import EnvConfig,StationEnv
from atomic_model_station_faults import Fault
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_workspace.common import digest
from atomic_model_station_capabilities import passport,dump_config,load_config,migrate_config,CONFIG_FORMAT,RAW_FORMAT
from atomic_model_station_capabilities import model
from atomic_model_station_capabilities.__main__ import write_new,read_document


def test_passport_has_current_recipe_evidence_without_general_validation_claims(monkeypatch):
    monkeypatch.setattr(StationEnv,'reset',lambda *a,**k:pytest.fail('passport must not simulate'))
    p=passport()
    assert {r['kind'] for r in p['qualifications']}=={'nominal','tutorials'}
    assert all(r['current_source_matches'] for r in p['qualifications'])
    assert not p['industrial_plant_qualified'] and not p['experimental_validation'] and not p['full_r1_accepted']
    assert not p['water_chart']['whole_plant_qualified_range']
    assert p['observations']['checkpoint_privileged'] and not p['observations']['latent_state_in_agent_transport']


def test_stale_nominal_receipt_is_not_reported_current(monkeypatch):
    from scripts import qualify_station_equilibrium as study
    monkeypatch.setattr(study,'identity',lambda:{'changed_source':True})
    p=passport();rows={r['kind']:r for r in p['qualifications']}
    assert not rows['nominal']['current_source_matches'] and rows['tutorials']['current_source_matches']


def test_source_change_during_passport_read_is_refused(monkeypatch):
    versions=iter([1,2]);monkeypatch.setattr(model,'model_identity',lambda:next(versions))
    with pytest.raises(ValueError,match='source changed'):passport()


def test_corrupted_index_seal_is_refused(monkeypatch):
    class Resource:
        def joinpath(self,*a):return self
        def read_text(self):return json.dumps(dict(payload=dict(format='forged'),sha256='0'*64))
    monkeypatch.setattr(model,'files',lambda *a:Resource())
    with pytest.raises(ValueError,match='envelope'):passport()


def test_unknown_or_duplicate_receipt_is_refused(monkeypatch):
    records=deepcopy(model._records());records[1]=deepcopy(records[0])
    class Resource:
        def joinpath(self,*a):return self
        def read_text(self):return json.dumps(seal(dict(format='station-evidence-index/v1',records=records)))
    monkeypatch.setattr(model,'files',lambda *a:Resource())
    with pytest.raises(ValueError,match='duplicate'):passport()


def test_configuration_roundtrip_preserves_all_fields_and_faults():
    config=EnvConfig(max_steps=3,tick_s=.02,automatic=False,
        faults=(Fault('drive','pump_off','feed',1),))
    document=json.loads(json.dumps(dump_config(config)))
    assert load_config(document)==config
    migrated,audit=migrate_config(asdict(config),from_format=RAW_FORMAT)
    assert load_config(migrated)==config and audit['fields_dropped']==[]
    assert not audit['values_changed'] and audit['requalification_required']


@pytest.mark.parametrize('change',[{'unknown':1},{'tick_s':float('nan')},{'max_steps':True},
    {'tick_s':20.},{'film_model':'unsupported'},{'faults':[{'unknown':1}]}])
def test_bad_configuration_never_silently_drops_or_changes_values(change):
    values=asdict(EnvConfig());values.update(change)
    with pytest.raises(ValueError):migrate_config(values,from_format=RAW_FORMAT)


def test_missing_field_does_not_acquire_a_new_implicit_default():
    values=asdict(EnvConfig());del values['film_model']
    with pytest.raises(ValueError,match='missing fields'):migrate_config(values,from_format=RAW_FORMAT)


def test_source_change_requires_explicit_rebind():
    config=EnvConfig();old=unseal(dump_config(config));old['model_source']='0'*64;old=seal(old)
    with pytest.raises(ValueError,match='model source changed'):load_config(old)
    with pytest.raises(ValueError,match='Explicit rebind'):migrate_config(old,from_format=CONFIG_FORMAT)
    new,audit=migrate_config(old,from_format=CONFIG_FORMAT,allow_rebind=True)
    assert load_config(new)==config and audit['previous_model_source']=='0'*64
    assert audit['requalification_required'] and not audit['checkpoint_state_migrated']


def test_current_config_migration_is_idempotent_without_requalification():
    document=dump_config(EnvConfig())
    new,audit=migrate_config(document,from_format=CONFIG_FORMAT)
    assert new==document and not audit['requalification_required']


def test_checkpoint_and_unknown_formats_cannot_be_treated_as_configuration():
    checkpoint=seal(dict(format='station-env/v1',config=asdict(EnvConfig()),history=[]))
    with pytest.raises(ValueError,match='checkpoints'):load_config(checkpoint)
    with pytest.raises(ValueError):migrate_config(checkpoint,from_format=RAW_FORMAT)
    with pytest.raises(ValueError,match='Unsupported'):migrate_config({},from_format='invented')


def test_fault_canonicalization_is_reported_and_preserves_fault_specs():
    cfg=asdict(EnvConfig());cfg['faults']=[asdict(Fault('b','pump_off','feed',2)),asdict(Fault('a','pump_off','cooling',1))]
    document,audit=migrate_config(cfg,from_format=RAW_FORMAT)
    actual=asdict(load_config(document))['faults']
    assert audit['fault_order_normalized'] and {digest(f) for f in actual}=={digest(f) for f in cfg['faults']}


def test_existing_files_and_symlink_parents_are_preserved(tmp_path):
    p=tmp_path/'keep.json';p.write_text('keep')
    with pytest.raises(ValueError):write_new(p,{'changed':True})
    assert p.read_text()=='keep'
    (tmp_path/'linked').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):write_new(tmp_path/'linked/new.json',{})


def test_oversized_configuration_is_refused_before_json_decode(tmp_path):
    p=tmp_path/'oversized';p.write_bytes(b'x'*(1024*1024+1))
    with pytest.raises(ValueError,match='1 MiB'):read_document(p)


def test_migrated_configuration_has_exact_native_transitions_and_restart():
    cfg=EnvConfig(max_steps=2,tick_s=.02,automatic=False,rtol=1e-10)
    document,_=migrate_config(asdict(cfg),from_format=RAW_FORMAT)
    original=StationEnv(cfg);migrated=StationEnv(load_config(document))
    assert original.reset()==migrated.reset()
    assert original.step({'operator':{'load_fraction':.99}})==migrated.step({'operator':{'load_fraction':.99}})
    restored=StationEnv.restore(migrated.checkpoint())
    assert original.step({})==migrated.step({})==restored.step({})
    assert original.checkpoint()==migrated.checkpoint()==restored.checkpoint()
