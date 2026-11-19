from copy import deepcopy
from dataclasses import replace

import pytest

from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_station_hot import HotConfig, HotSession
from atomic_model_station_hot import model
from atomic_model_station_readiness import ReadinessConfig


def short():
    # Permissive short fixture exercises the lifecycle, never station acceptance.
    return HotConfig(maximum_warmup_s=.04,hold_s=.03,tick_s=.01,power_target_w=3e9,
        readiness=ReadinessConfig(window_s=.02,maximum_gap_s=.01,relative_span=1.,stored_power_relative=1.))


@pytest.mark.parametrize('kw',[dict(tick_s=True),dict(hold_s=0.),dict(maximum_warmup_s=.01),
    dict(maximum_warmup_s=4000.,hold_s=3600.),dict(inventory_deadband_m=.002),dict(max_step_s=2.),
    dict(readiness=ReadinessConfig(window_s=.025,maximum_gap_s=.01)),dict(rtol=float('nan'))])
def test_invalid_hot_config(kw):
    with pytest.raises(ValueError):replace(short(),**kw)


def test_short_preparation_and_hold_share_clock_pi_and_stocks():
    session=HotSession(short())
    while not session.finished:session.step()
    r=session.report()
    assert r['hold_passed'] and r['warmup_end_s']==pytest.approx(.03)
    assert r['time_s']==pytest.approx(.06) and r['hold_elapsed_s']==pytest.approx(.03)
    assert len(session.owner.history)==r['accepted_steps']==6
    assert not r['canonical_hold_passed'] and not r['stationary_accepted']
    assert not any(r[k] for k in ('time_reset','inventories_projected','pi_reset'))
    assert abs(session.inventory_summary()['total_mass_change_kg'])<1e-8


def test_unprepared_state_never_starts_hold_and_is_replayable():
    cfg=replace(short(),readiness=ReadinessConfig(window_s=.02,maximum_gap_s=.01,relative_span=1e-12))
    s=HotSession(cfg)
    while not s.finished:s.step()
    assert s.phase=='unprepared' and s.failure=='warmup_exhausted'
    assert s.report()['hold_elapsed_s']==0. and s.warmup_end_s is None
    assert HotSession.restore(s.checkpoint()).checkpoint()==s.checkpoint()
    before=s.checkpoint()
    with pytest.raises(ValueError):s.step()
    assert s.checkpoint()==before


def test_restart_during_hold_reproduces_all_records_and_final_checkpoint():
    s=HotSession(short())
    for _ in range(4):s.step()
    assert s.phase=='hold'
    restored=HotSession.restore(s.checkpoint())
    while not s.finished:assert s.step()==restored.step()
    assert s.checkpoint()==restored.checkpoint()


@pytest.mark.parametrize('part',['records','owner','source'])
def test_tampered_hot_checkpoint_is_not_a_continuation(part):
    s=HotSession(short());s.step();p=deepcopy(unseal(s.checkpoint()))
    if part=='records':p['records'][-1]['sample']['state'][0]+=1.
    elif part=='owner':p['owner']['payload']['initial']['inventory_deadband_m']=0.
    else:p['source']={}
    with pytest.raises(ValueError):HotSession.restore(seal(p))


def test_exported_checkpoint_does_not_alias_owned_history_or_readiness():
    s=HotSession(short());s.step();before=s.checkpoint();exported=s.checkpoint()
    exported['payload']['owner']['payload']['history'][0]['targets']['rod_target']=1.
    exported['payload']['records'][0]['sample']['state'][0]=12.
    report=s.report();report['readiness']['reason']='corrupt'
    assert s.checkpoint()==before


def test_rejected_tick_keeps_the_accepted_owner_prefix(monkeypatch):
    s=HotSession(short());before=s.owner.checkpoint()
    def reject(*args,**kwargs):raise ValueError('synthetic physical rejection')
    monkeypatch.setattr(model.FaultSession,'step',reject)
    result=s.step()
    assert result['phase']=='failed' and result['accepted_steps']==0 and result['attempts']==1
    assert s.owner.checkpoint()==before and len(s.records)==1
    assert HotSession.restore(s.checkpoint()).checkpoint()==s.checkpoint()


def test_lost_readiness_stops_hold_without_rolling_time_back(monkeypatch):
    s=HotSession(short())
    for _ in range(3):s.step()
    assert s.phase=='hold'
    assess=model.assess
    def drift(*args,**kwargs):
        result=assess(*args,**kwargs);result.update(ready=False,reason='synthetic_drift');return result
    monkeypatch.setattr(model,'assess',drift)
    result=s.step()
    assert s.phase=='failed' and s.failure=='hold_readiness_lost'
    assert result['time_s']==pytest.approx(.04) and result['accepted_steps']==4


def test_configuration_manifest_roundtrip_and_attempt_domain():
    from dataclasses import asdict
    assert HotConfig.from_payload(asdict(short()))==short()
    s=HotSession(short());p=deepcopy(unseal(s.checkpoint()));p['attempts']=True
    with pytest.raises(ValueError):HotSession.restore(seal(p))


def test_default_configuration_is_canonical_but_has_not_accepted_a_hold():
    result=HotSession().report()
    assert result['canonical_criteria'] and not result['hold_passed'] and not result['canonical_hold_passed']
    assert not result['stationary_accepted'] and result['time_s']==0.
