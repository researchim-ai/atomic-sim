from dataclasses import replace
import json

import pytest

from atomic_model_station_control import PI, PIConfig
from atomic_model_station_faults import FaultSession


def config(**kw):
    base=PIConfig('feed_level','feed_inventory_fraction',2.,.1,kp=2.5,ki=.02,bias=.5,
                  tracking_tau_s=10.,inventory_deadband_m=.0002)
    return replace(base,**kw)


def test_in_band_neutralizes_old_integral_without_projecting_measurement():
    pi=PI(config());pi.integral=.3;pi.valid_last=True
    p=pi.propose(2.0001,2.,.8,1.)
    assert p['raw']==.5 and p['command']==.5 and p['error']==0.
    pi.commit(p,.5,1.,automatic=True)
    assert pi.integral==0.
    p=pi.propose(2.0003,2.,.5,1.)
    assert p['raw']<.5 and p['error']<0.


def test_neutral_request_respects_rate_limit_and_actual_tracking():
    pi=PI(config(rate_per_s=.01));p=pi.propose(2.,2.,.8,1.)
    assert p['raw']==.5 and p['command']==pytest.approx(.79)
    pi.commit(p,.79,1.,automatic=True)
    assert pi.integral==pytest.approx(.029)
    assert pi.propose(2.,2.,.79,1.)['raw']==.5


def test_zero_band_retains_original_pi_even_at_exact_target():
    pi=PI(config(inventory_deadband_m=0.));pi.integral=.1;pi.valid_last=True
    assert pi.propose(2.,2.,.6,1.)['raw']==pytest.approx(.6)


@pytest.mark.parametrize('kw',[dict(inventory_deadband_m=True),dict(inventory_deadband_m=-1.),
    dict(inventory_deadband_m=.002),dict(output='steam_bypass'),dict(bias=.6),dict(lower=.6,bias=.6),
    dict(feedforward_sensor='feed_flow')])
def test_invalid_band_configuration(kw):
    with pytest.raises(ValueError):config(**kw)


def test_configured_native_restart_preserves_band_pi_and_continuation():
    owner=FaultSession(tick_s=.05,inventory_deadband_m=.0002)
    owner.step(modes={'feed_inventory_fraction':'auto'})
    cp=json.loads(json.dumps(owner.checkpoint(),sort_keys=True))
    restored=FaultSession.restore(cp)
    assert restored.initial['inventory_deadband_m']==.0002
    assert restored.control.loops['feed_inventory_fraction'].config.inventory_deadband_m==.0002
    assert owner.step()==restored.step()
    assert owner.checkpoint()==restored.checkpoint()


def test_in_band_targets_keep_real_valves_closed_but_do_not_hide_target_error():
    owner=FaultSession(tick_s=.05,inventory_deadband_m=.0002)
    target=owner.control.targets['feed_inventory_fraction']+.0001
    for i in range(4):
        result=owner.step(modes={'feed_inventory_fraction':'auto'} if i==0 else None,
                          targets={'feed_inventory_fraction':target} if i==0 else None)
    snap=owner.control.plant.snapshot()
    assert snap['feed_train']['makeup_flow_kg_s']==snap['feed_train']['drain_flow_kg_s']==0.
    observed=result['observation']['measurements']['feed_level']['value']
    assert 0<abs(observed-target)<.0002


def test_invalid_configured_command_is_atomic():
    owner=FaultSession(tick_s=.05,inventory_deadband_m=.0002);before=owner.checkpoint()
    with pytest.raises(ValueError):owner.step(targets={'feed_inventory_fraction':float('nan')})
    assert owner.checkpoint()==before


def test_preparation_selects_policy_and_continuation_cannot_change_it():
    from atomic_model_station_readiness import prepare_hot_state, ReadinessConfig
    c=ReadinessConfig(window_s=.02,maximum_gap_s=.01,relative_span=1e-12)
    result=prepare_hot_state(maximum_warmup_s=.03,tick_s=.01,config=c,inventory_deadband_m=.0002)
    assert result['checkpoint']['payload']['initial']['inventory_deadband_m']==.0002
    owner=FaultSession(tick_s=.01,inventory_deadband_m=.0002);before=owner.checkpoint()
    with pytest.raises(ValueError):
        prepare_hot_state(maximum_warmup_s=.03,tick_s=.01,config=c,initial_session=owner,inventory_deadband_m=0.)
    assert owner.checkpoint()==before


def test_latched_trip_overrides_neutral_policy():
    owner=FaultSession(tick_s=.05,inventory_deadband_m=.0002)
    result=owner.step({'scram':True},modes={'feed_inventory_fraction':'auto'})
    assert result['receipt']['trip_latched']
    snap=owner.control.plant.snapshot()
    assert not snap['controls']['feed_inventory_control']
    assert snap['feed_train']['makeup_flow_kg_s']==snap['feed_train']['drain_flow_kg_s']==0.
