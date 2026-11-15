import json
import math
from decimal import Decimal,localcontext

import numpy as np
import pytest
from atomic_model_station import PWRBypassSession
from atomic_model_station_signals import Sensor,StationSignals
from atomic_model_multiphysics.provenance import seal,unseal


def signal(**kw):return Sensor('power','fission_power_w',**kw)
def push(owner,tick,value):return owner.ingest(tick,{'fission_power_w':value})['measurements']['power']


def test_linear_ramp_matches_independent_analytic_first_order_response():
    s=StationSignals([signal(lag_s=2.)])
    for t in range(21):
        r=push(s,t,float(t))
        expected=t-2+2*math.exp(-t/2)
        assert r['value']==pytest.approx(expected,abs=1e-13)
        assert r['sampled_at_s']==t and r['available_at_s']==t


def test_small_ratio_lag_remains_accurate_against_decimal_reference():
    s=StationSignals([signal(lag_s=1e6)],tick_s=.01)
    push(s,0,0.);value=push(s,1,1.)['value']
    with localcontext() as ctx:
        ctx.prec=60;x=Decimal('1e-8')
        expected=float(1-(1-(-x).exp())/x)
    assert value==pytest.approx(expected,rel=1e-14)


def test_sample_and_transport_clocks_never_deliver_future_truth():
    s=StationSignals([signal(period_ticks=2,delay_ticks=3)])
    for t in range(10):
        r=push(s,t,float(t))
        if t<3:assert r['value'] is None and r['quality']==['initializing']
        else:
            sampled=((t-3)//2)*2
            assert r['value']==sampled and r['sampled_at_s']==sampled
            assert r['available_at_s']==sampled+3 and r['age_s']==t-sampled


def test_staleness_and_range_quality_preserve_unclipped_reading():
    s=StationSignals([signal(period_ticks=5,delay_ticks=1,stale_after_ticks=1,measurement_max=5.)])
    for t in range(4):r=push(s,t,10.)
    assert r['value']==10. and r['quality']==['stale','out_of_range']


def test_noise_has_declared_scale_without_clipping():
    s=StationSignals([signal(sigma=2.)],seed=17)
    values=np.array([push(s,t,0.)['value'] for t in range(2000)])
    assert abs(values.mean())<.15 and 1.85<values.std()<2.15
    assert values.min()<0<values.max()


def test_adding_or_reordering_channels_does_not_change_existing_noise_stream():
    a=StationSignals([signal(sigma=1.)],seed=5)
    b=StationSignals([Sensor('aardvark','fission_power_w',sigma=1.),signal(sigma=1.)],seed=5)
    for t in range(30):assert push(a,t,3.)==push(b,t,3.)


def test_checkpoint_restores_pending_deliveries_analog_memory_and_noise_exactly():
    s=StationSignals([signal(period_ticks=2,delay_ticks=5,lag_s=3.,sigma=1.)],seed=7)
    for t in range(9):push(s,t,float(t))
    saved=json.loads(json.dumps(s.checkpoint()));r=StationSignals.restore(saved)
    assert r.checkpoint()==saved
    for t in range(9,60):assert push(r,t,float(t))==push(s,t,float(t))


def test_initial_checkpoint_restores_before_any_measurement():
    s=StationSignals([signal(sigma=1.)]);r=StationSignals.restore(json.loads(json.dumps(s.checkpoint())))
    assert push(s,0,1.)==push(r,0,1.)


def test_operator_payload_and_caller_mutation_cannot_expose_or_modify_private_state():
    s=StationSignals([signal(delay_ticks=2,sigma=1.)]);push(s,0,10.)
    before=s.checkpoint();view=s.operator_frame()
    assert set(view)=={'time_s','measurements'}
    assert set(view['measurements']['power'])=={'value','unit','sampled_at_s','available_at_s','age_s','quality'}
    view['measurements']['power']['value']=-1.;view['measurements']['power']['quality'].append('fake')
    assert s.checkpoint()==before


@pytest.mark.parametrize('tick,values',[(2,{'fission_power_w':1.}),(True,{'fission_power_w':1.}),(1,{}),
    (1,{'fission_power_w':True}),(1,{'fission_power_w':float('nan')}),(1,{'fission_power_w':float('inf')}),
    (1,{'fission_power_w':10**400}),(1,{'fission_power_w':1.,'hidden':2.})])
def test_rejected_truth_preserves_instrument_memory_queue_clock_and_rng(tick,values):
    s=StationSignals([signal(sigma=2.)]);push(s,0,1.);before=s.checkpoint()
    with pytest.raises(ValueError):s.ingest(tick,values)
    assert s.checkpoint()==before


@pytest.mark.parametrize('changes',[dict(period_ticks=True),dict(delay_ticks=-1),dict(lag_s=float('nan')),
    dict(sigma=-1.),dict(quantity=[]),dict(measurement_min=2.,measurement_max=1.)])
def test_invalid_sensor_contract(changes):
    kw=dict(name='x',quantity='fission_power_w');kw.update(changes)
    with pytest.raises(ValueError):Sensor(**kw)


@pytest.mark.parametrize('field',['queue','latest','analog','source'])
def test_restore_rejects_inconsistent_or_foreign_state(field):
    s=StationSignals([signal(delay_ticks=2)])
    for t in range(4):push(s,t,float(t))
    p=unseal(s.checkpoint())
    if field=='source':p['source']={}
    elif field=='queue':p['channels']['power']['queue'].pop()
    elif field=='latest':p['channels']['power']['latest']['available_tick']=99
    else:p['channels']['power']['analog']=True
    with pytest.raises(ValueError):StationSignals.restore(seal(p))


def test_adapter_reads_declared_quantities_only_and_does_not_mutate_plant():
    plant=PWRBypassSession();before=plant.checkpoint()
    s=StationSignals([Sensor('p','steam_pressure_pa',delay_ticks=1),signal()])
    frame=s.ingest_station(plant.snapshot())
    assert frame['measurements']['p']['unit']=='Pa' and frame['measurements']['p']['value'] is None
    assert frame['measurements']['power']['value']==plant.snapshot()['core']['fission_total_w']
    assert plant.checkpoint()==before
    before=s.checkpoint();bad=plant.snapshot();bad['time_s']=.5
    with pytest.raises(ValueError):s.ingest_station(bad)
    assert s.checkpoint()==before


def test_quantity_catalog_is_immutable():
    from atomic_model_station_signals.model import CATALOG
    with pytest.raises(TypeError):CATALOG['fission_power_w']=(('hidden_state',),'arbitrary')
