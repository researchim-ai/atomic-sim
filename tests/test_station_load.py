from copy import deepcopy
from dataclasses import asdict
import json
import pytest
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_station_load import LoadCycle,assess,compare
from atomic_model_station_load.model import LOOPS
from scripts.qualify_station_load_cycle import run


@pytest.mark.parametrize('kwargs',[
    {'tick_s':True},{'duration_s':1801},{'points':((0,1),(450,.9))},
    {'points':((0,.98),(450,1))},{'points':((0,1),(900,.98),(450,1))},
    {'points':((0,1),(451,.98))},{'points':((0,1),(1795,.98))},
    {'settling_window_s':61},{'points':((0,1),(450,float('nan')))},
    {'points':((0,1),(450,True))}])
def test_rejects_ambiguous_schedule_and_incomplete_plateaus(kwargs):
    with pytest.raises(ValueError):LoadCycle(**kwargs)


def test_causal_edges_and_exact_episode_roundtrip():
    c=LoadCycle();e=c.episode();assert e.equilibrium_start and e.automatic
    assert e.ticks==360 and c.reference(89)==1 and c.reference(90)==.98 and c.reference(180)==1.02
    assert c.reference(270)==1 and e.commands[1]['tick']==90
    assert LoadCycle(**json.loads(json.dumps(asdict(c))))==c
    for tick in (-1,360,True,1.):
        with pytest.raises(ValueError):c.reference(tick)


def fake_records(c):
    rows=[]
    for i in range(c.ticks):
        f=c.reference(i);t=(i+1)*c.tick_s
        rows.append(dict(state=[1.,2.],load_fraction=f,
            physical=dict(time_s=t,powers=dict(gross_electrical_w=100.,own_needs_w=2.),energy_relative=0.,mass_relative=0.,boron_relative=0.),
            result=dict(receipt=dict(time_s=t-c.tick_s,completed_at_s=t,targets=dict(rod_target=3.15e9*f,feed_flow_fraction=1.,condensate_flow_fraction=1.,feed_inventory_fraction=1.),trip_latched=False,modes={k:'auto' for k in LOOPS}),
                        observation=dict(measurements={k:dict(value=v,quality=['good']) for k,v in [('power',3.15e9*f),('pressure',6.3e6),('speed',157.),('sg_level',1.),('hotwell_level',1.),('feed_level',1.)]}))))
    return rows


def test_full_trajectory_gates_do_not_hide_midpoint_problem_or_missing_samples():
    c=LoadCycle();rows=fake_records(c);assert assess(rows,c)['passed']
    assert not assess(rows[:-1],c)['passed']
    cases=[('clock',lambda r:r['physical'].update(time_s=123)),
           ('trip',lambda r:r['result']['receipt'].update(trip_latched=True)),
           ('mode',lambda r:r['result']['receipt']['modes'].update({'rod_target':'manual'})),
           ('command',lambda r:r.update(load_fraction=.97)),
           ('budget',lambda r:r['physical'].update(energy_relative=.1)),
           ('sensor',lambda r:r['result']['observation']['measurements']['power'].update(value=None,quality=['unavailable']))]
    for _,mutate in cases:
        bad=deepcopy(rows);mutate(bad[100]);a=assess(bad,c)
        assert not a['passed'];json.dumps(a,allow_nan=False)


def test_every_plateau_own_needs_and_tracking_are_checked():
    c=LoadCycle();rows=fake_records(c)
    rows[170]['physical']['powers']['own_needs_w']=101
    assert not assess(rows,c)['passed']
    rows=fake_records(c);rows[170]['result']['observation']['measurements']['power']['value']*=.9
    assert not assess(rows,c)['passed']


def test_refinement_checks_interior_and_refuses_layout_clock_and_nonfinite():
    c=LoadCycle();a=fake_records(c);b=deepcopy(a)
    b[100]['state'][0]=1.1
    assert not compare(a,b)['passed'] and compare(a,b)['worst']['tick']==100
    assert compare(a,b,accounting_indices=(0,))['passed']
    for mutate in [lambda r:r['physical'].update(time_s=0),lambda r:r.update(state=[1.]),lambda r:r.update(state=[1.,float('nan')])]:
        b=deepcopy(a);mutate(b[100]);assert not compare(a,b)['passed']
    assert not compare(a,a[:-1])['passed']


def test_real_coupled_cycle_refinement_and_native_restart(tmp_path):
    c=LoadCycle(tick_s=.01,duration_s=.08,points=((0.,1.),(.02,.999),(.04,1.001),(.06,1.)),settling_window_s=.02)
    a=run(tmp_path/'a',c,max_step_s=.1)
    b=run(tmp_path/'b',c,max_step_s=.05,rtol=2e-8,reference=tmp_path/'a')
    n=run(tmp_path/'native',c,max_step_s=.1,reference=tmp_path/'a',restart=True)
    assert a['completed'] and a['assessment']['passed'] and a['passed']
    assert b['passed'] and b['refinement']['passed']
    assert n['passed'] and n['native_prefix_replayed'] and n['native_whole_trajectory_exact']
    assert a['closed_tasks']==[] and not a['r1_accepted']
    cp=unseal(json.loads((tmp_path/'a/final.json').read_text()))
    assert len(cp['history'])==8 and cp['initial']['equilibrium_start']


def test_reference_refuses_changed_cycle_source_and_native_numerics_before_work(tmp_path):
    from scripts.qualify_station_load_cycle import identity
    c=LoadCycle();root=tmp_path/'reference';root.mkdir()
    p=dict(source=identity(),cycle=asdict(c),episode=c.episode().payload())
    path=root/'protocol.json';path.write_text(json.dumps(seal(p)))
    for kwargs in [dict(cycle=LoadCycle(points=((0.,1.),(450.,.99),(900.,1.02),(1350.,1.)))),dict(max_step_s=.5,restart=True)]:
        with pytest.raises(ValueError):run(tmp_path/'bad',reference=root,**kwargs)
        assert not (tmp_path/'bad').exists()
    p['source']['driver']='stale';path.write_text(json.dumps(seal(p)))
    with pytest.raises(ValueError,match='source'):run(tmp_path/'bad',reference=root)


def test_nonfinite_transition_clock_and_sensor_value_are_never_accepted():
    c=LoadCycle();rows=fake_records(c)
    rows[3]['result']['receipt']['time_s']=float('nan')
    assert not assess(rows,c)['passed']
    rows=fake_records(c);rows[170]['result']['observation']['measurements']['power']['value']=None
    report=assess(rows,c);assert not report['passed'];json.dumps(report,allow_nan=False)


def test_inventory_level_and_bad_quality_cannot_hide_behind_power_tracking():
    c=LoadCycle()
    for sensor in ('sg_level','hotwell_level','feed_level'):
        rows=fake_records(c);rows[170]['result']['observation']['measurements'][sensor]['value']+=.1
        assert not assess(rows,c)['passed']
        rows=fake_records(c);rows[100]['result']['observation']['measurements'][sensor]['quality']=['stale']
        assert not assess(rows,c)['passed']
