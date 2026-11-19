from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
import shutil
import pytest
from atomic_model_multiphysics.provenance import seal,unseal
from atomic_model_training.spec import atomic_json
from atomic_model_station_load import LoadCycle
from scripts import qualify_station_load_cycle as driver
from scripts import qualify_station_load_envelope as study
from scripts.qualify_station_inventory import read


@pytest.mark.parametrize('a,b,accepted',[
    ([],[],True),([{'kind':'switch','time_s':1.}],[{'kind':'switch','time_s':1.04}],True),
    ([{'kind':'switch','time_s':1.}],[{'kind':'switch','time_s':1.06}],False),
    ([{'kind':'switch','time_s':1.}],[{'kind':'other','time_s':1.}],False),
    ([{'kind':'switch','time_s':1.}],[],False)])
def test_event_time_and_kind_are_both_required(a,b,accepted):
    assert study.event_agreement(a,b)['passed']==accepted


@pytest.mark.parametrize('event',[{'kind':'x','time_s':float('nan')},{'kind':'x','time_s':True},
    {'kind':'x','time_s':-1.},{'kind':1,'time_s':1.}])
def test_event_audit_refuses_nonphysical_data(event):
    with pytest.raises(ValueError):study.event_agreement([event],[])


@pytest.fixture(scope='module')
def cases(tmp_path_factory):
    root=tmp_path_factory.mktemp('load-envelope')
    c=LoadCycle(tick_s=.01,duration_s=.08,points=((0.,1.),(.02,.999),(.04,1.001),(.06,1.)),settling_window_s=.02)
    for name,(step,tol) in study.SETTINGS.items():
        report=driver.run(root/name,c,max_step_s=step,rtol=tol,reference=None if name=='coarse' else root/'coarse')
        assert report['passed']
    atomic_json(root/'protocol.json',seal(dict(source=study.identity(),cycle=asdict(c))))
    return root,c


def test_full_audit_recomputes_tracking_limits_and_every_tick_own_needs(cases,tmp_path):
    root,c=cases;check,rows,layout=study.inspect_case(root/'coarse',c,study.SETTINGS['coarse'])
    assert check['passed'] and check['own_needs_supplied_every_tick'] and check['actuator_bounds_valid']
    assert layout['size']==78 and len(rows)==c.ticks
    for name,change in [('actuator',lambda r:r['result']['receipt']['actual_positions'].update(rod=1.01)),
        ('own_needs',lambda r:r['physical']['powers'].update(own_needs_w=1e20)),
        ('trip',lambda r:r['result']['receipt'].update(trip_latched=True))]:
        p=tmp_path/name;shutil.copytree(root/'coarse',p);data=read(p/'trajectory.json');change(data['records'][0]);atomic_json(p/'trajectory.json',seal(data))
        assert not study.inspect_case(p,c,study.SETTINGS['coarse'])[0]['passed']


def test_complete_state_schema_and_declared_settings_are_mandatory(cases,tmp_path):
    root,c=cases
    with pytest.raises(ValueError,match='settings'):study.inspect_case(root/'coarse',c,study.SETTINGS['fine'])
    p=tmp_path/'incomplete';shutil.copytree(root/'coarse',p)
    data=read(p/'trajectory.json');data['records'].pop();atomic_json(p/'trajectory.json',seal(data))
    with pytest.raises(ValueError,match='Complete'):study.inspect_case(p,c,study.SETTINGS['coarse'])
    p=tmp_path/'mask';shutil.copytree(root/'coarse',p)
    data=read(p/'initial.json');data['layout']['accounting_indices'].append(26);atomic_json(p/'initial.json',seal(data))
    with pytest.raises(ValueError,match='schema'):study.inspect_case(p,c,study.SETTINGS['coarse'])


def test_three_setting_summary_and_interior_difference(cases,tmp_path):
    root,c=cases;normal=study.summarize(root,c)
    assert normal['passed'] and normal['load_cycle_candidate'] and not normal['npp_11_02_accepted']
    assert normal['closed_tasks']==[] and all(e['passed'] for e in normal['event_agreement'].values())
    p=tmp_path/'perturbed';shutil.copytree(root,p)
    data=read(p/'fine/trajectory.json');data['records'][2]['state'][26]*=1.01
    atomic_json(p/'fine/trajectory.json',seal(data))
    report=study.summarize(p,c)
    assert not report['passed'] and report['refinement']['fine']['worst']['tick']==2


def test_native_restores_actual_midpoint_and_compares_entire_suffix(cases,tmp_path):
    root,c=cases;report=study.native(tmp_path/'native',root/'coarse')
    assert report['passed'] and report['prefix_exact'] and report['suffix_and_final_exact']
    assert report['prefix_s']==report['suffix_s']==.04 and not report['copied_prefix_rows']
    expected=read(root/'coarse/trajectory.json')['records'][4:]
    assert read(tmp_path/'native/suffix.json')['records']==expected
    assert (tmp_path/'native/final.json').read_bytes()==(root/'coarse/final.json').read_bytes()


def test_native_refuses_corrupted_prefix(cases,tmp_path):
    root,c=cases;p=tmp_path/'bad';shutil.copytree(root/'coarse',p)
    data=read(p/'midpoint.json');data['history'][0]['operator']={'load_fraction':.95}
    atomic_json(p/'midpoint.json',seal(data))
    with pytest.raises(ValueError):study.native(tmp_path/'native',p)


def test_physical_switch_difference_fails_even_when_coordinates_match(cases,tmp_path):
    root,c=cases;p=tmp_path/'events';shutil.copytree(root,p)
    final=read(p/'finer/final.json')
    final['control']['payload']['state']['plant']['payload']['snapshot']['events'].append(dict(kind='bypass_supply_close',time_s=.03))
    atomic_json(p/'finer/final.json',seal(final))
    report=study.summarize(p,c)
    assert not report['passed'] and report['refinement']['coarse']['passed'] and not report['event_agreement']['coarse']['passed']


def test_source_epoch_and_unknown_worker_are_refused(cases,tmp_path):
    root,c=cases;p=tmp_path/'stale';shutil.copytree(root,p)
    protocol=read(p/'protocol.json');protocol['source']['envelope_sha256']='stale';atomic_json(p/'protocol.json',seal(protocol))
    with pytest.raises(ValueError,match='protocol'):study.summarize(p,c)
    with pytest.raises(ValueError,match='role'):study.worker('unknown',tmp_path/'out',root/'coarse')
