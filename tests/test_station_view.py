from copy import deepcopy
import json
import hashlib
from pathlib import Path
import pytest
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_view import capture,document,load,write,MAX_BYTES
from atomic_model_station_view.__main__ import demo
from atomic_model_workspace.common import canonical

@pytest.fixture(scope='module')
def recording():return demo()


def test_capture_does_not_change_live_checkpoint_or_next_transition():
    env=StationEnv(EnvConfig(max_steps=2,tick_s=.05,automatic=False));env.reset();other=deepcopy(env)
    before=env.checkpoint();a=capture(env);b=capture(env)
    assert a==b and env.checkpoint()==before
    assert env.step({})==other.step({}) and env.checkpoint()==other.checkpoint()
    assert 'pwr_state' not in a and 'state' not in a and 'checkpoint' not in a
    assert a['flows']['export']['value']==a['flows']['gross']['value']-a['flows']['auxiliary']['value']


def test_native_demo_retains_fault_quality_and_trip(recording):
    body=load(recording);frames=body['frames'];assert len(frames)==5
    assert [f['time_s'] for f in frames]==[0.,.05,.1,.15000000000000002,.2]
    assert frames[1]['equipment']['requested']['feed_pump_on'] is False
    assert frames[1]['equipment']['effective_drives']['feed_pump'] is False
    assert frames[1]['flows']['feed']['value']==0 # native check valve has closed
    assert any(f['measurements']['pressure']['quality']!=['good'] for f in frames)
    assert frames[-1]['protections']['scram_latched']
    assert frames[-1]['events']
    assert body['privileged'] is True and len(body['source']['station_sha256'])==64


def test_export_is_detached_and_digest_covers_exact_utf8(recording):
    x=deepcopy(recording);body=load(x);body['frames'][0]['flows']['steam']['value']=0
    assert load(x)['frames'][0]['flows']['steam']['value']!=0
    assert hashlib.sha256(x['document'].encode()).hexdigest()==x['sha256']
    x['document']+=' '
    with pytest.raises(ValueError,match='integrity'):load(x)

@pytest.mark.parametrize('mutation',[
 lambda b:b.update(privileged=False),
 lambda b:b.update(format='other'),
 lambda b:b['frames'].append(deepcopy(b['frames'][0])),
 lambda b:b['frames'][0]['flows']['steam'].update(unit='MW'),
 lambda b:b['frames'][0]['flows']['steam'].update(value=float('nan')),
 lambda b:b['frames'][0]['equipment']['effective_drives'].update(feed_pump='yes'),
 lambda b:b['frames'][0]['protections'].pop('trip_reasons'),
 lambda b:b['frames'][0]['measurements']['pressure'].update(age_s=10),
 lambda b:b['frames'][0]['balances'].update(energy_relative=-1),
 lambda b:b['frames'][0]['events'].append(dict(kind='future',time_s=1)),
 lambda b:b['source'].update(station_sha256='x'*64),
])
def test_malformed_trace_refused_before_publication(recording,mutation):
    b=load(recording);mutation(b)
    with pytest.raises(ValueError):
        text=canonical(b)
        load(dict(format='station-view-file/v1',document=text,sha256=hashlib.sha256(text.encode()).hexdigest()))


def test_output_is_new_bounded_and_symlink_safe(recording,tmp_path):
    target=tmp_path/'nested/view.json';write(target,recording)
    assert load(json.loads(target.read_text()))==load(recording)
    with pytest.raises(ValueError):write(target,recording)
    (tmp_path/'link').symlink_to(target.parent,target_is_directory=True)
    with pytest.raises(ValueError):write(tmp_path/'link/other.json',recording)
    with pytest.raises(ValueError):load(dict(format='station-view-file/v1',sha256='x',document='x'*MAX_BYTES))
    with pytest.raises(ValueError):document([])
    with pytest.raises(ValueError):document(load(recording)['frames']*300)


def test_uninitialized_env_refused():
    with pytest.raises(ValueError):capture(StationEnv())
    with pytest.raises(ValueError):capture(None)


def test_unpowered_primary_pump_keeps_native_rotation():
    env=StationEnv(EnvConfig(max_steps=1,tick_s=.01,automatic=False));env.reset()
    initial=capture(env);env.step({'operator':{'primary_pump_on':False}});f=capture(env)
    assert f['equipment']['effective_drives']['primary_pump'] is False
    assert 0<f['equipment']['primary_speed_rad_s']<initial['equipment']['primary_speed_rad_s']
