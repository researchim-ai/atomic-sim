"""Bounded portable visualization records with an exact UTF-8 integrity envelope."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from atomic_model_station_env import StationEnv
from atomic_model_station_env.model import signature
from atomic_model_workspace.common import canonical, digest
from atomic_model_training.spec import atomic_json

MAX_BYTES = 8 * 1024 * 1024
MAX_FRAMES = 1025
DRIVES = {'primary_pump','feed_pump','condensate_pump','cooling_pump','pressurizer_heater'}
REQUESTED = {'primary_pump_on','feed_pump_on','condensate_pump_on','cooling_pump_on','dhr_requested'}
FLOW_SPEC = {
    'steam': ('sg', 'turbine', 'kg/s'),
    'feed': ('feed', 'sg', 'kg/s'),
    'condensate': ('condenser', 'feed', 'kg/s'),
    'condenser_heat': ('condenser', 'cooling', 'W'),
    'external_heat': ('cooling', 'external', 'W'),
    'gross': ('turbine', 'bus', 'W'),
    'export': ('bus', 'grid', 'W'),
    'auxiliary': ('bus', 'auxiliary', 'W'),
    'dhr_heat': ('sg', 'cooling', 'W'),
}


def capture(env):
    """Inspect a copy of the physical owner without advancing instruments or caches."""
    if type(env) is not StationEnv:
        raise ValueError('Initialized StationEnv required')
    env._require()
    observation, info = env.observation()
    s = deepcopy(env._session.control.plant).snapshot()
    policy = env._session.control.policy
    values = dict(steam=s['steam_flow_kg_s'], feed=s['feed_train']['feed_flow_kg_s'],
        condensate=s['feed_train']['condensate_flow_kg_s'],
        condenser_heat=s['powers']['condenser_heat_out_w'],
        external_heat=s['powers']['external_cooling_heat_out_w'],
        gross=s['powers']['gross_electrical_w'], export=s['powers']['net_export_w'],
        auxiliary=s['powers']['own_needs_w'], dhr_heat=s['powers']['dhr_heat_w'])
    frame = dict(time_s=s['time_s'],
        flows={k:dict(source=a, target=b, unit=u, value=values[k]) for k,(a,b,u) in FLOW_SPEC.items()},
        equipment=dict(requested={k:s['controls'][k] for k in ('primary_pump_on','feed_pump_on',
            'condensate_pump_on','cooling_pump_on','dhr_requested')},
            effective_drives=deepcopy(s['electrical']['effective_drives']),
            primary_speed_rad_s=s['primary_pump_speed_rad_s'], turbine_speed_rad_s=s['turbine_speed_rad_s'],
            feed_check_open=s['feed_train']['feed_check_open'],
            condensate_check_open=s['feed_train']['condensate_check_open'],
            steam_inlet_position=s['actuator_status']['steam_inlet_position'],
            rod_position=s['actuator_status']['rod_position'],
            dhr_active=s['electrical']['dhr_active'], grid_available=s['electrical']['grid_available']),
        protections=dict(trip_latched=policy.trip_latched, trip_reasons=list(policy.trip_reasons),
            pressure_interlock=policy.pressure_active,
            scram_latched=s['actuator_status']['scram_latched'],
            turbine_trip_latched=s['actuator_status']['turbine_trip_latched'],
            alarms=deepcopy(info['detection']['alarms'])),
        measurements=deepcopy(observation['measurements']),
        balances={k:s[k] for k in ('energy_relative','mass_relative','boron_relative',
            'energy_residual_j','instantaneous_energy_defect_w')},
        events=deepcopy(s['events']))
    return json.loads(canonical(frame))


def _validate(body):
    if type(body) is not dict or set(body) != {'format','privileged','source','frames'}:
        raise ValueError('Complete station view document required')
    if body['format'] != 'station-view/v1' or body['privileged'] is not True:
        raise ValueError('Privileged station-view/v1 required')
    source=body['source']
    if type(source) is not dict or set(source) != {'station_sha256','viewer_sha256'}:
        raise ValueError('Source hashes required')
    if any(type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in source.values()):
        raise ValueError('SHA256 source required')
    frames=body['frames']
    if type(frames) is not list or not 1<=len(frames)<=MAX_FRAMES:
        raise ValueError('One to 1025 frames required')
    previous=-1.
    for f in frames:
        if type(f) is not dict or set(f)!={'time_s','flows','equipment','protections','measurements','balances','events'}:
            raise ValueError('Complete station frame required')
        t=f['time_s']
        if type(t) not in (int,float) or not math.isfinite(t) or t<0 or t<=previous:
            raise ValueError('Finite strictly increasing nonnegative frame clock required')
        previous=t
        if type(f['flows']) is not dict or set(f['flows'])!=set(FLOW_SPEC):raise ValueError('Fixed flow graph required')
        for k,(a,b,u) in FLOW_SPEC.items():
            flow=f['flows'][k]
            if (type(flow) is not dict or set(flow)!={'source','target','unit','value'} or
                (flow['source'],flow['target'],flow['unit'])!=(a,b,u) or
                type(flow['value']) not in (int,float) or not math.isfinite(flow['value'])):
                raise ValueError('Finite flow in declared SI units and orientation required')
        for key in ('equipment','protections','measurements','balances'):
            if type(f[key]) is not dict:raise ValueError('Structured station fields required')
        e=f['equipment']
        if set(e)!={'requested','effective_drives','primary_speed_rad_s','turbine_speed_rad_s',
            'feed_check_open','condensate_check_open','steam_inlet_position','rod_position','dhr_active','grid_available'}:
            raise ValueError('Complete equipment required')
        for k,names in [('requested',REQUESTED),('effective_drives',DRIVES)]:
            if type(e[k]) is not dict or set(e[k])!=names or any(type(v) is not bool for v in e[k].values()):
                raise ValueError('Complete boolean drive flags required')
        finite=lambda v:type(v) in (int,float) and math.isfinite(v)
        for k in ('primary_speed_rad_s','turbine_speed_rad_s','steam_inlet_position','rod_position'):
            if not finite(e[k]):raise ValueError('Finite equipment value required')
        for k in ('feed_check_open','condensate_check_open','dhr_active','grid_available'):
            if type(e[k]) is not bool:raise ValueError('Boolean equipment state required')
        p=f['protections']
        if set(p)!={'trip_latched','trip_reasons','pressure_interlock','scram_latched','turbine_trip_latched','alarms'}:
            raise ValueError('Complete protections required')
        if any(type(p[k]) is not bool for k in ('trip_latched','pressure_interlock','scram_latched','turbine_trip_latched')):
            raise ValueError('Boolean protection state required')
        if type(p['trip_reasons']) is not list or any(type(v) is not str for v in p['trip_reasons']) or type(p['alarms']) is not list:
            raise ValueError('Protection reason/alert lists required')
        if len(f['measurements'])>64:raise ValueError('At most 64 measurements')
        for r in f['measurements'].values():
            if (type(r) is not dict or set(r)!={'value','unit','sampled_at_s','available_at_s','age_s','quality'} or
                not (r['value'] is None or finite(r['value'])) or type(r['unit']) is not str or
                type(r['quality']) is not list or not r['quality'] or any(type(v) is not str for v in r['quality'])):
                raise ValueError('Complete measurement required')
            if (not all(finite(r[k]) for k in ('sampled_at_s','available_at_s','age_s')) or r['age_s']<0 or
                not r['sampled_at_s']<=r['available_at_s']<=t or abs(r['age_s']-(t-r['sampled_at_s']))>1e-8):
                raise ValueError('Consistent measurement clock required')
        b=f['balances']
        if set(b)!={'energy_relative','mass_relative','boron_relative','energy_residual_j','instantaneous_energy_defect_w'} or not all(finite(v) for v in b.values()):
            raise ValueError('Complete finite balances required')
        if any(b[k]<0 for k in ('energy_relative','mass_relative','boron_relative')):raise ValueError('Nonnegative relative residual required')
        if type(f['events']) is not list or len(f['events'])>10000:raise ValueError('Bounded event journal required')
        for event in f['events']:
            if (type(event) is not dict or type(event.get('kind')) is not str or
                not finite(event.get('time_s')) or not 0<=event['time_s']<=t+1e-8):
                raise ValueError('Timestamped physical event required')
    # Reject NaN even in a nested field which is not used by a particular viewer.
    canonical(body)
    return body


def document(frames):
    body=_validate(dict(format='station-view/v1',privileged=True,
        source=dict(station_sha256=digest(signature()),viewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        frames=deepcopy(frames)))
    text=canonical(body)
    result=dict(format='station-view-file/v1',sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),document=text)
    if len(canonical(result).encode('utf-8'))>MAX_BYTES:raise ValueError('Station view exceeds 8 MiB')
    return result


def load(value):
    if type(value) is not dict or set(value)!={'format','sha256','document'}:
        raise ValueError('Station view envelope required')
    if len(canonical(value).encode('utf-8'))>MAX_BYTES:raise ValueError('Station view exceeds 8 MiB')
    if value['format']!='station-view-file/v1' or type(value['document']) is not str:
        raise ValueError('Station view file version required')
    if hashlib.sha256(value['document'].encode('utf-8')).hexdigest()!=value['sha256']:
        raise ValueError('Station view integrity mismatch')
    return _validate(json.loads(value['document']))


def write(path, value):
    load(value)
    path=Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path,*path.parents)):
        raise ValueError('Fresh ordinary output path required')
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(path,value)
