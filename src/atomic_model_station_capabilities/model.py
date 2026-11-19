"""Separate API availability, input validity and recipe-specific numerical evidence."""
from copy import deepcopy
from dataclasses import asdict,fields
from importlib.resources import files
from pathlib import Path
import json
import hashlib
import re

from atomic_model_station_env import EnvConfig
from atomic_model_station_env.model import signature as model_identity,FLOAT_ACTIONS,BOOL_ACTIONS
from atomic_model_station_faults import Fault
from atomic_model_station.film_correlations import MODELS
from atomic_model_plant_extension import water
from atomic_model_multiphysics.provenance import seal,unseal,sha
from atomic_model_workspace.common import digest

CONFIG_FORMAT='station-env-config/v2'
RAW_FORMAT='envconfig-raw/v1'
_CONFIG_KEYS={f.name for f in fields(EnvConfig)}


def _hash(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def _config(values):
    if type(values) is not dict or set(values)!=_CONFIG_KEYS:
        raise ValueError('Complete EnvConfig fields required; unknown or missing fields are not discarded')
    values=deepcopy(values)
    if type(values['faults']) not in (list,tuple):raise ValueError('Explicit fault list required')
    if len(values['faults'])>64 or any(type(f) is not dict for f in values['faults']):
        raise ValueError('At most 64 fault dictionaries required')
    try:values['faults']=tuple(Fault(**f) for f in values['faults'])
    except TypeError as error:raise ValueError('Fault fields differ') from error
    return EnvConfig(**values)


def dump_config(config):
    if type(config) is not EnvConfig:raise ValueError('EnvConfig required')
    return seal(dict(format=CONFIG_FORMAT,model_source=digest(model_identity()),config=asdict(config)))


def _document(document):
    p=unseal(deepcopy(document))
    if type(p) is not dict or set(p)!={'format','model_source','config'} or p['format']!=CONFIG_FORMAT:
        raise ValueError('Station configuration format differs; checkpoints are not configuration files')
    if not _hash(p['model_source']):raise ValueError('Model source SHA256 required')
    return p


def load_config(document):
    p=_document(document)
    if p['model_source']!=digest(model_identity()):
        raise ValueError('Configuration model source changed; explicit configuration rebind required')
    return _config(p['config'])


def migrate_config(document,*,from_format,allow_rebind=False):
    """Return a new config envelope plus audit; never migrate latent state/history."""
    if type(allow_rebind) is not bool:raise ValueError('Boolean rebind flag required')
    previous=None
    if from_format==RAW_FORMAT:
        values=deepcopy(document)
    elif from_format==CONFIG_FORMAT:
        p=_document(document);values=p['config'];previous=p['model_source']
    else:raise ValueError('Unsupported explicit input configuration format')
    current=digest(model_identity())
    if previous is not None and previous!=current and not allow_rebind:
        raise ValueError('Explicit rebind required for a different model source')
    config=_config(values);result=seal(dict(format=CONFIG_FORMAT,model_source=current,config=asdict(config)))
    audit=dict(from_format=from_format,to_format=CONFIG_FORMAT,input_sha256=digest(document),
        previous_model_source=previous,model_source=current,fields_dropped=[],values_changed=False,
        fault_order_normalized=digest(values['faults'])!=digest(asdict(config)['faults']),
        requalification_required=previous!=current,simulation_executed=False,
        checkpoint_state_migrated=False,config_sha256=digest(asdict(config)))
    return result,audit


def _records():
    data=json.loads(files('atomic_model_station_capabilities').joinpath('data/evidence.json').read_text())
    p=unseal(data)
    if type(p) is not dict or set(p)!={'format','records'} or p['format']!='station-evidence-index/v1':
        raise ValueError('Evidence index format differs')
    if type(p['records']) is not list or len(p['records'])!=2:raise ValueError('Two published evidence records required')
    kinds=set()
    for r in p['records']:
        if (type(r) is not dict or set(r)!={'kind','source_sha256','receipt_sha256','receipt'} or
            r['kind'] not in ('nominal','tutorials') or r['kind'] in kinds or
            not _hash(r['source_sha256']) or not _hash(r['receipt_sha256'])):
            raise ValueError('Invalid or duplicate published evidence record')
        expected='station_nominal_requalification_v105.json' if r['kind']=='nominal' else 'station_examples_v106.json'
        if r['receipt']!=expected:raise ValueError('Unexpected evidence reference')
        kinds.add(r['kind'])
    return p['records']


def passport():
    from scripts.qualify_station_equilibrium import identity as nominal_identity
    from atomic_model_station_examples.model import identity as tutorial_identity
    source=digest(model_identity())
    current=dict(nominal=digest(nominal_identity()),tutorials=digest(tutorial_identity()))
    qualifications=[]
    scopes=dict(nominal='Declared 3600 s point-core equilibrium recipe; two source settings and installed fine/native replay',
        tutorials='Five short installed API tutorials; not physical scenario qualification')
    for r in _records():
        qualifications.append(dict(**r,current_source_matches=r['source_sha256']==current[r['kind']],
            scope=scopes[r['kind']],receipt_location='Published repository docs/benchmarks receipt; embedded index only'))
    resources=files('atomic_model_station_capabilities').joinpath('data/evidence.json').read_bytes()
    result=dict(format='station-capabilities/v1',model_source=source,
        implementation_source={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        evidence_index_sha256=hashlib.sha256(resources).hexdigest(),
        scope='StationEnv station API; not every solver in the repository',
        interfaces=dict(python=['reset','step','observation','available_actions','checkpoint','restore'],
            jsonl=['reset','step','observe','actions'],gymnasium='Measured normalized channels with quality masks'),
        observations=dict(measured_only=True,checkpoint_privileged=True,latent_state_in_agent_transport=False),
        operator_actions=dict(numbers={k:dict(minimum=v[0],maximum=v[1]) for k,v in FLOAT_ACTIONS.items()},booleans=list(BOOL_ACTIONS)),
        configuration=dict(format=CONFIG_FORMAT,legacy_input=RAW_FORMAT,full_fields_required=True,
            defaults=asdict(EnvConfig()),model_source_bound=True,checkpoint_migration=False),
        water_chart=dict(pressure_pa=[water.P_MIN,water.P_MAX],temperature_k=[water.T_MIN,water.T_MAX],
            regions=[1,2,4],whole_plant_qualified_range=False),film_models=list(MODELS),
        qualifications=qualifications,
        limitations=[
            dict(id='coastdown',ru='Полный выбег не принят; область теплоотдачи и экран плавучести сохраняются.',
                en='Full coastdown remains unaccepted; heat-transfer bounds and buoyancy screening remain active.'),
            dict(id='validation',ru='Экспериментальная валидация реальной установки отсутствует.',
                en='No experimental validation of a real installation is claimed.'),
            dict(id='r1',ru='Полная обязательная матрица R1 ещё не принята.',en='The complete mandatory R1 matrix remains unaccepted.'),
            dict(id='scope',ru='Принятый рецепт и область EOS не квалифицируют произвольную конфигурацию.',
                en='An accepted recipe and EOS chart bounds do not qualify arbitrary configurations.')],
        experimental_validation=False,full_r1_accepted=False,industrial_plant_qualified=False)
    if source!=digest(model_identity()):raise ValueError('Model source changed while reading passport')
    return result
