"""Read and verify curated metadata and already-bundled primary artifacts offline."""
from importlib.resources import files
from pathlib import PurePosixPath,Path
import hashlib
import json
import re
from atomic_model_multiphysics.provenance import unseal,sha

PACKAGES={'atomic_model_plant_extension','atomic_model_validation','atomic_model_acceptance'}
FIELDS={'id','system','source','conditions','observables','rights','uncertainty','gaps','artifacts','bundled_data_level'}


def _manifest():
    return unseal(json.loads(files('atomic_model_station_evidence').joinpath('data/candidates.json').read_text()))


def candidate_registry():
    p=_manifest()
    if type(p) is not dict or set(p)!={'format','candidates'} or p['format']!='station-data-candidates/v1':
        raise ValueError('Candidate registry format differs')
    if type(p['candidates']) is not list or not 1<=len(p['candidates'])<=32:raise ValueError('Bounded nonempty candidate registry required')
    rows=[];ids=set()
    for record in p['candidates']:
        if type(record) is not dict or set(record)!=FIELDS:raise ValueError('Exact candidate metadata fields required')
        key=record['id']
        if type(key) is not str or re.fullmatch('[a-z][a-z0-9_-]{0,63}',key) is None or key in ids:
            raise ValueError('Safe unique candidate identifier required')
        ids.add(key)
        for field in ('system','conditions','bundled_data_level'):
            if type(record[field]) is not str or not record[field].strip():raise ValueError('Explicit candidate scope required')
        source=record['source']
        if type(source) is not dict or set(source)!={'primary_url','version','locator'}:
            raise ValueError('Versioned primary source required')
        if not all(type(x) is str and x.strip() for x in source.values()) or not source['primary_url'].startswith('https://'):
            raise ValueError('Explicit primary source URL, version and locator required')
        observables=record['observables']
        if type(observables) is not list or not observables:raise ValueError('Measured observables required')
        for observable in observables:
            if (type(observable) is not dict or set(observable)!={'quantity','unit'} or
                type(observable['quantity']) is not str or not observable['quantity'].strip() or
                observable['unit'] not in ('K','kg/s','pcm','1')):
                raise ValueError('Explicit supported observable units required')
        rights=record['rights'];uncertainty=record['uncertainty']
        if (type(rights) is not dict or set(rights)!={'publication_license','raw_data_rights','raw_measurements_bundled'} or
            type(rights['raw_measurements_bundled']) is not bool or
            any(type(rights[k]) is not str or not rights[k].strip() for k in ('publication_license','raw_data_rights'))):
            raise ValueError('Publication and raw-measurement rights must be separate')
        if (type(uncertainty) is not dict or set(uncertainty)!={'complete_joint_budget','statement'} or
            type(uncertainty['complete_joint_budget']) is not bool or type(uncertainty['statement']) is not str or not uncertainty['statement'].strip()):
            raise ValueError('Explicit uncertainty semantics required')
        if type(record['gaps']) is not list or not record['gaps'] or any(type(v) is not str or not v.strip() for v in record['gaps']):
            raise ValueError('Explicit unresolved station-applicability gaps required')
        if type(record['artifacts']) is not list or not record['artifacts']:raise ValueError('Pinned bundled evidence required')
        verified=[]
        for artifact in record['artifacts']:
            if type(artifact) is not dict or set(artifact)!={'package','path','sha256'}:raise ValueError('Exact artifact fields required')
            package=artifact['package'];path=artifact['path'];expected=artifact['sha256']
            if (type(package) is not str or package not in PACKAGES or type(path) is not str or not path or '\\' in path or ':' in path or
                PurePosixPath(path).is_absolute() or any(s in ('','..','.') for s in path.split('/'))):
                raise ValueError('Bundled evidence path outside allowed packages')
            if type(expected) is not str or re.fullmatch('[0-9a-f]{64}',expected) is None:raise ValueError('Pinned artifact SHA256 required')
            actual=hashlib.sha256(files(package).joinpath(path).read_bytes()).hexdigest()
            if actual!=expected:raise ValueError('Bundled candidate artifact changed: '+key+'/'+path)
            verified.append(dict(**artifact,integrity_verified=True))
        rows.append(dict(**{k:v for k,v in record.items() if k!='artifacts'},artifacts=verified,
            admitted_for_station_validation=False,experimental_validation=False,
            status='candidate-only; checksum integrity is not experimental admission'))
    resource=files('atomic_model_station_evidence').joinpath('data/candidates.json').read_bytes()
    return dict(format='station-data-candidates/v1',candidates=rows,total_candidates=len(rows),
        registry_sha256=hashlib.sha256(resource).hexdigest(),
        implementation_source={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        experimental_validation=False,full_r2_accepted=False)
