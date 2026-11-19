"""Load locally generated independent numerical profiles with sealed provenance."""
from importlib.resources import files
import json
from importlib import import_module
from atomic_model_multiphysics.provenance import sha
from atomic_model_multiphysics.provenance import unseal


def load_external():
    data=unseal(json.loads(files(__package__).joinpath('data/openpipeflow_mean.json').read_text()))
    if data.get('format')!='openpipeflow-hp-mean-only/v1' or data.get('subspace')!=dict(axial_modes=1,azimuthal_modes=1,turbulent=False):
        raise ValueError('Explicit mean-only external fixture required')
    if data['external_archive_sha256']!='4652325064ce4ecb2b351875d17bf1b3baa58f2f89529c365191bc9ad98bc269':
        raise ValueError('External archive identity differs')
    if len(data['records'])!=14:raise ValueError('Complete external boundary/grid/time matrix required')
    if data['harness_sha256']!=sha(import_module('scripts.replay_station_heated_reference').__file__):
        raise ValueError('External replay harness differs')
    expected={(bc,c,n,.1 if bc=='delta_temperature' else .01) for bc in ('delta_temperature','heat_flux') for c in (0,2,5) for n in (32,64)}
    expected.update({('delta_temperature',2,64,.05),('heat_flux',2,64,.005)})
    actual={(v['boundary'],v['c'],v['cells'],v['step']) for v in data['records'].values()}
    if actual!=expected:raise ValueError('External case matrix differs')
    return data
