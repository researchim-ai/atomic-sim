"""Sealed corrected-reference profiles; explicit provenance, no implicit run."""
from importlib import import_module
from importlib.resources import files
import json
from atomic_model_multiphysics.provenance import unseal,sha
from . import model


def load_fixture():
    result=unseal(json.loads(files(__package__).joinpath('data/corrected_reference.json').read_text()))
    if result.get('format')!='openpipeflow-corrected-radial-quadrature/v1' or result.get('quadrature_policy')!=model.POLICY:
        raise ValueError('Explicit corrected quadrature reference required')
    runner=import_module('scripts.replay_station_corrected_quadrature')
    if result['harness_sha256']!=sha(runner.__file__) or result['math_source_sha256']!=sha(model.__file__):raise ValueError('Corrected replay source identity differs')
    original=import_module('scripts.replay_station_heated_reference')
    if result['original_harness_sha256']!=sha(original.__file__) or result['archive_sha256']!=original.ARCHIVE_SHA:raise ValueError('Original replay identity differs')
    expected=set(runner.cases());rows=list(result['records'].values())
    actual={(v['boundary'],v['c'],v['cells'],v['step']) for v in rows}
    if len(rows)!=14 or actual!=expected or result['subspace']!=dict(axial_modes=1,azimuthal_modes=1,turbulent=False):
        raise ValueError('Complete mean-only corrected matrix required')
    if result['physical_model_coefficients_changed'] or result['original_v67_fixture_updated']:
        raise ValueError('Numerical-only variant and immutable original fixture required')
    return result
