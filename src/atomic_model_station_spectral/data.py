"""Compact replay summaries; large third-party checkpoints stay external."""
from importlib import import_module
from importlib.resources import files
import json
from atomic_model_multiphysics.provenance import unseal,sha
from . import io,modes


def load_fixture():
    p=unseal(json.loads(files(__package__).joinpath('data/spatial_probe.json').read_text()))
    runner=import_module('scripts.replay_station_spatial')
    if p['format']!='heated-spatial-integration-probe/v1' or p['harness_sha256']!=sha(runner.__file__):raise ValueError('Spatial producer identity differs')
    if p['audit_source_sha256']!=sha(io.__file__) or p['modes_source_sha256']!=sha(modes.__file__):raise ValueError('Spatial audit implementation differs')
    if p['seed_archive_sha256']!=runner.SEED_SHA or set(p['records'])!={'whole','time_half','fourier_fine','radial_fine','restart'}:
        raise ValueError('Complete pinned spatial experiment matrix required')
    return p
