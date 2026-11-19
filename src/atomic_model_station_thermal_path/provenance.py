"""Checkpoint source/environment identity independent of qualification scripts."""
import importlib
import platform
from pathlib import Path
import numpy as np
import scipy
from atomic_model_multiphysics.provenance import sha


def runtime_identity():
    names = ('atomic_model_station_thermal_path','atomic_model_station_thermal_resolved',
             'atomic_model_station_thermal','atomic_model_station_refined',
             'atomic_model_station_regular','atomic_model_station_stokes_pressure',
             'atomic_model_station_wall','atomic_model_station_advection')
    packages = {}
    for name in names:
        root = Path(importlib.import_module(name).__file__).parent
        packages[name] = {p.name:sha(p) for p in sorted(root.glob('*.py'))}
    codec={name:sha(importlib.import_module(name).__file__) for name in
           ('atomic_model_multiphysics.provenance','atomic_model_workspace.common')}
    return dict(packages=packages,codec=codec,python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
