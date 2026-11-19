"""Identity of the independent mixed-convection study, not the station solver."""
from pathlib import Path
import platform
import numpy
import scipy
from physics.materials import iapws
from atomic_model_multiphysics.provenance import sha


def identity():
    root=Path(__file__).parent
    return dict(files={p.name:sha(p) for p in sorted(root.glob('*.py'))},
                resources={p.name:sha(p) for p in sorted((root/'data').glob('*.json'))},
                water_properties_sha256=sha(iapws.__file__),python=platform.python_version(),
                numpy=numpy.__version__,scipy=scipy.__version__)
