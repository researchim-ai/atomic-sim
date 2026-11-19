"""Identity kept separate from the unchanged v63-v67 studies."""
from pathlib import Path
import platform
import numpy,scipy
from atomic_model_multiphysics.provenance import sha


def identity():
    root=Path(__file__).parent
    return dict(files={p.name:sha(p) for p in sorted(root.glob('*.py'))},
        resources={p.name:sha(p) for p in sorted((root/'data').glob('*.json'))},
        python=platform.python_version(),numpy=numpy.__version__,scipy=scipy.__version__)
