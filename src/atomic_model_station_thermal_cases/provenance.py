from pathlib import Path
from atomic_model_multiphysics.provenance import sha
from atomic_model_station_thermal_path.provenance import runtime_identity


def case_identity():
    return dict(files={p.name:sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                backend=runtime_identity())
