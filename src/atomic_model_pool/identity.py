import hashlib
from pathlib import Path

from atomic_model_bench.spec import source_identity
from atomic_model_workspace.common import digest


def identity():
    return dict(
        pool_sha256=digest(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            }
        ),
        baseline=source_identity(),
    )
