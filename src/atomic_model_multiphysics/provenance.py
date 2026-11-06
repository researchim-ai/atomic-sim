"""Content-addressed manifests for evaluated and numerical coefficient families."""

import hashlib
from pathlib import Path

from atomic_model_spatial.workflow import identity as spatial_identity
from atomic_model_training.episode import identity as training_identity
from atomic_model_workspace.common import digest


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024**2):
            h.update(block)
    return h.hexdigest()


def identity():
    return dict(
        spatial=spatial_identity(),
        training=training_identity(),
        files={p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        resources={
            str(p.relative_to(Path(__file__).parent)): sha(p)
            for p in sorted(Path(__file__).parent.joinpath("data").glob("*"))
            if p.is_file()
        },
    )


def seal(value):
    return dict(payload=value, sha256=digest(value))


def unseal(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"payload", "sha256"}
        or digest(value["payload"]) != value["sha256"]
    ):
        raise ValueError("Invalid content envelope")
    return value["payload"]
