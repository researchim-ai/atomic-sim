"""Separate evidence registers and scope-specific promotion with real replay."""

from pathlib import Path

from atomic_model_controlled_plant.session import Session
from atomic_model_workspace.common import digest

from .episode import identity
from .runner import envelope, unpack
from .spec import atomic_json, promotion


class EvidenceStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def record(self, kind, artifact):
        promotion(kind, "implemented")  # Kind admission, never physical promotion.
        entry = dict(
            kind=kind,
            artifact=artifact,
            artifact_sha256=digest(artifact),
            source_sha256=digest(identity()),
            validated=False,
            plant_referenced=False,
        )
        folder = self.directory / kind
        folder.mkdir(exist_ok=True)
        path = folder / (digest(entry) + ".json")
        if path.exists():
            if unpack(path) != entry:
                raise ValueError("Evidence identity collision")
        else:
            atomic_json(path, envelope(entry))
        return path

    def promote(self, path, requested):
        path = Path(path)
        if path.parent.parent.resolve() != self.directory.resolve():
            raise ValueError("Evidence outside register")
        entry = unpack(path)
        if (
            path.parent.name != entry["kind"]
            or digest(entry["artifact"]) != entry["artifact_sha256"]
        ):
            raise ValueError("Evidence kind/content differs")
        result = promotion(entry["kind"], requested)
        if requested == "verified":
            receipt = entry["artifact"]
            if (
                entry["kind"] != "numerical"
                or entry["source_sha256"] != digest(identity())
                or receipt.get("source_sha256") != digest(identity())
                or receipt.get("status") != "completed"
            ):
                raise ValueError(
                    "Only completed current-core numerical replay can be verified here"
                )
            Session.restore(receipt["checkpoint"], max_replay_steps=256)
            result["replayed_checkpoint_sha256"] = receipt["checkpoint"]["sha256"]
        return result
