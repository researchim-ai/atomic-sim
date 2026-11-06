"""Fixed loopback UI operations; uploads contain bytes, never server file paths."""

import base64
import binascii

from atomic_model_study import evaluate, store
from atomic_model_study.manager import Manager
from atomic_model_workspace.common import canonical, keys


class Backend:
    study_enabled = True
    source_identity = staticmethod(store.source_identity)
    inspect = staticmethod(store.inspect)
    preview = staticmethod(store.preview)
    edit = staticmethod(store.edit)
    restore = staticmethod(store.restore)
    export = staticmethod(store.export)

    def __init__(self, folder, **kwargs):
        self.manager = Manager(folder, **kwargs)

    def dispatch(self, operation, folder, value):
        if operation in ("data-preview", "data-attach"):
            keys(value, ("request", "csv_base64"))
            if not isinstance(value["csv_base64"], str) or len(value["csv_base64"]) > 11184812:
                raise ValueError("CSV upload limit is 8 MiB")
            if len(canonical(value["request"]).encode()) > 1024**2:
                raise ValueError("Reference declaration/request limit is 1 MiB")
            try:
                raw = base64.b64decode(value["csv_base64"], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Invalid base64 CSV") from exc
            if len(raw) > 8 * 1024**2:
                raise ValueError("CSV upload limit is 8 MiB")
            fn = store.preview_data if operation == "data-preview" else store.attach_data
            return fn(folder, value["request"], raw)
        if operation == "data-detach":
            keys(value, ("expected_revision", "expected_checkpoint_sha256", "id"))
            return store.detach(folder, **value)
        if operation == "evaluation-start":
            keys(value, ("binding_id", "expected_revision", "expected_checkpoint_sha256"))
            return self.manager.start(**value)
        if operation in (
            "evaluation-status",
            "evaluation-cancel",
            "evaluation-resume",
            "evaluation-summary",
        ):
            keys(value, ("evaluation_id",))
            fn = {
                "evaluation-status": self.manager.status,
                "evaluation-cancel": self.manager.cancel,
                "evaluation-resume": self.manager.resume,
                "evaluation-summary": lambda id: dict(
                    evaluate.summary(folder, id), stale=self.manager.status(id)["stale"]
                ),
            }[operation]
            return fn(value["evaluation_id"])
        if operation == "evaluation-curve":
            keys(value, ("evaluation_id", "channel"))
            return dict(
                evaluate.curve(folder, value["evaluation_id"], value["channel"]),
                stale=self.manager.status(value["evaluation_id"])["stale"],
            )
        raise ValueError("Unknown study operation")

    def close(self):
        self.manager.close()
