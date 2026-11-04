"""Optional UI extension; original study source identity and journals stay intact."""

import tempfile
from pathlib import Path

from atomic_model_results import bundle
from atomic_model_study.backend import Backend as StudyBackend
from atomic_model_workspace.common import keys, read


class Backend(StudyBackend):
    results_enabled = True

    def inspect(self, folder):
        result = super().inspect(folder)
        rows = []
        ids = sorted(p.name for p in (Path(folder) / "evaluations").iterdir() if p.is_dir())
        for id in ids[:64]:
            try:
                bundle.evaluate.evaluation_path(folder, id)
                state = read(Path(folder) / "evaluations" / id / "status.json")
                rows.append(
                    dict(
                        evaluation_id=id,
                        status=state["status"],
                        binding_id=state["binding_id"],
                        study_revision=state["study_revision"],
                        stale=state["study_checkpoint_sha256"] != result["checkpoint_sha256"],
                    )
                )
            except (ValueError, KeyError, OSError, TypeError):
                continue
        result.update(
            results_enabled=True,
            results_source=bundle.source_identity(),
            result_evaluations=rows,
            result_list_truncated=len(ids) > 64,
        )
        return result

    def compare(self, folder, value):
        keys(value, ("evaluation_ids",))
        with tempfile.TemporaryDirectory(prefix="atomic-results-ui-") as tmp:
            bundle.export(folder, value["evaluation_ids"], Path(tmp) / "results")
            # export already verifies this newly created private snapshot,
            # including native reintegration; do not repeat it for rendering.
            return read(Path(tmp) / "results/summary.json")

    def export_results(self, folder, value, output):
        keys(value, ("evaluation_ids",))
        return bundle.export(folder, value["evaluation_ids"], output)
