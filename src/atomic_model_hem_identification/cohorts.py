"""Two source-curated operating series; no prospective holdout claim."""

from atomic_model_workspace.common import digest

README_SHA = "d0e5f5e32b6f955c1887968207b020a341e8179fb8a340d57cf8c2ee7c9822e8"
GROUPS = dict(
    development=dict(run_ids=list(range(215, 224)), inlet_temperature_sanity_k=[600.0, 610.0]),
    evaluation=dict(run_ids=list(range(258, 270)), inlet_temperature_sanity_k=[580.0, 595.0]),
)
PROVENANCE = dict(
    source="https://mediatum.ub.tum.de/1839312",
    readme_pdf_sha256=README_SHA,
    source_location="Table3.1, PDF page12 / printed8, first two operating series",
    interpretation="Previously inspected data, retrospective operating-series separation, same facility; not independent campaign validation",
    known_source_ambiguity="README repeats run258 later at20.9MPa; only first two15.4..15.5MPa series used, checked against pinned workbook conditions",
)


def split(rows, manifest):
    if manifest["readme_pdf_sha256"] != README_SHA:
        raise ValueError("Operating-series source changed")
    if not rows:
        raise ValueError("No eligible physical rows")
    groups = {k: [] for k in GROUPS}
    seen = set()
    for row in rows:
        if row["row_id"] in seen:
            raise ValueError("Duplicate physical observation")
        seen.add(row["row_id"])
        matching = [name for name, value in GROUPS.items() if row["run"] in value["run_ids"]]
        if len(matching) != 1:
            raise ValueError("Uncurated/overlapping physical series")
        name = matching[0]
        lo, hi = GROUPS[name]["inlet_temperature_sanity_k"]
        if not 15e6 <= row["pressure_pa"] <= 16e6 or not lo <= row["inlet_temperature_k"] <= hi:
            raise ValueError("Run identity conflicts with physical operating series")
        groups[name].append(row)
    a = {r["run"] for r in groups["development"]}
    b = {r["run"] for r in groups["evaluation"]}
    if a & b or not a or not b:
        raise ValueError("Physical series leak or empty cohort")
    return groups, dict(
        provenance=PROVENANCE,
        groups=GROUPS,
        row_binding_sha256=digest(
            {name: [r["row_id"] for r in values] for name, values in groups.items()}
        ),
        runs_disjoint=True,
        operating_series_disjoint=True,
        independent_facility=False,
        prospective_holdout=False,
        fitting_performed=False,
    )
