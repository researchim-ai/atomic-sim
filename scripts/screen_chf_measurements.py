"""Offline screening of pinned, licensed KAERI uniform CHF measurements.

Domain screening is not experimental validation. No fitting or domain expansion.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

from atomic_model_hem_qualified import closures
from atomic_model_multiphysics.provenance import seal, sha
from atomic_model_training.spec import atomic_json, read_json

RECORD = 23566164
DOI = "10.5281/zenodo.23566164"
PINS = {
    "uniform_standardized.xml": "e87c2e0a7324408f77076554ae44291c7aee06034d933f7a3113d7a0c50cb281",
    "README_uniform.md": "13c44bcf7e90b077175ba19c09fc10ea5fca823cb47db3b026a65e8a1fd723ef",
}
FIELDS = ("Diameter", "Length", "Pressure", "MassFlux", "XOUT", "HeatFlux", "Power", "Perimeter", "LocalCriticalHeatFlux")


def reasons(row, model):
    if model not in ("biasi_inl", "table1995"):
        raise ValueError("Explicit CHF model required")
    p, G, x, D, L = (row[k] for k in ("Pressure", "MassFlux", "XOUT", "Diameter", "Length"))
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (p, G, x, D, L)):
        raise ValueError("Nonfinite screening coordinate")
    result = []
    if model == "biasi_inl":
        for key, value, lower, upper in [("pressure", p, .27e6, 14e6), ("mass_flux", G, 100, 600),
                                         ("quality", x, 0, .1), ("diameter", D, .005, .0375), ("length", L, .2, 6)]:
            if not lower <= value <= upper:
                result.append(key)
    else:
        if D != .008:
            result.append("exact_8mm_diameter")
        low = 3e5 <= p <= 5e5 and 1000 <= G <= 1500 and 0 <= x <= .05
        high = 15e6 <= p <= 16e6 and 1000 <= G <= 2000 and 0 <= x <= .1
        if not (low or high):
            result.append("disconnected_pressure_massflux_quality_box")
    # The runtime law is authoritative even inside the numeric box.
    if not result:
        try:
            if model == "biasi_inl":
                closures.biasi_chf(p, G, x, D, L)
            else:
                closures.table_chf(p, G, x, D)
        except ValueError:
            result.append("runtime_law_refusal")
    return result


def read_pinned(xml, readme, record):
    xml, readme, record = Path(xml), Path(readme), Path(record)
    for path, name in [(xml, "uniform_standardized.xml"), (readme, "README_uniform.md")]:
        if path.stat().st_size > 2_000_000 or sha(path) != PINS[name]:
            raise ValueError("Primary file checksum/size differs: " + name)
    meta = read_json(record)
    if meta.get("id") != RECORD or meta["metadata"].get("doi") != DOI or meta["metadata"].get("license", {}).get("id") != "cc-by-4.0" or meta["metadata"].get("access_right") != "open":
        raise ValueError("Dataset identity or redistribution license differs")
    raw = xml.read_bytes()
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("XML declarations not admitted")
    root = ElementTree.fromstring(raw)
    if root.tag != "struct" or len(root) != 651 or any(d.tag != "dataset" for d in root):
        raise ValueError("Pinned dataset structure/count differs")
    rows = []
    for dataset in root:
        def scalar(key):
            values = dataset.findall(key)
            if len(values) != 1 or values[0].text is None:
                raise ValueError("Missing/duplicate scalar: " + key)
            return values[0].text
        row = dict(TestID=int(scalar("TestID")), Source=scalar("Source"), Provenance=scalar("Provenance"))
        for field in FIELDS:
            value = float(scalar(field))
            if not math.isfinite(value):
                if field != "LocalCriticalHeatFlux" or not math.isnan(value):
                    raise ValueError("Nonfinite measurement field: " + field)
                value = None
            row[field] = value
        if scalar("Fluid") != "water" or scalar("HeatFluxDefinition") != "power_per_heated_area":
            raise ValueError("Unexpected fluid or heat-flux definition")
        if any(row[k] <= 0 for k in ("Diameter", "Length", "Pressure", "MassFlux", "HeatFlux", "Power", "Perimeter")):
            raise ValueError("Nonpositive physical input")
        row["heat_flux_algebra_absolute_w_m2"] = abs(row["HeatFlux"] - row["Power"] / (row["Perimeter"] * row["Length"]))
        rows.append(row)
    if len({r["TestID"] for r in rows}) != len(rows):
        raise ValueError("Duplicate TestID")
    return rows


def screen(rows):
    records = []
    for row in rows:
        record = dict(row)
        record["numeric_domain_exclusions"] = {model: reasons(row, model) for model in ("biasi_inl", "table1995")}
        records.append(record)
    models = {}
    for model in ("biasi_inl", "table1995"):
        candidates = [r["TestID"] for r in records if not r["numeric_domain_exclusions"][model]]
        counts = Counter(reason for r in records for reason in r["numeric_domain_exclusions"][model])
        models[model] = dict(numeric_domain_candidates=len(candidates), candidate_ids=candidates,
                             nonexclusive_exclusion_counts=dict(counts),
                             experimental_validation_admitted=False,
                             further_requirements=["Primary geometry/orientation and CHF-location audit", "Measurement uncertainties and common-input covariance", "Source-family independence from correlation calibration"])
    return dict(rows=records, models=models, source_family_counts=dict(Counter(r["Source"] for r in records)),
                unresolved_local_chf_count=sum(r["LocalCriticalHeatFlux"] is None for r in records),
                total_rows=len(records), screening_completed=True, experimental_validation_admitted=False,
                predictions_fitted=False, validated=False, plant_referenced=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = screen(read_pinned(args.xml, args.readme, args.record))
    report.update(dataset_doi=DOI, license="CC-BY-4.0", record_sha256=sha(args.record),
                  files=PINS, driver_sha256=sha(__file__), closure_source_sha256=sha(closures.__file__),
                  source="https://zenodo.org/records/23566164",
                  attribution="Bourisaw et al., KAERI/TR-1665/2000 digitized/corrected dataset v3; original experimental sources retained per row",
                  scope="Numeric-domain census of published processed measurements; no new experimental uncertainty or validation claim")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, seal(report))
    print(json.dumps(dict(screened=report["total_rows"], candidates={k: v["numeric_domain_candidates"] for k, v in report["models"].items()}, experimental_validation_admitted=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
