"""Independent, read-only primary XLSM verification; optional openpyxl dependency."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

import openpyxl


def main():
    p = argparse.ArgumentParser(
        description="Independent primary HIPER reader; no production importer used"
    )
    p.add_argument("archive", type=Path)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    manifest = json.loads((a.dataset / "manifest.json").read_text())
    if a.archive.stat().st_size > 32 * 1024**2:
        raise ValueError("Archive limit")
    z = zipfile.ZipFile(a.archive)
    if sum(m.file_size for m in z.infolist()) > 32 * 1024**2:
        raise ValueError("Archive expansion limit")
    rows = {
        r["row_id"]: r
        for r in csv.DictReader(
            io.StringIO(gzip.decompress((a.dataset / "observations.csv.gz").read_bytes()).decode())
        )
    }
    fields = [
        "length_m",
        "diameter_m",
        "length_over_diameter",
        "inlet_temperature_k",
        "inlet_subcooling_k",
        "wall_temperature_k",
        "wall_superheat_k",
        "bulk_temperature_k",
        "pressure_pa",
        "bulk_enthalpy_j_kg",
        "quality",
        "heat_flux_w_m2",
        "mass_flux_kg_m2_s",
    ]
    columns = {
        "diameter_m": (6, 7),
        "inlet_temperature_k": (14, 15),
        "bulk_temperature_k": (18, 19),
        "wall_temperature_k": (22, 23),
        "heat_flux_w_m2": (26, 27),
        "pressure_pa": (30, 31),
        "bulk_enthalpy_j_kg": (34, 35),
        "mass_flux_kg_m2_s": (38, 39),
    }

    def cv(k, x):
        return (
            float(x) + 273.15
            if k.endswith("temperature_k")
            else float(x) * 1e5
            if k == "pressure_pa"
            else float(x)
        )

    seen = set()
    checks = 0
    for name in sorted(n for n in z.namelist() if n.endswith(".xlsm")):
        raw = z.read(name)
        if hashlib.sha256(raw).hexdigest() != manifest["workbooks"][Path(name).name]:
            raise ValueError("Primary workbook hash mismatch")
        w = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        values = list(w.worksheets[0].values)[2:]
        u = list(w.worksheets[1].values)[2:]
        if len(values) != 70 or len(u) != 70:
            raise ValueError("Segment count mismatch")
        for i, (v, b) in enumerate(zip(values, u), 3):
            key = f"{int(v[0]):04}:{i:02}"
            r = rows[key]
            if key in seen:
                raise ValueError("Duplicate observation identity")
            seen.add(key)
            for k, x in zip(fields, v[2:15]):
                if cv(k, x) != float(r[k]):
                    raise ValueError((key, k, "Nominal value mismatch"))
                checks += 1
            for k, (lo, hi) in columns.items():
                for side, c in [("lower", lo), ("upper", hi)]:
                    if cv(k, b[c - 1]) != float(r[k + "_" + side]):
                        raise ValueError((key, k, side, "Uncertainty value mismatch"))
                    checks += 1
        w.close()
    if set(rows) != seen or len(seen) != 12320:
        raise ValueError("Observation identity/cardinality mismatch")
    result = dict(
        schema_version=1,
        reader="openpyxl",
        reader_version=openpyxl.__version__,
        raw_workbooks=176,
        rows=len(seen),
        numeric_values_checked=checks,
        exact_float_matches=True,
        macros_executed=False,
        source_archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),
        independence="Independent OOXML parser and extraction loop, same published source; not independent experiment",
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
