"""Read cached numeric OOXML values without executing macros or external links."""

import csv
import gzip
import hashlib
import io
import math
import re
import zipfile
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HEADERS = [
    "RUN",
    "Coolant",
    "Total length",
    "d_i",
    "l/d_i",
    "T_b,in",
    "T_u,in",
    "T_wi,n",
    "T_wi,n-T_sat,n",
    "T_b,n",
    "p_n",
    "h_b,n",
    "X",
    "q_w,n",
    "G",
]
UNITS = [
    "#",
    "#",
    "m",
    "m",
    "#",
    "°C",
    "K",
    "°C",
    "K",
    "°C",
    "bar",
    "J/kg",
    "#",
    "W/m^2",
    "kg/m^2/s",
]
FIELDS = [
    "run",
    "coolant",
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
BOUNDS = {
    "diameter_m": ("F", "G", "m"),
    "inlet_temperature_k": ("N", "O", "°C"),
    "bulk_temperature_k": ("R", "S", "°C"),
    "wall_temperature_k": ("V", "W", "°C"),
    "heat_flux_w_m2": ("Z", "AA", "W/m^2"),
    "pressure_pa": ("AD", "AE", "bar"),
    "bulk_enthalpy_j_kg": ("AH", "AI", "J/kg"),
    "mass_flux_kg_m2_s": ("AL", "AM", "kg/m^2/s"),
}
COLUMNS = [
    "row_id",
    "source_workbook",
    "source_row",
    *FIELDS,
    *[f"{k}_{side}" for k in BOUNDS for side in ("lower", "upper")],
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bounded_zip(data, *, max_bytes=32 * 1024**2):
    if len(data) > max_bytes:
        raise ValueError("Archive size limit")
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("Invalid OOXML/measurement ZIP") from exc
    names = set()
    size = 0
    for m in z.infolist():
        p = PurePosixPath(m.filename)
        if (
            m.filename in names
            or p.is_absolute()
            or ".." in p.parts
            or "\\" in m.filename
            or m.flag_bits & 1
            or (m.external_attr >> 16) & 0o170000 == 0o120000
        ):
            z.close()
            raise ValueError("Unsafe archive member")
        names.add(m.filename)
        size += m.file_size
        if m.file_size > 8 * 1024**2 or size > max_bytes or len(names) > 4096:
            z.close()
            raise ValueError("Archive expansion limit")
    return z


def xml(data):
    if b"\0" in data or b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise ValueError("XML entities are forbidden")
    return ET.fromstring(data)


def workbook(data):
    with bounded_zip(data) as z:
        strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            strings = [
                "".join(t.text or "" for t in item.findall(".//s:t", NS))
                for item in xml(z.read("xl/sharedStrings.xml")).findall("s:si", NS)
            ]
        sheets = []
        for name in ("xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"):
            rows = {}
            for row in xml(z.read(name)).findall(".//s:row", NS):
                number = int(row.attrib["r"])
                if number in rows:
                    raise ValueError("Duplicate worksheet row")
                cells = {}
                for cell in row.findall("s:c", NS):
                    address = cell.attrib["r"]
                    if (
                        not re.fullmatch(r"[A-Z]+[1-9][0-9]*", address)
                        or int(re.sub("[A-Z]", "", address)) != number
                    ):
                        raise ValueError("Invalid cell address")
                    col = re.sub("[0-9]", "", address)
                    if col in cells:
                        raise ValueError("Duplicate cell")
                    value = cell.find("s:v", NS)
                    if cell.find("s:f", NS) is not None:
                        raise ValueError(
                            "Formula cells require independent evaluation; cached formula refused"
                        )
                    if value is None:
                        text = cell.find("s:is", NS)
                        if text is not None:
                            cells[col] = "".join(text.itertext())
                        continue
                    text = value.text
                    cells[col] = strings[int(text)] if cell.get("t") == "s" else text
                if cells:
                    rows[number] = cells
            sheets.append(rows)
    return sheets


def finite(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError("Nonfinite observation")
    return x


def convert(value, unit):
    x = finite(value)
    return x + 273.15 if unit == "°C" else x * 1e5 if unit == "bar" else x


def extract(archive):
    """Return all source rows, with workbook hashes and exact source-cell lineage."""
    result = []
    sources = {}
    notes = []
    with bounded_zip(archive) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".xlsm"))
        if len(names) != 176:
            raise ValueError("Expected176 primary workbooks")
        for name in names:
            base = PurePosixPath(name).name
            if not re.fullmatch(r"Database Steady State Run_[0-9]{4}_m.xlsm", base):
                raise ValueError("Unexpected source workbook name")
            data = z.read(name)
            sources[base] = sha(data)
            values, uncertainty = workbook(data)
            cols = [chr(65 + i) for i in range(15)]
            if [values[1].get(c) for c in cols] != HEADERS or [
                values[2].get(c) for c in cols
            ] != UNITS:
                raise ValueError("Source columns or units changed")
            data_rows = sorted(r for r in values if r > 2)
            if len(data_rows) != 70 or data_rows != list(range(3, 73)):
                raise ValueError("Expected70 source measurement segments")
            if set(uncertainty) != set(values):
                raise ValueError("Unpaired uncertainty rows")
            for r in data_rows:
                cells = values[r]
                bounds = uncertainty[r]
                row = {
                    field: (cells[col] if field == "coolant" else convert(cells[col], unit))
                    for field, col, unit in zip(FIELDS, cols, UNITS)
                }
                run = int(row["run"])
                if (
                    row["run"] != run
                    or f"{run:04}" not in base
                    or cells["B"] != "Water"
                    or bounds["A"] != cells["A"]
                    or bounds["B"] != "Water"
                ):
                    raise ValueError("Source run/coolant mismatch")
                row["run"] = run
                if (
                    row["length_m"] <= 0
                    or row["diameter_m"] <= 0
                    or row["pressure_pa"] <= 0
                    or row["mass_flux_kg_m2_s"] <= 0
                ):
                    raise ValueError("Nonpositive physical source coordinate")
                for field, (lo, hi, unit) in BOUNDS.items():
                    if uncertainty[2].get(lo) != unit or uncertainty[2].get(hi) != unit:
                        raise ValueError("Uncertainty unit mismatch")
                    left, right = convert(bounds[lo], unit), convert(bounds[hi], unit)
                    if not left <= row[field] <= right:
                        raise ValueError("Uncertainty interval does not enclose nominal")
                    row[field + "_lower"] = left
                    row[field + "_upper"] = right
                row.update(row_id=f"{run:04}:{r:02}", source_workbook=base, source_row=r)
                result.append(row)
    if len({r["row_id"] for r in result}) != len(result):
        raise ValueError("Duplicate source identity")
    return result, sources, notes


def encode(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return gzip.compress(out.getvalue().encode(), mtime=0)
