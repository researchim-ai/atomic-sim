"""Reproducible vector-symbol extraction from NIST TN 2294, Figure 18 (PDF page 30).

pypdf is needed only to independently re-extract the distributed source PDF.
The frozen form coordinates and axis ticks have been visually checked against the page.
"""

from collections import Counter
from pathlib import Path

AXES = dict(x0=19.313, x1=138.35, re0=0, re1=120000, y0=9.2238, y1=96.2, f0=0.016, f1=0.030)


def extract(pdf):
    from pypdf import PdfReader
    from pypdf.generic import ContentStream

    reader = PdfReader(pdf)
    form = reader.pages[29]["/Resources"]["/XObject"]["/Meta301"].get_object()
    operations = ContentStream(form, reader).operations
    path = []
    color = None
    records = []
    for index, (args, op) in enumerate(operations):
        if op in (b"g", b"rg"):
            color = tuple(float(v) for v in args)
        if op in (b"m", b"l", b"c", b"re", b"h"):
            path.append((op, [float(v) for v in args]))
        if op not in (b"f", b"S", b"f*", b"n"):
            continue
        shape = tuple(p[0] for p in path)
        series = None
        if op == b"f":
            if color == (0.0,) and shape == (b"m", b"l", b"l", b"h"):
                series = "test_1"
            elif color == (0.0, 0.0, 1.0) and shape == (b"m",) + (b"c",) * 8 + (b"h",):
                series = "test_2"
            elif color == (1.0, 0.0, 0.0) and shape == (b"m", b"l", b"l", b"l", b"h"):
                series = "test_3"
            elif color == (0.0, 0.502, 0.0) and shape == (b"re",):
                series = "test_4"
        if series:
            if shape == (b"re",):
                x, y, w, h = path[0][1]
                xs, ys = [x, x + w], [y, y + h]
            else:
                xs = [v for _, a in path for v in a[::2]]
                ys = [v for _, a in path for v in a[1::2]]
            xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
            x, y = (xlo + xhi) / 2, (ylo + yhi) / 2
            # Exclude legend symbols by its visually checked rectangular boundary.
            legend = 101 <= x <= 137 and 74 <= y <= 95
            if 1 <= xhi - xlo <= 2 and 1 <= yhi - ylo <= 3 and not legend:
                if not (AXES["x0"] < x < AXES["x1"] and AXES["y0"] < y < AXES["y1"]):
                    raise ValueError("Symbol outside plot")
                records.append(
                    dict(
                        series=series,
                        operation_index=index,
                        bbox=[xlo, ylo, xhi, yhi],
                        center=[x, y],
                    )
                )
        path = []
    records.sort(key=lambda r: (r["series"], r["center"][0]))
    counts = Counter(r["series"] for r in records)
    if set(counts) != {"test_1", "test_2", "test_3", "test_4"}:
        raise ValueError("Missing repeat series")
    return dict(
        schema_version=1,
        pdf_page=30,
        printed_page=23,
        figure=18,
        form="/Meta301",
        axes=AXES,
        counts=dict(counts),
        symbols=records,
    )


def rows(vectors):
    a = vectors["axes"]
    sx = (a["re1"] - a["re0"]) / (a["x1"] - a["x0"])
    sy = (a["f1"] - a["f0"]) / (a["y1"] - a["y0"])
    result = []
    for symbol in vectors["symbols"]:
        x, y = symbol["center"]
        xlo, ylo, xhi, yhi = symbol["bbox"]
        result.append(
            dict(
                series=symbol["series"],
                reynolds=a["re0"] + (x - a["x0"]) * sx,
                darcy_f=a["f0"] + (y - a["y0"]) * sy,
                reynolds_reading_bound=(xhi - xlo + 0.02) * sx / 2,
                darcy_f_reading_bound=(yhi - ylo + 0.02) * sy / 2,
            )
        )
    return result


def main(argv=None):
    import argparse
    import json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pdf", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    a.output.write_text(json.dumps(extract(a.pdf), indent=2) + "\n")


if __name__ == "__main__":
    main()
