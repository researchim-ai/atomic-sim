"""Frozen NIST copper-pipe experimental screen of the existing Darcy/Haaland closure."""

import csv
import hashlib
import io
import math
import shutil
from pathlib import Path

from atomic_model_validation import digitize
from atomic_model_workspace.common import atomic_json, digest, local_file, read
from physics.hydraulics import closures

DATA = Path(__file__).parent / "data/nist_tn2294"
MANIFEST_SHA256 = "0d9cf1255ba5336b29b0c933fbaecce695125782cd402ec1045858f9b2dd64c4"


def sha(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(16 * 1024**2 + 1)
    if len(raw) > 16 * 1024**2:
        raise ValueError("Experimental artifact exceeds 16 MiB")
    return hashlib.sha256(raw).hexdigest()


def dataset(folder=DATA):
    folder = Path(folder)
    if sha(local_file(folder, "manifest.json")) != MANIFEST_SHA256:
        raise ValueError("Frozen experimental manifest changed")
    manifest = read(folder / "manifest.json")
    for name, expected in manifest["artifacts"].items():
        if sha(local_file(folder, name)) != expected:
            raise ValueError("Experimental source changed: " + name)
    vectors = read(folder / "vectors.json")
    rows = digitize.rows(vectors)
    saved = list(csv.DictReader(io.StringIO((folder / "measurements.csv").read_text())))
    normalized = [
        dict(series=r["series"], **{k: float(v) for k, v in r.items() if k != "series"})
        for r in saved
    ]
    if rows != normalized:
        raise ValueError("Digitized CSV does not match frozen vector symbols")
    if len(rows) != 38 or vectors["counts"] != dict(test_1=14, test_2=9, test_3=9, test_4=6):
        raise ValueError("Experimental series inventory changed")
    return read(folder / "protocol.json"), rows


def source_identity():
    return dict(
        validation_sha256=digest(
            {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        ),
        hydraulics_sha256=digest(
            {p.name: sha(p) for p in sorted(Path(closures.__file__).parent.glob("*.py"))}
        ),
        dataset_manifest_sha256=MANIFEST_SHA256,
    )


def native_factor(reynolds, protocol):
    # Synthetic normalization only: f is dimensionless. Do not claim measured Pa/m.
    rho, mu = 998.0, 0.001
    d = protocol["diameter_m"]
    flow = reynolds * math.pi * d * mu / 4
    dp = closures.friction_darcy(flow, d, mu, protocol["roughness_m"], 1.0, rho)
    velocity = flow / (rho * math.pi * d * d / 4)
    f = dp * 2 * d / (rho * velocity**2)
    direct = closures.haaland_friction_factor(reynolds, protocol["roughness_m"], d)
    if not math.isclose(f, direct, rel_tol=2e-14):
        raise ValueError("Darcy integration and direct correlation disagree")
    if not math.isfinite(f) or f <= 0:
        raise ValueError("Invalid native friction factor")
    return f


def evaluate(folder=DATA):
    protocol, rows = dataset(folder)
    comparisons = []
    for row in rows:
        re, observed = row["reynolds"], row["darcy_f"]
        modeled = native_factor(re, protocol)
        dx, dy = row["reynolds_reading_bound"], row["darcy_f_reading_bound"]
        # Haaland is monotone decreasing in this fixed-roughness Reynolds interval.
        lower = native_factor(re + dx, protocol)
        upper = native_factor(re - dx, protocol)
        mismatch = max(0.0, lower - (observed + dy), observed - dy - upper)
        comparisons.append(
            dict(
                row,
                modeled_darcy_f=modeled,
                relative_error=(modeled - observed) / observed,
                modeled_reading_interval=[lower, upper],
                reading_envelope_gap=mismatch,
                center_passed=abs(modeled - observed) / observed <= protocol["max_relative_error"],
            )
        )
    series = {}
    for name in sorted({r["series"] for r in comparisons}):
        selected = [r for r in comparisons if r["series"] == name]
        series[name] = dict(
            samples=len(selected),
            max_abs_relative_error=max(abs(r["relative_error"]) for r in selected),
            mean_relative_error=sum(r["relative_error"] for r in selected) / len(selected),
            rms_relative_error=math.sqrt(
                sum(r["relative_error"] ** 2 for r in selected) / len(selected)
            ),
            passed=all(r["center_passed"] for r in selected),
        )
    return dict(
        schema_version=1,
        protocol=protocol,
        source=source_identity(),
        quantity="Darcy friction factor",
        units="1",
        samples=len(rows),
        series=series,
        reynolds_range=[min(r["reynolds"] for r in rows), max(r["reynolds"] for r in rows)],
        max_abs_relative_error=max(abs(r["relative_error"]) for r in comparisons),
        passed=all(r["center_passed"] for r in comparisons),
        rows=comparisons,
        evidence_kind="digitized_external_experiment",
        calibrated=False,
        experimental_uncertainty_complete=False,
        independence_from_correlation_training="not_established",
        validated=False,
        plant_referenced=False,
    )


def plot(result, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, err) = plt.subplots(
        2,
        1,
        figsize=(9, 8),
        sharex=True,
        gridspec_kw=dict(height_ratios=[2, 1]),
        layout="constrained",
    )
    colors = ["black", "royalblue", "crimson", "forestgreen"]
    markers = ["^", "o", "d", "s"]
    for (name, _), color, marker in zip(result["series"].items(), colors, markers):
        rows = [r for r in result["rows"] if r["series"] == name]
        x = [r["reynolds"] for r in rows]
        ax.errorbar(
            x,
            [r["darcy_f"] for r in rows],
            xerr=[r["reynolds_reading_bound"] for r in rows],
            yerr=[r["darcy_f_reading_bound"] for r in rows],
            fmt=marker,
            color=color,
            label=name.replace("_", " "),
            capsize=2,
            markersize=4,
        )
        err.plot(x, [100 * r["relative_error"] for r in rows], marker, color=color, markersize=4)
    lo, hi = result["reynolds_range"]
    x = [lo + (hi - lo) * i / 150 for i in range(151)]
    ax.plot(
        x,
        [native_factor(v, result["protocol"]) for v in x],
        color="darkorange",
        label="Existing Darcy/Haaland closure",
    )
    ax.set_ylabel("Darcy friction factor [1]")
    ax.set_title("NIST TN 2294, Figure 18: digitized experiment vs existing model")
    ax.legend(fontsize=9)
    bound = 100 * result["protocol"]["max_relative_error"]
    err.axhline(bound, color="gray", linestyle="--")
    err.axhline(-bound, color="gray", linestyle="--")
    err.axhline(0, color="gray", linewidth=0.6)
    err.set_ylabel("Model minus observation [%]")
    err.set_xlabel("Reynolds number [1]")
    for axis in [ax, err]:
        axis.grid(alpha=0.25)
    fig.suptitle(
        "Reading bounds are symbol extents, not experimental uncertainty. No parameter fitting.",
        fontsize=10,
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output):
    output = Path(output)
    result = evaluate()
    output.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(DATA, output / "data")
        atomic_json(output / "report.json", result)
        plot(result, output / "comparison.png")
        paths = sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())
        atomic_json(
            output / "manifest.json",
            dict(
                schema_version=1,
                source=source_identity(),
                artifacts={name: sha(output / name) for name in paths},
            ),
        )
        verify(output)
    except BaseException:
        shutil.rmtree(output)
        raise
    return result


def verify(folder):
    folder = Path(folder)
    manifest = read(local_file(folder, "manifest.json"))
    expected = {"schema_version", "source", "artifacts"}
    if (
        set(manifest) != expected
        or manifest["schema_version"] != 1
        or manifest["source"] != source_identity()
    ):
        raise ValueError("Experimental result source/schema mismatch")
    files = {"data/" + p.name for p in DATA.iterdir()} | {"report.json", "comparison.png"}
    if set(manifest["artifacts"]) != files:
        raise ValueError("Experimental result inventory mismatch")
    actual = {
        p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() or p.is_symlink()
    }
    if actual != files | {"manifest.json"}:
        raise ValueError("Unexpected experimental result files")
    for name, expected_sha in manifest["artifacts"].items():
        if sha(local_file(folder, name)) != expected_sha:
            raise ValueError("Experimental result artifact changed")
    result = evaluate(folder / "data")
    if read(local_file(folder, "report.json")) != result:
        raise ValueError("Experimental comparison recomputation differs")
    return result
