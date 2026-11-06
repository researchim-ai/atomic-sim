"""Actual OpenMC subprocess entry: retained XML, statepoint and explicit MGXS axes."""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from atomic_model_training.spec import atomic_json, read_json
from data.catalog import assert_nuclear_data_allowed

from .provenance import seal, sha

SCORES = [
    "absorption",
    "nu-fission",
    "fission",
    "scatter matrix",
    "chi",
    "inverse-velocity",
    "diffusion-coefficient",
    "kappa-fission",
]


def execute(folder, specification):
    import resource

    from .transport import admit

    specification = admit(specification)
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (600, 600))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128 * 1024**2, 128 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import openmc

    folder = Path(folder).absolute()
    xs = Path(os.environ["OPENMC_CROSS_SECTIONS"]).resolve()
    assert_nuclear_data_allowed(str(xs))
    root = ET.parse(xs).getroot()
    data = {}
    for name in specification["nuclides"]:
        matches = [
            node for node in root.findall("library") if name in node.get("materials", "").split()
        ]
        if len(matches) != 1:
            raise ValueError("Ambiguous or missing evaluated nuclide " + name)
        p = Path(matches[0].get("path"))
        p = p if p.is_absolute() else xs.parent / p
        data[name] = dict(sha256=sha(p), filename=p.name)
    openmc.reset_auto_ids()
    material = openmc.Material(name="declared-homogeneous-research-mixture")
    material.set_density("g/cm3", specification["density_kg_m3"] / 1000)
    material.temperature = specification["temperature_k"]
    for name, fraction in specification["nuclides"].items():
        material.add_nuclide(name, fraction)
    materials = openmc.Materials([material])
    materials.cross_sections = str(xs)
    width = specification["width_m"] * 100
    left = openmc.XPlane(x0=0, boundary_type="reflective")
    right = openmc.XPlane(x0=width, boundary_type="reflective")
    ym = openmc.YPlane(y0=-1, boundary_type="reflective")
    yp = openmc.YPlane(y0=1, boundary_type="reflective")
    zm = openmc.ZPlane(z0=-1, boundary_type="reflective")
    zp = openmc.ZPlane(z0=1, boundary_type="reflective")
    cell = openmc.Cell(fill=material, region=+left & -right & +ym & -yp & +zm & -zp)
    geometry = openmc.Geometry([cell])
    settings = openmc.Settings()
    settings.batches = specification["batches"]
    settings.inactive = specification["inactive"]
    settings.particles = specification["particles"]
    settings.seed = specification["seed"]
    settings.temperature = {"method": "interpolation", "range": (250.0, 1200.0)}
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box((1e-5, -0.9, -0.9), (width - 1e-5, 0.9, 0.9))
    )
    settings.output = {"summary": True, "tallies": False}
    settings.statepoint = {"batches": [settings.batches]}
    groups = openmc.mgxs.EnergyGroups(specification["energy_bounds_ev_ascending"])
    library = openmc.mgxs.Library(geometry)
    library.energy_groups = groups
    library.domain_type = "material"
    library.domains = [material]
    library.mgxs_types = SCORES
    library.legendre_order = 0
    library.build_library()
    tallies = openmc.Tallies()
    library.add_to_tallies_file(tallies)
    model = openmc.Model(geometry, materials, settings, tallies)
    model.export_to_xml(directory=folder)
    # OpenMC itself is the child of this worker. Supervisor bounds the process group.
    statepoint = model.run(cwd=folder, threads=1)
    means = {}
    standard_errors = {}
    with openmc.StatePoint(statepoint) as sp:
        library.load_from_statepoint(sp)
        k = sp.keff
        for score in SCORES:
            mg = library.get_mgxs(material, score)
            options = dict(order_groups="increasing")
            if score == "scatter matrix":
                options.update(row_column="inout", moment=0)
            means[score] = np.asarray(mg.get_xs(value="mean", **options)).tolist()
            standard_errors[score] = np.asarray(mg.get_xs(value="std_dev", **options)).tolist()
        result = dict(
            source_family="evaluated",
            producer_worker_sha256=sha(__file__),
            k_mean=float(k.n),
            k_std=float(k.s),
            means=means,
            standard_errors=standard_errors,
            openmc_version=openmc.__version__,
            specification=specification,
            nuclear_data=dict(
                library="ENDF/B-VIII.0",
                cross_sections_sha256=sha(xs),
                license=read_json(str(xs) + ".license.json"),
                nuclides=data,
            ),
            axes=dict(
                groups="fast to thermal, OpenMC group indices increasing",
                scatter="incoming,outgoing",
                macro_unit="1/cm",
                diffusion_unit="cm",
                inverse_velocity_unit="s/cm",
                kappa_fission_unit="eV/cm",
            ),
            retained={
                p.name: sha(p) for p in sorted(folder.iterdir()) if p.suffix in (".xml", ".h5")
            },
            validated=False,
            plant_referenced=False,
        )
    atomic_json(folder / "mgxs.json", seal(result))
    return result


if __name__ == "__main__":
    import sys

    execute(sys.argv[1], read_json(sys.argv[2]))
