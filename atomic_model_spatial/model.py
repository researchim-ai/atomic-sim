"""Cell-centred slab FV, SI units; scatter[to, from, cell]."""

import numpy as np
from scipy import sparse

FIELDS = {
    "faces_m",
    "diffusion_m",
    "absorption_per_m",
    "nu_fission_per_m",
    "scatter_per_m",
    "speed_m_s",
    "chi_prompt",
    "chi_delayed",
    "beta",
    "decay_per_s",
    "boundaries",
    "provenance",
}


def array(value, shape, name, positive=False):
    if np.iscomplexobj(value):
        raise ValueError(f"Complex {name} is unsupported")
    a = np.array(value, dtype=float, copy=True)
    if a.shape != shape or not np.isfinite(a).all() or np.any(a <= 0 if positive else a < 0):
        raise ValueError(f"Invalid {name}: shape {shape}, finite and nonnegative required")
    a.setflags(write=False)
    return a


class Model:
    """Owned immutable arrays; no broadcasting, feedback or microscopic-data inference."""

    def __setattr__(self, key, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Model is immutable; construct a new configuration")
        object.__setattr__(self, key, value)

    def __init__(self, config):
        if not isinstance(config, dict) or set(config) != FIELDS:
            raise ValueError("Model requires exact version-1 fields")
        faces = np.array(config["faces_m"], dtype=float, copy=True)
        if (
            faces.ndim != 1
            or not 2 <= len(faces) <= 513
            or not np.isfinite(faces).all()
            or np.any(np.diff(faces) <= 0)
        ):
            raise ValueError("Faces must be finite, strictly increasing, 1..512 cells")
        self.n = len(faces) - 1
        raw = np.asarray(config["diffusion_m"])
        if raw.ndim != 2 or not 1 <= raw.shape[0] <= 8:
            raise ValueError("Require 1..8 energy groups")
        self.g = raw.shape[0]
        decay = np.asarray(config["decay_per_s"])
        if decay.ndim != 1 or not 1 <= len(decay) <= 8:
            raise ValueError("Require 1..8 local precursor families")
        self.j = len(decay)
        if (self.g + self.j) * self.n > 4096:
            raise ValueError("State budget exceeded")
        self.faces = faces
        self.faces.setflags(write=False)
        self.width = np.diff(faces)
        self.width.setflags(write=False)
        for name in [
            "diffusion_m",
            "absorption_per_m",
            "nu_fission_per_m",
            "speed_m_s",
            "chi_prompt",
            "chi_delayed",
        ]:
            setattr(
                self,
                name,
                array(config[name], (self.g, self.n), name, name in ("diffusion_m", "speed_m_s")),
            )
        self.scatter = array(config["scatter_per_m"], (self.g, self.g, self.n), "scatter")
        if np.any(self.scatter[np.arange(self.g), np.arange(self.g)] != 0):
            raise ValueError("Within-group scatter must be zero; absorption is separate")
        self.beta = array(config["beta"], (self.j, self.n), "beta")
        self.decay = array(decay, (self.j,), "decay", True)
        if np.any(self.beta.sum(axis=0) >= 1):
            raise ValueError("Total delayed fraction must be <1")
        for chi in [self.chi_prompt, self.chi_delayed]:
            if not np.allclose(chi.sum(axis=0), 1, rtol=0, atol=1e-12):
                raise ValueError("Spectra must sum to one in every cell")
        bc = config["boundaries"]
        if (
            not isinstance(bc, list)
            or len(bc) != 2
            or any(b not in ("reflecting", "zero_flux") for b in bc)
        ):
            raise ValueError("Boundaries: reflecting or zero_flux on the physical face")
        self.boundaries = tuple(bc)
        provenance = config["provenance"]
        if not isinstance(provenance, str) or not provenance.strip() or len(provenance) > 4096:
            raise ValueError("Explicit data provenance required")
        self.provenance = provenance
        self.loss, self.fission, self.kinetics, self.leakage = self._assemble()
        self.leakage.setflags(write=False)
        for operator in (self.loss, self.fission, self.kinetics):
            for buffer in (operator.data, operator.indices, operator.indptr):
                buffer.setflags(write=False)
        self._sealed = True

    def config(self):
        return dict(
            faces_m=self.faces.tolist(),
            **{
                name: getattr(self, name).tolist()
                for name in [
                    "diffusion_m",
                    "absorption_per_m",
                    "nu_fission_per_m",
                    "speed_m_s",
                    "chi_prompt",
                    "chi_delayed",
                ]
            },
            scatter_per_m=self.scatter.tolist(),
            beta=self.beta.tolist(),
            decay_per_s=self.decay.tolist(),
            boundaries=list(self.boundaries),
            provenance=self.provenance,
        )

    def _assemble(self):
        n, g, j = self.n, self.g, self.j
        size = n * g
        loss = sparse.lil_matrix((size, size))
        fission = sparse.lil_matrix((size, size))
        prompt = sparse.lil_matrix((size, size))
        leakage = np.zeros((g, n))
        b = self.beta.sum(axis=0)
        for h in range(g):
            for c in range(n):
                k = h * n + c
                loss[k, k] += (
                    self.absorption_per_m[h, c] + self.scatter[:, h, c].sum()
                ) * self.width[c]
                for to in range(g):
                    loss[to * n + c, k] -= self.scatter[to, h, c] * self.width[c]
                    fission[to * n + c, k] = (
                        ((1 - b[c]) * self.chi_prompt[to, c] + b[c] * self.chi_delayed[to, c])
                        * self.nu_fission_per_m[h, c]
                        * self.width[c]
                    )
                    prompt[to * n + c, k] = (
                        (1 - b[c])
                        * self.chi_prompt[to, c]
                        * self.nu_fission_per_m[h, c]
                        * self.width[c]
                    )
            for c in range(n - 1):
                conductance = 1 / (
                    self.width[c] / (2 * self.diffusion_m[h, c])
                    + self.width[c + 1] / (2 * self.diffusion_m[h, c + 1])
                )
                left, right = h * n + c, h * n + c + 1
                loss[left, left] += conductance
                loss[right, right] += conductance
                loss[left, right] -= conductance
                loss[right, left] -= conductance
            for side, c in [(0, 0), (1, n - 1)]:
                if self.boundaries[side] == "zero_flux":
                    value = 2 * self.diffusion_m[h, c] / self.width[c]
                    loss[h * n + c, h * n + c] += value
                    leakage[h, c] += value
        emission = sparse.lil_matrix((size, j * n))
        production = sparse.lil_matrix((j * n, size))
        for family in range(j):
            for c in range(n):
                for h in range(g):
                    emission[h * n + c, family * n + c] = (
                        self.chi_delayed[h, c] * self.decay[family] * self.speed_m_s[h, c]
                    )
                    production[family * n + c, h * n + c] = (
                        self.beta[family, c] * self.nu_fission_per_m[h, c]
                    )
        scale = sparse.diags((self.speed_m_s / self.width).ravel())
        kinetics = sparse.bmat(
            [
                [scale @ (prompt - loss), emission],
                [production, -sparse.diags(np.repeat(self.decay, n))],
            ],
            format="csc",
        )
        if not np.isfinite(kinetics.data).all():
            raise ValueError("Operator overflow")
        return loss.tocsc(), fission.tocsc(), kinetics, leakage
