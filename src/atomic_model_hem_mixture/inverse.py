"""Deterministic seed retries; every candidate passes unchanged strict IF97 checks."""

from atomic_model_hem import eos


def conserved(density, internal_energy, pressure_seed, temperature_seed):
    last = None
    for offset in [0.0, 0.01, -0.01, 1.0, -1.0]:
        temperature = temperature_seed + offset
        if not eos.T_MIN <= temperature <= eos.T_MAX:
            continue
        try:
            return eos.rho_u(
                density,
                internal_energy,
                pressure_seed_pa=pressure_seed,
                temperature_seed_k=temperature,
            )
        except ValueError as error:
            last = error
    raise ValueError("All strict conserved EOS seeds failed; no projection applied") from last
