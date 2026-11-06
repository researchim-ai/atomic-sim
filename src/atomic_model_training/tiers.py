"""Mandatory actual-core reference replay for an explicitly bounded tier claim."""

from copy import deepcopy

import numpy as np

from .episode import run_episode
from .spec import admit


def control_replay(case, *, state_limit=1e-3, balance_limit=1e-5):
    case = admit(case)
    reference = deepcopy(case)
    reference["tier"] = "reference"
    training = run_episode(case)
    truth = run_episode(reference)
    error = float(
        np.max(
            np.abs(np.array(training["state"]) - np.array(truth["state"]))
            / np.maximum(1.0, np.abs(truth["state"]))
        )
    )
    balances = max(
        r["balances"][key]
        for r in (training, truth)
        for key in ("energy_relative", "mass_relative", "boron_relative")
    )
    events_equal = all(
        all(a[key] == b[key] for key in ("events", "protection_event_s"))
        for a, b in zip(
            training["checkpoint"]["payload"]["history"], truth["checkpoint"]["payload"]["history"]
        )
    )
    return dict(
        fidelity="reduced_single_loop_core",
        surrogate=False,
        state_scaled_error=error,
        balance_relative=balances,
        events_equal=events_equal,
        accepted=error <= state_limit and balances <= balance_limit and events_equal,
        training=training,
        reference=truth,
        validated=False,
        plant_referenced=False,
    )
