"""Audited publication rights are distinct from raw-data rights and model applicability."""

import hashlib
from importlib.resources import files

from atomic_model_training.spec import exact


def admit(candidate):
    exact(
        candidate,
        {
            "id",
            "publication",
            "raw_data",
            "configuration_complete",
            "observables",
            "uncertainty_semantics",
            "model_applicability",
            "calibration_case",
            "validation_cases",
        },
    )
    exact(candidate["publication"], {"url", "license", "sha256"})
    exact(
        candidate["raw_data"],
        {"available", "redistribution_allowed", "clock_documented", "artifact_sha256"},
    )
    if not isinstance(candidate["id"], str) or not candidate["id"]:
        raise ValueError("Missing candidate id")
    if type(candidate["configuration_complete"]) is not bool:
        raise ValueError("Configuration completeness must be explicit")
    for key in ["available", "redistribution_allowed", "clock_documented"]:
        if type(candidate["raw_data"][key]) is not bool:
            raise ValueError("Data admission flags must be explicit booleans")
    if set(candidate["validation_cases"]) & {candidate["calibration_case"]}:
        raise ValueError("Experimental calibration/validation leakage")
    reasons = []
    raw = candidate["raw_data"]
    for key, reason in [
        ("available", "raw_measurements_unavailable"),
        ("redistribution_allowed", "raw_data_rights_unconfirmed"),
        ("clock_documented", "native_clock_unconfirmed"),
    ]:
        if not raw[key]:
            reasons.append(reason)
    if not raw["artifact_sha256"]:
        reasons.append("raw_measurement_checksum_absent")
    if not candidate["configuration_complete"]:
        reasons.append("experiment_configuration_incomplete")
    if candidate["model_applicability"] != "admitted":
        reasons.append("model_applicability_not_admitted")
    if candidate["uncertainty_semantics"] != "complete_declared_budget":
        reasons.append("experimental_uncertainty_incomplete")
    if not candidate["observables"] or not candidate["validation_cases"]:
        reasons.append("heldout_observables_absent")
    return dict(
        id=candidate["id"],
        admitted=False,
        metadata_requirements_met=not reasons,
        raw_integrity_verified=False,
        admission_semantics="metadata prerequisites only; raw artifacts and experimental applicability require independent verification",
        reasons=reasons,
        validated=False,
        plant_referenced=False,
    )


def motel_candidate():
    root = files("atomic_model_acceptance").joinpath("data/motel_2025")
    raw = root.joinpath("source.pdf").read_bytes()
    return dict(
        id="motel-ned-2025-114056",
        publication=dict(
            url="https://doi.org/10.1016/j.nucengdes.2025.114056",
            license="CC-BY-4.0",
            sha256=hashlib.sha256(raw).hexdigest(),
        ),
        raw_data=dict(
            available=False,
            redistribution_allowed=False,
            clock_documented=False,
            artifact_sha256=None,
        ),
        configuration_complete=False,
        observables=["core fluid temperature", "downcomer mass flow"],
        uncertainty_semantics="published temperature bound +/-2 K; no complete raw joint covariance",
        model_applicability="not_admitted_multidimensional_crossflow_and_helical_SG",
        calibration_case=None,
        validation_cases=[],
    )
