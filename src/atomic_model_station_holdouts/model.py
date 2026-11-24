"""Privileged audit derived from executed configurations, never family labels.

This proves disjoint declared factors, not statistical independence, a new
physical object, or generalization beyond the finite configurations evaluated.
"""
from dataclasses import asdict, replace
import hashlib
from pathlib import Path

from atomic_model_station_evaluation import validate_splits, evaluate
from atomic_model_station_evaluation.model import identity as backend_identity
from atomic_model_workspace.common import digest

PARAMETERS = ('equilibrium_start', 'inventory_deadband_m', 'film_model')
AXES = (*PARAMETERS, 'fault_mechanism', 'fault_magnitude', 'fault_onset_s')


def identity():
    return dict(backend=backend_identity(), files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


def _values(config, axis):
    if axis in PARAMETERS:
        if axis == 'inventory_deadband_m':
            if not config.automatic:
                raise ValueError('Inventory deadband holdout requires automatic control')
            value = float(config.inventory_deadband_m)
            return [0.0 if value == 0 else value]
        return [getattr(config, axis)]
    if not config.faults:
        raise ValueError('Fault holdout requires an exercised fault in every case')
    if any((f.kind in ('bias', 'drift') and f.value == 0) or
           (f.kind == 'cooling_loss' and f.value == 1) for f in config.faults):
        raise ValueError('No-effect fault cannot define a physical holdout')
    if axis == 'fault_mechanism':
        return sorted({(f.domain, f.target, f.kind) for f in config.faults})
    if axis == 'fault_magnitude':
        # Zero/unused magnitudes cannot establish an amplitude holdout.
        if any(f.kind not in ('bias', 'drift', 'drive_limit', 'cooling_loss') for f in config.faults):
            raise ValueError('Every fault must have an active magnitude parameter')
        return sorted({(f.domain, f.target, f.kind, 0.0 if f.value == 0 else float(f.value))
                       for f in config.faults})
    if axis == 'fault_onset_s':
        return sorted({f.start_tick * config.tick_s for f in config.faults})
    raise ValueError('Unknown physical factor')


def audit_splits(cases, assignment, *, axes):
    """Require cross-split disjoint values for *each* requested physical factor.

    Solver tolerances, seeds, names, horizons and arbitrary metadata never count
    as parameter holdouts. Onset means simulated fault onset, not independent
    operating-history windows. One station geometry is available: object
    holdout is deliberately unsupported.
    """
    if (type(axes) not in (list, tuple) or not axes or any(type(a) is not str for a in axes)
            or len(set(axes)) != len(axes) or any(a not in AXES for a in axes)):
        raise ValueError('Explicit supported physical axes required; object holdout unavailable')
    # Reuse strict case/assignment/duplicate-scenario checks without treating
    # user-supplied family names as evidence. The preliminary check validates
    # types before accessing case attributes; a derived unique name prevents
    # irrelevant shared labels from rejecting genuinely disjoint factors.
    from atomic_model_station_evaluation import EvaluationCase
    if type(cases) not in (list, tuple) or any(type(c) is not EvaluationCase for c in cases):
        raise ValueError('EvaluationCase collection required')
    derived = [replace(c, parameter_family=c.scenario_hash) for c in cases]
    base = validate_splits(derived, assignment, axes=('parameter_family',))
    rows = []; owners = {}
    for case in sorted(cases, key=lambda c: c.name):
        horizon = case.config.max_steps * case.config.tick_s
        for f in case.config.faults:
            if f.start_tick * case.config.tick_s >= horizon:
                raise ValueError('Unexercised fault cannot define a holdout')
        factors = {a: _values(case.config, a) for a in sorted(axes)}
        for axis, values in factors.items():
            for value in values:
                key = (axis, digest(value))
                previous = owners.setdefault(key, assignment[case.name])
                if previous != assignment[case.name]:
                    raise ValueError('Physical cross-split leakage on ' + axis)
        rows.append(dict(name=case.name, split=assignment[case.name],
                         config=asdict(case.config), factors=factors))
    payload = dict(format='station-physical-holdouts/v1', source=identity(),
                   metadata_manifest=base, axes=sorted(axes), cases=rows,
                   physical_factor_disjoint=True, unseen_objects_validated=False,
                   operating_history_holdout=False, statistical_independence_proven=False)
    return dict(payload, sha256=digest(payload))


def evaluate_holdouts(cases, assignment, policy_factory, *, axes, split='test', policy_id='unversioned'):
    """Audit before running, then retain every result including short/tripped runs."""
    audit = audit_splits(cases, assignment, axes=axes)
    derived = [replace(c, parameter_family=c.scenario_hash) for c in cases]
    result = evaluate(derived, assignment, policy_factory, split=split,
                      axes=('parameter_family',), policy_id=policy_id)
    horizons = {c.name: c.config.max_steps * c.config.tick_s for c in cases}
    for row in result['episodes']:
        row['full_horizon_completed'] = bool(row['status'] == 'completed'
            and abs(row['simulated_s'] - horizons[row['case']]) <= 1e-10)
    if audit['source'] != identity():
        raise ValueError('Physical holdout source changed during execution')
    rows = result['episodes']; count = len(rows)
    return dict(audit=audit, evaluation=result, summary=dict(attempted=count,
        full_horizon_completed=sum(r['full_horizon_completed'] for r in rows),
        execution_failures=sum(r['status'] == 'failed' for r in rows),
        early_terminations=sum(r['status'] == 'completed' and not r['full_horizon_completed'] for r in rows),
        total_wall_s=sum(r['wall_s'] for r in rows),
        total_simulated_s=sum(r['simulated_s'] for r in rows)),
        complete_denominator=True, industrial_generalization_validated=False)
