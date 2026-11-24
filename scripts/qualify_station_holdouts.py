"""Reproducible short native station holdouts and adversarial split controls."""
import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

from atomic_model_multiphysics.provenance import seal
from atomic_model_training.spec import atomic_json
from atomic_model_station_env import EnvConfig
from atomic_model_station_faults import Fault
from atomic_model_station_evaluation import EvaluationCase, hold_policy
from atomic_model_station_holdouts import audit_splits, evaluate_holdouts
from atomic_model_station_holdouts.model import identity

AXES = ('inventory_deadband_m', 'fault_magnitude', 'fault_onset_s')


def cohort():
    cases = [EvaluationCase(str(i), EnvConfig(max_steps=5, tick_s=.01,
        automatic=True, inventory_deadband_m=i*.0004,
        faults=(Fault('pressure_bias', 'bias', 'pressure', i+1, value=(i+1)*150000.),)),
        'single_station', 'unused', 'unused', 'unused') for i in range(3)]
    return cases, dict(zip(('0','1','2'), ('train','dev','test')))


def invalid_policy(observation, schema):
    return {'operator': {'load_fraction':99}}


def scram_policy(observation, schema):
    return {'operator': {'scram':True}}


def scientific(value):
    """Only measured execution cost is excluded from deterministic comparison."""
    if isinstance(value, dict):
        return {k:scientific(v) for k,v in value.items() if k not in ('wall_s','total_wall_s')}
    if isinstance(value, list):
        return [scientific(v) for v in value]
    return value


def run(output):
    out = Path(output).absolute()
    if out.exists() or any(p.is_symlink() for p in (out,*out.parents)):
        raise ValueError('Fresh ordinary output required')
    out.mkdir(parents=True)
    recipe_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cases, assignment = cohort()
    audit = audit_splits(cases, assignment, axes=AXES)
    refusals = {}
    for axis in AXES:
        modified = list(cases)
        c = cases[2].config; f = c.faults[0]
        cfg = (replace(c, inventory_deadband_m=0) if axis=='inventory_deadband_m' else
            replace(c, faults=(replace(f, **({'value':150000.} if axis=='fault_magnitude' else {'start_tick':1})),)))
        modified[2] = replace(cases[2], config=cfg)
        try:
            audit_splits(modified, assignment, axes=[axis])
        except ValueError as error:
            refusals[axis] = str(error)
        else:
            raise ValueError('Leakage control unexpectedly accepted: '+axis)
    episodes = {}
    for policy_id, policy in [('hold',hold_policy), ('invalid',invalid_policy), ('scram',scram_policy)]:
        for split in ('train','dev','test'):
            selected = cases if policy_id!='scram' else [replace(c, config=replace(c.config,
                terminal_on_trip=True)) for c in cases]
            episodes[policy_id+'/'+split] = evaluate_holdouts(selected, assignment,
                lambda p=policy:p, axes=AXES, split=split, policy_id=policy_id+'/v1')
    checks = dict(
        source_unchanged=audit['source']==identity() and recipe_sha256==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        three_factor_leaks_refused=len(refusals)==3,
        held_full_horizon=all(episodes['hold/'+s]['summary']['full_horizon_completed']==1 for s in ('train','dev','test')),
        failed_controls_retained=all(episodes['invalid/'+s]['summary']['execution_failures']==1 and
            episodes['invalid/'+s]['summary']['attempted']==1 for s in ('train','dev','test')),
        trips_do_not_pass_horizon=all(episodes['scram/'+s]['summary']['early_terminations']==1 and
            episodes['scram/'+s]['summary']['full_horizon_completed']==0 for s in ('train','dev','test')))
    result = dict(audit=audit, recipe_sha256=recipe_sha256, leakage_refusals=refusals, episodes=episodes,
        checks=checks, passed=all(checks.values()), closed_tasks=[],
        full_object_parameter_fault_history_holdouts=False,
        finite_configuration_factors_only=True, industrial_generalization_validated=False)
    atomic_json(out/'episodes.json', seal(result))
    atomic_json(out/'scientific.json', seal(scientific(result)))
    if not result['passed']:
        raise ValueError('Station holdout check failed; all attempts retained')
    return dict(passed=result['passed'], checks=checks, attempts=len(episodes))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    print(run(parser.parse_args().output))
