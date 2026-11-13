"""Frozen-policy initial-bias and bounded sensor-drift stress tests."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

import numpy as np
from atomic_model_multiphysics.provenance import seal, unseal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_thermal_observer import ThermalMonitor
from atomic_model_thermal_recovery import ThermalRecovery, signature
from scripts import train_adaptive_thermal_recovery as adaptive
from scripts.audit_adaptive_thermal_recovery import audit as audit_parent, replay_episode

SCENARIOS = (('healthy', 0., 0), ('initial_low', -.15, 0), ('initial_high', .15, 0),
             ('drift_low', -.24, 0), ('drift_high', .24, 0))
SEEDS = (1010, 1011)
POLICIES = ('continue', 'four', 'selected')


def sensor_observation(observation, scenario, index, steps):
    if (tuple(scenario) not in SCENARIOS or type(index) is not int or index < 0
            or type(steps) is not int or steps < 2):
        raise ValueError('Declared drift scenario and sample domain required')
    o = np.asarray(observation)
    if o.shape != (6,) or o.dtype.kind not in 'if' or not np.isfinite(o).all():
        raise ValueError('Finite observation6 required')
    o = o.astype(float, copy=True)
    kind, amplitude, _ = scenario
    fraction = min(index / (steps - 1), 1.) if kind.startswith('drift') else 1.
    o[0] += amplitude * fraction
    return o


class DriftEnv(adaptive.prior.objective.TailEnv):
    def __init__(self, config, scenario):
        sensor_observation([0.] * 6, scenario, 0, config.steps)
        self.scenario = tuple(scenario)
        super().__init__(config)

    def _observation(self, snapshot):
        return sensor_observation(super()._observation(snapshot), self.scenario,
                                  self._index, self.decision_steps)


def rollout(cfg, scenario, policy, candidate, training, seed):
    if policy not in POLICIES or candidate not in adaptive.CANDIDATES:
        raise ValueError('Declared frozen policy required')
    env = DriftEnv(cfg, scenario); obs, _ = env.reset(seed=seed)
    actor = adaptive.prior.faults.Actor('learned', training['selected'], training['imitation']['weights'], cfg.action_dt_s, cfg.max_power_w)
    monitor = ThermalMonitor(adaptive.prior.monitor_config(cfg, scenario, 'test'))
    controller = ThermalRecovery(adaptive.config_for(candidate, cfg))
    trace = []; first = None
    for index in range(env.config.steps):
        diagnostic = monitor.observe(obs)
        if diagnostic['alarm'] and first is None: first = index
        normal = actor.act(obs); recovery = controller.act(obs, diagnostic['alarm'], normal)
        action = recovery['action'] if candidate.startswith('model_') else adaptive.prior.recovery_action(candidate, normal, diagnostic['alarm'])
        if index >= env.decision_steps: action = 0.
        before = obs.tolist(); obs, _, terminated, truncated, info = env.step([action])
        trace.append(dict(**info, action=action, policy_observation=before, tail=index >= env.decision_steps,
                          diagnostic=diagnostic, recovery=recovery))
        if terminated or truncated: break
    drained = env._power == 0. and all(v == 0. for v in env._queue)
    row = dict(policy=policy, candidate=candidate, scenario=list(scenario), seed=seed, split='test',
               config=asdict(cfg), trace=trace, alarm_index=first, actuator_drained=drained,
               completed=len(trace) == env.config.steps and not terminated and drained,
               iae_k_s=math.fsum(s['iae_increment_k_s'] for s in trace), energy_j=math.fsum(s['energy_j'] for s in trace))
    row['score'] = adaptive.prior.objective.score(row)
    return row


def summarize(rows):
    expected = {(p, c, s[0], seed) for p in POLICIES for c in range(4) for s in SCENARIOS for seed in SEEDS}
    lookup = {}
    for r in rows:
        key = (r['policy'], r['case'], r['scenario'][0], r['seed'])
        if key in lookup or key not in expected or tuple(r['scenario']) not in SCENARIOS or r['split'] != 'test':
            raise ValueError('Stress matrix provenance/coverage')
        if any(type(r[k]) not in (float, int) or not math.isfinite(r[k]) or r[k] < 0 for k in ('iae_k_s', 'energy_j')):
            raise ValueError('Finite stress metrics required')
        lookup[key] = r
    if set(lookup) != expected: raise ValueError('Incomplete stress matrix')
    result = []
    rows = sorted(rows, key=lambda r: (r['policy'], r['case'], r['scenario'][0], r['seed']))
    for policy in POLICIES:
        for scenario in SCENARIOS:
            group = [r for r in rows if r['policy'] == policy and r['scenario'][0] == scenario[0]]
            if any(r['completed'] and (type(r['score']) not in (int, float) or not math.isfinite(r['score']) or r['score'] < 0) for r in group):
                raise ValueError('Finite stress objective required')
            result.append(dict(policy=policy, scenario=scenario[0], episodes=len(group),
                completed=sum(r['completed'] for r in group), alarms=sum(r['alarm_index'] is not None for r in group),
                tracking_degraded=sum(r['iae_k_s'] > 1.25 * lookup[(policy, r['case'], 'healthy', r['seed'])]['iae_k_s'] + 1e-6 for r in group),
                mean_iae_k_s=math.fsum(r['iae_k_s'] for r in group) / len(group),
                total_energy_j=math.fsum(r['energy_j'] for r in group),
                mean_score=math.fsum(r['score'] for r in group) / len(group) if all(r['completed'] for r in group) else None))
    return result


def _task(args):
    case, cfg, scenario, policy, candidate, training, seed = args
    row = rollout(cfg, scenario, policy, candidate, training, seed); row['case'] = case
    return f'test-{case}-{scenario[0]}-{seed}-{policy}.json', row


def run(output, training_path, parent, workers=1):
    if type(workers) is not int or not 1 <= workers <= 8: raise ValueError('Workers1..8 required')
    output = Path(output).absolute(); training_path = Path(training_path).absolute(); parent = Path(parent).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)): raise ValueError('Symlink output')
    audit_parent(parent, training_path)
    read = lambda f: unseal(json.loads(f.read_text()))
    training = read(training_path); selection = read(parent / 'selection.json')
    output.mkdir(parents=True, exist_ok=False)
    drivers = {str(Path(p).absolute()): sha(p) for p in (__file__, adaptive.__file__, adaptive.prior.__file__, adaptive.prior.faults.__file__, adaptive.prior.objective.__file__, adaptive.prior.faults.original.__file__)}
    protocol = dict(source=adaptive.prior.signature(), monitor_source=adaptive.prior.monitor_signature(), recovery_source=signature(),
                    drivers=drivers, training_sha256=sha(training_path), parent_selection_sha256=sha(parent / 'selection.json'),
                    parent_report_sha256=sha(parent / 'report.json'), selected=selection['selected'], scenarios=SCENARIOS,
                    seeds=SEEDS, decision_steps=16, workers=workers, test_used_for_selection=False, blind_test=False)
    atomic_json(output / 'protocol.json', seal(protocol))
    tasks = [(case, replace(cfg, steps=16), scenario, policy, selection['selected'] if policy == 'selected' else policy, training, seed)
             for case, cfg in enumerate(adaptive.prior.faults.original.cases('test')) for scenario in SCENARIOS for seed in SEEDS for policy in POLICIES]
    rows = []
    def collect(results):
        for name, row in results:
            atomic_json(output / name, seal(row)); rows.append(row)
            if len(rows) % 12 == 0: print('episodes', len(rows), '/', len(tasks), flush=True)
    if workers == 1: collect(map(_task, tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool: collect(pool.map(_task, tasks))
    for row in rows: replay_episode(row, training)
    unchanged = (protocol['source'] == adaptive.prior.signature() and protocol['monitor_source'] == adaptive.prior.monitor_signature()
                 and protocol['recovery_source'] == signature() and all(sha(p) == h for p, h in drivers.items()))
    report = dict(completed=True, passed=unchanged and all(r['completed'] for r in rows), source_unchanged=unchanged,
                  episodes=len(rows), summary=summarize(rows), decisions_replayed=True, solver_replayed=False,
                  superiority_established=False, validated=False, plant_referenced=False)
    atomic_json(output / 'report.json', seal(report)); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True); parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args(); run(args.output, args.training, args.parent, args.workers)
