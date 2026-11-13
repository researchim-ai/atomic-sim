"""Public train-only comparison of causal model-based thermal fallback."""
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, unseal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_thermal_observer import ThermalMonitor
from atomic_model_thermal_recovery import RecoveryConfig, ThermalRecovery, signature
from scripts import train_thermal_recovery as prior

CANDIDATES = ('continue', 'four', 'model_mid', 'model_light')
TRAIN = (('healthy', 0., 2), ('bias_low', -.1, 2), ('bias_high', .1, 2),
         ('stuck', 0., 2), ('model_mismatch', 0., 2))
TEST = (('healthy', 0., 5), ('bias_low', -.25, 5), ('bias_high', .25, 5),
        ('stuck', 0., 5), ('model_mismatch', 0., 5))
TRAIN_SEEDS = (810,)
TEST_SEEDS = (910, 911)
POLICIES = ('continue', 'four', 'selected')


def config_for(candidate, cfg):
    if candidate not in CANDIDATES:
        raise ValueError('Unknown recovery candidate')
    c, k = (150., 4000.) if candidate == 'model_light' else (300., 8000.)
    return RecoveryConfig(dt_s=cfg.action_dt_s, max_power_w=cfg.max_power_w,
                          capacity_j_k=c, conductance_w_k=k)


def rollout(cfg, scenario, candidate, training, seed, split):
    controller = ThermalRecovery(config_for(candidate, cfg))
    env = prior.RecoveryEnv(cfg, scenario); obs, _ = env.reset(seed=seed)
    actor = prior.faults.Actor('learned', training['selected'], training['imitation']['weights'],
                              cfg.action_dt_s, cfg.max_power_w)
    monitor = ThermalMonitor(prior.monitor_config(cfg, scenario, split))
    trace = []; first = None
    for index in range(env.config.steps):
        diagnostic = monitor.observe(obs)
        if diagnostic['alarm'] and first is None: first = index
        normal = actor.act(obs)
        recovery = controller.act(obs, diagnostic['alarm'], normal)
        action = (recovery['action'] if candidate.startswith('model_') else
                  prior.recovery_action(candidate, normal, diagnostic['alarm']))
        if index >= env.decision_steps: action = 0.
        before = obs.tolist(); obs, _, terminated, truncated, info = env.step([action])
        trace.append(dict(**info, action=action, policy_observation=before,
                          tail=index >= env.decision_steps, diagnostic=diagnostic, recovery=recovery))
        if terminated or truncated: break
    drained = env._power == 0. and all(x == 0. for x in env._queue)
    row = dict(candidate=candidate, split=split, seed=seed, scenario=list(scenario), config=asdict(cfg),
               trace=trace, alarm_index=first, actuator_drained=drained,
               completed=len(trace) == env.config.steps and not terminated and drained,
               iae_k_s=math.fsum(s['iae_increment_k_s'] for s in trace),
               energy_j=math.fsum(s['energy_j'] for s in trace))
    row['score'] = prior.objective.score(row)
    return row


def check_matrix(rows, split):
    scenarios, seeds, count, policies = (TRAIN, TRAIN_SEEDS, 2, CANDIDATES) if split == 'train' else (TEST, TEST_SEEDS, 4, POLICIES)
    field = 'candidate' if split == 'train' else 'policy'
    expected = {(p, c, s[0], seed) for p in policies for c in range(count) for s in scenarios for seed in seeds}
    seen = set()
    for row in rows:
        key = (row[field], row['case'], row['scenario'][0], row['seed'])
        if key in seen or key not in expected or row['split'] != split or tuple(row['scenario']) not in scenarios:
            raise ValueError('Episode coverage/provenance')
        seen.add(key)
        if row['completed'] and (type(row['score']) not in (float, int) or not math.isfinite(row['score']) or row['score'] < 0):
            raise ValueError('Finite objective required')
    if seen != expected: raise ValueError('Incomplete episode matrix')


def select(rows):
    check_matrix(rows, 'train')
    rows = sorted(rows, key=lambda r: (r['candidate'], r['case'], r['scenario'][0], r['seed']))
    candidates = []
    for candidate in CANDIDATES:
        subset = [r for r in rows if r['candidate'] == candidate]
        if not all(r['completed'] for r in subset): continue
        groups = [[r['score'] for r in subset if r['scenario'][0] == s[0]] for s in TRAIN]
        candidates.append(dict(candidate=candidate, worst_group_score=max(math.fsum(g) / len(g) for g in groups),
                               mean_score=math.fsum(r['score'] for r in subset) / len(subset)))
    if not candidates: raise ValueError('No completed train candidate')
    chosen = min(candidates, key=lambda c: (c['worst_group_score'], c['mean_score'], c['candidate']))
    return dict(selected=chosen['candidate'], candidates=candidates, test_used=False,
                criterion='Minimum worst scenario-group mean J, then overall mean J and id')


def summarize(rows):
    check_matrix(rows, 'test')
    rows = sorted(rows, key=lambda r: (r['policy'], r['case'], r['scenario'][0], r['seed']))
    healthy = {(r['policy'], r['case'], r['seed']): r for r in rows if r['scenario'][0] == 'healthy'}
    result = []
    for policy in POLICIES:
        for scenario in TEST:
            group = [r for r in rows if r['policy'] == policy and r['scenario'][0] == scenario[0]]
            result.append(dict(policy=policy, scenario=scenario[0], episodes=len(group),
                completed=sum(r['completed'] for r in group),
                mean_score=math.fsum(r['score'] for r in group) / len(group) if all(r['completed'] for r in group) else None,
                mean_iae_k_s=math.fsum(r['iae_k_s'] for r in group) / len(group),
                total_energy_j=math.fsum(r['energy_j'] for r in group),
                tracking_degraded=sum(r['iae_k_s'] > 1.25 * healthy[(policy, r['case'], r['seed'])]['iae_k_s'] + 1e-6 for r in group)))
    return result


def run(output, training_path):
    output = Path(output).absolute(); training_path = Path(training_path).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)): raise ValueError('Symlink output')
    training = unseal(json.loads(training_path.read_text()))
    frozen = unseal(json.loads(training_path.with_name('protocol.json').read_text()))
    if training['test_used'] is not False or frozen['source'] != prior.signature() or any(sha(p) != h for p, h in frozen['drivers'].items()):
        raise ValueError('Frozen base policy provenance')
    output.mkdir(parents=True, exist_ok=False)
    drivers = {str(Path(p).absolute()): sha(p) for p in (__file__, prior.__file__, prior.faults.__file__, prior.objective.__file__, prior.faults.original.__file__)}
    protocol = dict(source=prior.signature(), monitor_source=prior.monitor_signature(), recovery_source=signature(),
                    drivers=drivers, training_sha256=sha(training_path), candidates=CANDIDATES,
                    train_scenarios=TRAIN, test_scenarios=TEST, train_seeds=TRAIN_SEEDS, test_seeds=TEST_SEEDS,
                    decision_steps=16, candidate_configs={k: asdict(config_for(k, prior.faults.original.cases('train')[0])) for k in CANDIDATES},
                    blind_test=False, previous_test_informed_design=True,
                    note='Public development experiment; wall, fluid, ambient, target, power only. No industrial validation.')
    atomic_json(output / 'protocol.json', seal(protocol))
    rows = []
    for candidate in CANDIDATES:
        for case, cfg in enumerate([prior.faults.original.cases('train')[i] for i in (0, 3)]):
            cfg = replace(cfg, steps=16)
            for scenario in TRAIN:
                row = rollout(cfg, scenario, candidate, training, TRAIN_SEEDS[0], 'train'); row['case'] = case; rows.append(row)
                atomic_json(output / f'train-{candidate}-{case}-{scenario[0]}.json', seal(row))
        print('train', candidate, flush=True)
    selection = select(rows); atomic_json(output / 'selection.json', seal(selection)); print('selected', selection['selected'], flush=True)
    tests = []
    for case, cfg in enumerate(prior.faults.original.cases('test')):
        cfg = replace(cfg, steps=16)
        for scenario in TEST:
            for seed in TEST_SEEDS:
                for policy in POLICIES:
                    candidate = selection['selected'] if policy == 'selected' else policy
                    row = rollout(cfg, scenario, candidate, training, seed, 'test'); row.update(case=case, policy=policy); tests.append(row)
                    atomic_json(output / f'test-{case}-{scenario[0]}-{seed}-{policy}.json', seal(row))
        print('test case', case, flush=True)
    unchanged = (protocol['source'] == prior.signature() and protocol['monitor_source'] == prior.monitor_signature()
                 and protocol['recovery_source'] == signature() and all(sha(p) == h for p, h in drivers.items()))
    report = dict(completed=True, passed=unchanged and all(r['completed'] for r in rows + tests), source_unchanged=unchanged,
                  selection=selection, summary=summarize(tests), train_episodes=len(rows), test_episodes=len(tests),
                  superiority_established=False, validated=False, plant_referenced=False)
    atomic_json(output / 'report.json', seal(report)); return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.output, args.training)
