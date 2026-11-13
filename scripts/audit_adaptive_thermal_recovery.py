"""Audit adaptive fallback traces and replay decisions without running physics."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, unseal, sha
from atomic_model_training.spec import atomic_json
from atomic_model_thermal_agent import ThermalConfig
from atomic_model_thermal_observer import ThermalMonitor
from atomic_model_thermal_recovery import ThermalRecovery, signature
from scripts.audit_thermal_recovery import audit_episode
from scripts import train_adaptive_thermal_recovery as experiment


def replay_episode(row, training):
    audit_episode(row)
    cfg = ThermalConfig(**row['config'])
    expected = cfg.steps + cfg.delay_steps + math.ceil(cfg.max_power_w / (cfg.slew_w_per_s * cfg.action_dt_s)) + 2
    if len(row['trace']) != expected:
        raise ValueError('Trace length differs from declared horizon')
    controller = ThermalRecovery(experiment.config_for(row['candidate'], cfg))
    monitor = ThermalMonitor(experiment.prior.monitor_config(cfg, row['scenario'], row['split']))
    actor = experiment.prior.faults.Actor('learned', training['selected'], training['imitation']['weights'], cfg.action_dt_s, cfg.max_power_w)
    for index, step in enumerate(row['trace']):
        obs = step['policy_observation']
        diagnostic = monitor.observe(obs)
        if diagnostic != step['diagnostic']:
            raise ValueError('Monitor replay differs')
        normal = actor.act(obs)
        recovery = controller.act(obs, diagnostic['alarm'], normal)
        if recovery != step['recovery']:
            raise ValueError('Recovery state replay differs')
        action = (recovery['action'] if row['candidate'].startswith('model_') else
                  experiment.prior.recovery_action(row['candidate'], normal, diagnostic['alarm']))
        tail = index >= cfg.steps
        if tail: action = 0.
        if step['tail'] is not tail or step['action'] != action:
            raise ValueError('Action differs from frozen causal policy')
        if index and obs[4] != row['trace'][index - 1]['applied_power_w'] / cfg.max_power_w:
            raise ValueError('Applied power observation chronology differs')
    return True


def audit(directory, training_path):
    directory = Path(directory); training_path = Path(training_path)
    read = lambda f: unseal(json.loads(f.read_text()))
    protocol = read(directory / 'protocol.json'); training = read(training_path)
    if (sha(training_path) != protocol['training_sha256'] or training['test_used'] is not False
            or protocol['source'] != experiment.prior.signature()
            or protocol['monitor_source'] != experiment.prior.monitor_signature()
            or protocol['recovery_source'] != signature()
            or any(sha(p) != h for p, h in protocol['drivers'].items())):
        raise ValueError('Experiment provenance changed')
    train = [read(f) for f in sorted(directory.glob('train-*.json'))]
    test = [read(f) for f in sorted(directory.glob('test-*.json'))]
    selection = read(directory / 'selection.json'); report = read(directory / 'report.json')
    if experiment.select(train) != selection or report['selection'] != selection:
        raise ValueError('Selection differs from train matrix')
    if experiment.summarize(test) != report['summary']:
        raise ValueError('Test summary differs')
    for row in test:
        candidate = selection['selected'] if row['policy'] == 'selected' else row['policy']
        if row['candidate'] != candidate:
            raise ValueError('Test candidate differs from frozen choice')
    train_configs = [experiment.prior.faults.original.cases('train')[i] for i in (0, 3)]
    test_configs = experiment.prior.faults.original.cases('test')
    for row in train + test:
        configs = train_configs if row['split'] == 'train' else test_configs
        cfg = experiment.replace(configs[row['case']], steps=protocol['decision_steps'])
        if row['config'] != asdict(cfg):
            raise ValueError('Plant differs from declared matrix')
        replay_episode(row, training)
    if (report['train_episodes'], report['test_episodes']) != (len(train), len(test)) or any(report[k] is not True for k in ('passed', 'completed', 'source_unchanged')):
        raise ValueError('Incomplete experiment report')
    return dict(passed=True, train_episodes=len(train), test_episodes=len(test),
                selection_exact=True, decisions_replayed=True, trace_metrics_exact=True,
                solver_replayed=False, report_sha256=sha(directory / 'report.json'),
                validated=False, plant_referenced=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); atomic_json(args.output, seal(audit(args.input, args.training)))
