"""Independently reconstruct recovery metrics from sealed episode traces."""
import argparse
import json
import math
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, sha, unseal
from atomic_model_training.spec import atomic_json
from scripts import train_thermal_recovery as recovery


def audit_episode(row):
    """Check recorded metrics and actuator tail; does not rerun the solver."""
    trace = row['trace']
    if not trace or row['completed'] is not True or row['actuator_drained'] is not True:
        raise ValueError('Incomplete recovery episode')
    for key, increment in [('iae_k_s', 'iae_increment_k_s'), ('energy_j', 'energy_j')]:
        values = [step[increment] for step in trace]
        if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in values):
            raise ValueError('Invalid trace increment')
        if row[key] != math.fsum(values):
            raise ValueError('Metric differs from trace')
    if row['score'] != recovery.objective.score(row):
        raise ValueError('Objective differs from trace')
    first = next((i for i, step in enumerate(trace) if step['diagnostic']['alarm']), None)
    if first != row['alarm_index']:
        raise ValueError('Alarm index differs from trace')
    last_time = 0.
    for step in trace:
        if not math.isfinite(step['time_s']) or step['time_s'] <= last_time:
            raise ValueError('Nonmonotone trace time')
        last_time = step['time_s']
        ledger = step['global_ledger']
        if step['violations'] or any(not math.isfinite(ledger[k]) or abs(ledger[k]) > tol
                                   for k, tol in [('mass_residual_kg', 1e-9), ('energy_residual_j', 1e-5)]):
            raise ValueError('Observed limit or balance failure')
        if step['tail'] and step['action'] != 0.:
            raise ValueError('Nonzero tail request')
    if not trace[-1]['tail'] or any(trace[-1][k] != 0. for k in ('action', 'applied_power_w', 'delayed_power_w')):
        raise ValueError('Actuator tail not drained')


def audit(directory):
    directory = Path(directory)
    def read(name):
        return unseal(json.loads((directory / name).read_text()))
    protocol = read('protocol.json')
    if protocol['source'] != recovery.signature() or protocol['monitor_source'] != recovery.monitor_signature():
        raise ValueError('Source signature changed')
    if any(sha(p) != expected for p, expected in protocol['drivers'].items()):
        raise ValueError('Driver changed')
    train = [read(p.name) for p in sorted(directory.glob('train-*.json'))]
    test = [read(p.name) for p in sorted(directory.glob('test-*.json'))]
    for row in train + test:
        audit_episode(row)
    selection = read('selection.json')
    report = read('report.json')
    if recovery.select(train) != selection or report['selection'] != selection:
        raise ValueError('Frozen selection mismatch')
    for row in test:
        expected = selection['selected'] if row['policy'] == 'selected' else row['policy']
        if row['candidate'] != expected:
            raise ValueError('Test policy differs from frozen selection')
    if recovery.summary(test) != report['summary']:
        raise ValueError('Summary differs from episodes')
    if (report['train_episodes'], report['test_episodes']) != (len(train), len(test)):
        raise ValueError('Episode count mismatch')
    if any(report[k] is not True for k in ('completed', 'passed', 'source_unchanged')):
        raise ValueError('Experiment did not pass')
    return dict(passed=True, train_episodes=len(train), test_episodes=len(test),
                selection_exact=True, trace_metrics_exact=True,
                report_sha256=sha(directory / 'report.json'),
                solver_replayed=False, validated=False, plant_referenced=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, seal(audit(args.input)))
