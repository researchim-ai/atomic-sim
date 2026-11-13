from copy import deepcopy
from dataclasses import asdict
import pytest
from atomic_model_thermal_agent import ThermalConfig
from scripts.audit_thermal_recovery import audit_episode
from scripts.train_thermal_recovery import objective


def episode():
    step = dict(iae_increment_k_s=.0001, energy_j=1., time_s=.001,
                diagnostic={'alarm': False}, violations=False, tail=True,
                action=0., applied_power_w=0., delayed_power_w=0.,
                global_ledger=dict(mass_residual_kg=0., energy_residual_j=0.))
    row = dict(trace=[step], completed=True, actuator_drained=True,
               iae_k_s=.0001, energy_j=1., alarm_index=None,
               config=asdict(ThermalConfig()))
    row['score'] = objective.score(row)
    return row


def test_valid_trace_metrics():
    audit_episode(episode())


@pytest.mark.parametrize('damage', ['metric', 'nan', 'alarm', 'balance', 'tail', 'time', 'score', 'incomplete'])
def test_audit_refuses_internally_resealed_corruption(damage):
    row = deepcopy(episode()); step = row['trace'][0]
    if damage == 'metric': row['energy_j'] = 2.
    elif damage == 'nan': step['energy_j'] = float('nan')
    elif damage == 'alarm': row['alarm_index'] = 0
    elif damage == 'balance': step['global_ledger']['energy_residual_j'] = .1
    elif damage == 'tail': step['applied_power_w'] = 1.
    elif damage == 'time': step['time_s'] = 0.
    elif damage == 'score': row['score'] += 1.
    else: row['completed'] = False
    with pytest.raises(ValueError): audit_episode(row)
