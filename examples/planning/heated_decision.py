"""Try alternate wall powers without touching live state, then commit once.

Run: python examples/planning/heated_decision.py
The fixture is a research graph, not a plant operating recommendation.
"""
from atomic_model_decision import evaluate,commit
from atomic_model_multiphysics.provenance import unseal
from scripts.verify_agent_decisions import setup

model,candidates,target,limits=setup('wall_control')
packet=evaluate(model,candidates,target=target,limits=limits)
report=unseal(packet)
for result in report['results']:
    print(result['id'],result['status'],result['metrics'])
print('Selected:',report['selected_id'])
if report['selected_id'] is not None:
    observation=commit(model,packet)
    print('Committed simulation time:',observation['time_s'])
