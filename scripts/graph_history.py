"""Strict arbitrary-time graph comparison; no global benchmark-grid mutation."""
import math
from scripts.benchmark_modelica_network import BUDGETS,COLUMNS,topology


def validate_times(times):
    try:
        raw=tuple(times)
        if any(isinstance(t,bool) for t in raw):raise ValueError('Boolean time')
        values=tuple(float(t) for t in raw)
    except (TypeError,ValueError,OverflowError) as error:raise ValueError('Finite ordered time grid required') from error
    if len(values)<2 or values[0]!=0. or any(not math.isfinite(t) for t in values) or any(b<=a for a,b in zip(values,values[1:])):
        raise ValueError('Finite time grid must start at zero and increase strictly')
    return values


def compare(native,peer,*,times,budgets=BUDGETS):
    times=validate_times(times)
    if len(peer)!=len(times) or len(native['observations'])!=len(times):raise ValueError('Incomplete graph histories')
    errors={k:0. for k in budgets}
    names=topology(native['request'],native['observations'][0])[0]
    for t,a,b in zip(times,native['observations'],peer,strict=True):
        if any(not math.isfinite(v) for v in b.values()) or not math.isfinite(a['time_s']) or abs(a['time_s']-t)>1e-12 or abs(b['time']-t)>1e-12:raise ValueError('Mismatched/nonfinite graph times or values')
        if [c['id'] for c in a['cells']]!=names:raise ValueError('Graph cell identity changed')
        for i,c in enumerate(a['cells'],1):
            for field,column in COLUMNS.items():
                error=abs(c[field]-b[f'{column}[{i}]'])
                if not math.isfinite(error):raise ValueError('Nonfinite graph comparison')
                errors[field]=max(errors[field],error)
        ledger=a['local_ledgers']
        for field,name,key in [('face_mass_kg','faceMass','integrated_face_mass_kg'),('face_energy_j','faceEnergy','integrated_face_energy_j'),('support_kg_m_s','support','node_support_impulse_kg_m_s')]:
            count=len(names) if name=='support' else len(topology(native['request'],native['observations'][0])[2])
            if len(ledger[key])!=count:raise ValueError('Incomplete face/support ledger')
            for i,value in enumerate(ledger[key],1):
                error=abs(value-b[f'{name}[{i}]'])
                if not math.isfinite(error):raise ValueError('Nonfinite face/support comparison')
                errors[field]=max(errors[field],error)
    n=len(native['observations'][0]['cells']);first=peer[0]
    mass_error=max(abs(sum(row[f'mass[{i}]']-first[f'mass[{i}]'] for i in range(1,n+1))) for row in peer)
    energy_error=max(abs(sum(row[f'energy[{i}]']-first[f'energy[{i}]'] for i in range(1,n+1))) for row in peer)
    momentum_error=max(abs(sum(row[f'momentum[{i}]']-first[f'momentum[{i}]']-row[f'support[{i}]'] for i in range(1,n+1))) for row in peer)
    checks=dict(native_passed=native['record']['passed'],peer_agreement=all(errors[k]<=budgets[k] for k in errors),
        mass_balance=mass_error<=1e-9,energy_balance=energy_error<=1e-5,signed_axial_momentum_balance=momentum_error<=1e-9)
    return dict(passed=all(checks.values()),checks=checks,maximum_errors=errors,
                peer_mass_residual_kg=mass_error,peer_energy_residual_j=energy_error,peer_momentum_residual_kg_m_s=momentum_error)

