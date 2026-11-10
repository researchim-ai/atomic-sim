"""Branched topology, orientation covariance and rejection of corrupt evidence."""
from copy import deepcopy
import json
import pytest
from scripts import verify_hllc_branching as branch

SHORT_TIMES=(0.,.00005,.0001,.0002)

@pytest.fixture(scope='module')
def pair():
    request=branch.request_for('branch_reversal',2,branch.CAPS[0])
    before=branch.histories.capture(request,times=SHORT_TIMES,restart_index=2)
    after=branch.histories.capture(branch.reverse_edge(request,'jb'),times=SHORT_TIMES,restart_index=2)
    return before,after


@pytest.mark.parametrize('case,cells',branch.MATRIX)
def test_topology_has_two_loops_and_heterogeneous_ports(case,cells):
    request=branch.request_for(case,cells,branch.CAPS[-1]);spec=request['model']
    assert len(spec['nodes'])==4 and len(spec['edges'])==5
    assert len(spec['edges'])-len(spec['nodes'])+1==2
    assert sum(e['from_node']=='j' or e['to_node']=='j' for e in spec['edges'])==3
    assert len({e['diameter_m'] for e in spec['edges']})==5
    assert all(len(e['initial'])==cells for e in spec['edges'])


def test_axis_reversal_is_pure_involution_and_remaps_wall():
    request=branch.request_for('branch_reversal',4,branch.CAPS[-1]);snapshot=deepcopy(request)
    reversed_request=branch.reverse_edge(request,'jb')
    assert request==snapshot and reversed_request['model']['walls'][0]['cell']=='jb/3'
    assert json.dumps(branch.reverse_edge(reversed_request,'jb'),sort_keys=True)==json.dumps(request,sort_keys=True)
    assert reversed_request['model']['edges'][1]['initial'][0]['velocity_m_s']==.005


def test_real_short_network_preserves_covariance_and_support(pair):
    a,b=pair
    assert a['record']['passed'] and b['record']['passed']
    assert branch.audit(a)['passed'] and branch.audit(b)['passed']
    assert branch.orientation(a,b,'jb')['passed']
    # Scalar reactions depend on the selected local axes. They are audited
    # independently from the face tensor ledger, not compared as a3D invariant.
    assert a['observations'][-1]['local_ledgers']['node_support_impulse_kg_m_s']!=b['observations'][-1]['local_ledgers']['node_support_impulse_kg_m_s']


@pytest.mark.parametrize('damage',['time','nan_cell','nan_face','short_face','request'])
def test_orientation_refuses_inconsistent_or_nonfinite_evidence(pair,damage):
    a,b=deepcopy(pair)
    if damage=='time':b['observations'][1]['time_s']+=1e-4
    elif damage=='nan_cell':b['observations'][1]['cells'][0]['pressure_pa']=float('nan')
    elif damage=='nan_face':b['observations'][1]['local_ledgers']['integrated_face_mass_kg'][0]=float('nan')
    elif damage=='short_face':
        for packet in (a,b):packet['observations'][1]['local_ledgers']['integrated_face_energy_j'].pop()
    else:b['request']['model']['edges'][0]['length_m']+=.01
    with pytest.raises(ValueError):branch.orientation(a,b,'jb')


@pytest.mark.parametrize('damage',['nan_support','nan_momentum','zero_mass','node_momentum','short_ledger'])
def test_local_support_audit_refuses_corrupt_data(pair,damage):
    packet=deepcopy(pair[0]);snapshot=packet['observations'][1]
    if damage=='nan_support':snapshot['local_ledgers']['node_support_impulse_kg_m_s'][0]=float('nan')
    elif damage=='nan_momentum':snapshot['local_ledgers']['integrated_face_momentum_kg_m_s'][0]=float('nan')
    elif damage=='zero_mass':
        for c in snapshot['cells']:
            if c['id'].startswith('jb/'):c['mass_kg']=0.
    elif damage=='node_momentum':snapshot['cells'][0]['momentum_kg_m_s']=1e-6
    else:snapshot['local_ledgers']['node_support_impulse_kg_m_s'].pop()
    with pytest.raises(ValueError):branch.audit(packet)


@pytest.mark.parametrize('case,cells,cap',[('unknown',2,5e-6),('branched_rest',3,5e-6),('branch_reversal',2,.1)])
def test_unknown_matrix_refused(case,cells,cap):
    with pytest.raises(ValueError):branch.request_for(case,cells,cap)
