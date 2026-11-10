"""CSV admission fixtures are copied native rows, not external measurements."""
from copy import deepcopy
import csv
import pytest
from scripts import benchmark_modelica_network as bench
from scripts import verify_heterogeneous_network as qualification


@pytest.fixture(scope='module')
def native():return bench.execute('coupled_gradient',2,1e-5)


def write(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def test_complete_native_fixture_round_trip_and_nonzero_support(tmp_path,native):
    rows=qualification.as_peer(native['observations']);path=tmp_path/'fixture.csv';write(path,rows)
    n=len(native['observations'][0]['cells']);nf=len(bench.topology(native['request'],native['observations'][0])[2])
    peer=bench.load_csv(path,n,nf)
    assert bench.compare(native,peer)['passed']
    assert native['record']['checks']['support_nonzero']
    assert native['record']['checks']['restart_exact']


@pytest.mark.parametrize('damage',['prefix','nan','ordering','overshoot','negative_time','negative_mass','phase_domain','event','missing_column'])
def test_incomplete_or_corrupt_external_trace_refused(tmp_path,native,damage):
    rows=qualification.as_peer(native['observations'])
    if damage=='prefix':rows.pop()
    elif damage=='nan':rows[2]['faceMass[1]']=float('nan')
    elif damage=='ordering':rows.reverse()
    elif damage=='overshoot':rows.append(dict(rows[-1],time=.002))
    elif damage=='negative_time':rows.insert(0,dict(rows[0],time=-.001))
    elif damage=='negative_mass':rows[2]['mass[1]']=-1.
    elif damage=='phase_domain':rows[2]['temperature[1]']=500.
    elif damage=='missing_column':
        for row in rows:row.pop('support[1]')
    else:
        row=dict(rows[2]);row['faceEnergy[1]']+=1.;rows.insert(3,row)
    path=tmp_path/'fixture.csv';write(path,rows)
    n=len(native['observations'][0]['cells']);nf=len(bench.topology(native['request'],native['observations'][0])[2])
    with pytest.raises((ValueError,KeyError)):bench.load_csv(path,n,nf)


@pytest.mark.parametrize('damage',['native_nan_time','peer_nan_face','native_nan_face','prefix_ledger','cell_identity'])
def test_direct_comparison_cannot_hide_broken_identity_or_nonfinite_ledger(native,damage):
    n=deepcopy(native);peer=qualification.as_peer(n['observations'])
    if damage=='native_nan_time':n['observations'][2]['time_s']=float('nan')
    elif damage=='peer_nan_face':peer[2]['faceMass[1]']=float('nan')
    elif damage=='native_nan_face':n['observations'][2]['local_ledgers']['integrated_face_mass_kg'][0]=float('nan')
    elif damage=='prefix_ledger':n['observations'][2]['local_ledgers']['integrated_face_mass_kg'].pop()
    else:n['observations'][2]['cells'][0]['id']='other'
    with pytest.raises(ValueError):bench.compare(n,peer)


def test_export_has_independent_conserved_eos_and_graph_equations(native):
    text=bench.render(native)
    assert 'waterBaseProp_dT(rho,T,1,1)' in text and 'AtomicLiquidState' in text
    assert 'incidence' in text and 'der(faceEnergy[k])' in text
    assert 'der(support[i])' in text and 'der(momentum[i])=if i<=2 then 0' in text
    assert 'der(centeredEnergy[i])' in text
    assert 'ClosedVolume' not in text and 'DynamicPipe' not in text


@pytest.mark.parametrize('damage',['identity','volume','nan_volume'])
def test_export_refuses_unmatched_initial_geometry(native,damage):
    n=deepcopy(native);c=n['observations'][0]['cells'][0]
    if damage=='identity':c['id']='other'
    elif damage=='volume':c['volume_m3']+=.01
    else:c['volume_m3']=float('nan')
    with pytest.raises(ValueError):bench.render(n)


@pytest.mark.parametrize('case',bench.CASES)
def test_actual_heterogeneous_native_trajectory_balances_and_restarts(case):
    result=bench.execute(case,2,1e-5)
    assert result['record']['passed'],result['record']
    edges=result['request']['model']['edges']
    assert edges[0]['diameter_m']!=edges[1]['diameter_m']
    assert edges[0]['from_node']==edges[1]['to_node']


@pytest.mark.parametrize('cells',qualification.SPATIAL_MESHES)
def test_spatial_refinement_keeps_two_identical_physical_blocks(cells):
    r=qualification.contact_request(cells)
    for e in r['model']['edges']:
        assert len(e['initial'])==cells
        assert e['initial'][0]==e['initial'][cells//2-1]
        assert e['initial'][cells//2]==e['initial'][-1]
        assert e['initial'][0]!=e['initial'][-1]


def test_invalid_matrix_and_missing_compiler_do_not_create_fake_results(tmp_path):
    with pytest.raises(ValueError):bench.request_for('unknown')
    with pytest.raises(ValueError):qualification.contact_request(3)
    with pytest.raises(ValueError):bench.run(tmp_path/'out',omc=tmp_path/'missing',library=tmp_path)
    assert not (tmp_path/'out').exists()
