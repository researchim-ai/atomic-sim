from copy import deepcopy
import csv
import math
import pytest
from atomic_model_hem_heated import HeatedHEMNetwork
from scripts import benchmark_modelica_heated as bench
from scripts import verify_heated_hllc as heated

@pytest.fixture(scope='module')
def native():return heated.capture(heated.request_for('wall_relaxation',heated.CAPS[-1]),'hllc-local','wall_relaxation')


def encode(native):
    rows=[]
    for s in native['observations']:
        row=dict(time=s['time_s'],externalEntropy=s['global_ledger']['external_entropy_j_k'],thermalProduction=s['global_ledger']['thermal_entropy_production_j_k'])
        for i,c in enumerate(s['cells'],1):
            for field,col in bench.network.COLUMNS.items():row[f'{col}[{i}]']=c[field]
            row[f'fluidHeat[{i}]']=0.0  # This fixture has wall-only heat and zero prescribed fluid power.
        for i,w in enumerate(s['walls'],1):
            for field,col in [('temperature_k','wallTemperature'),('heat_to_fluid_j','wallHeat'),('external_heat_j','wallExternal')]:row[f'{col}[{i}]']=w[field]
        for col,key in [('faceMass','integrated_face_mass_kg'),('faceEnergy','integrated_face_energy_j'),('support','node_support_impulse_kg_m_s')]:
            for i,value in enumerate(s['local_ledgers'][key],1):row[f'{col}[{i}]']=value
        rows.append(row)
    return rows


def save(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)

@pytest.mark.parametrize('case',heated.CASES)
def test_modelica_emits_four_stationary_nodes_and_conservative_sources(case):
    request=heated.request_for(case,heated.CAPS[-1]);packet=dict(request=request,observations=[HeatedHEMNetwork(request).snapshot()])
    text=bench.render(packet,case)
    assert text.count('if i<=4 then')==2 and 'if i<=2 then' not in text
    assert 'capacity[j]*der(wallTemperature[j])=wallInput[j]-exchange[j]' in text
    assert 'net[3,i]+fluidPower[i]+sum(if port[j]==i then exchange[j] else 0' in text
    assert 'der(externalEntropy)' in text and 'der(thermalProduction)' in text
    assert ('time>=0.0001 then -5000' in text)==(case=='branched_pulse')
    assert 'external[nw]' not in text


def test_unknown_render_and_template_refused(native):
    with pytest.raises(ValueError):bench.render(native,'unknown')
    with pytest.raises(ValueError):bench.replace_once('a a','a','b')
    with pytest.raises(ValueError):bench.replace_once('x','a','b')


def test_same_data_compare_and_csv_roundtrip(native,tmp_path):
    rows=encode(native);path=tmp_path/'trace.csv';save(path,rows)
    loaded=bench.load(path,native)
    assert loaded==rows and bench.compare(native,rows)['passed']

@pytest.mark.parametrize('damage',['missing_time','nan','unordered','ambiguous'])
def test_trace_reader_rejects_incomplete_or_invalid_evidence(native,tmp_path,damage):
    rows=encode(native)
    if damage=='missing_time':rows.pop(1)
    elif damage=='nan':rows[1]['wallTemperature[1]']=math.nan
    elif damage=='unordered':rows[1],rows[2]=rows[2],rows[1]
    else:
        other=deepcopy(rows[1]);other['wallHeat[1]']+=1.;rows.insert(2,other)
    path=tmp_path/'trace.csv';save(path,rows)
    with pytest.raises(ValueError):bench.load(path,native)


def test_exact_grid_sample_distinct_from_later_event_timestamp(native,tmp_path):
    rows=encode(native);other=deepcopy(rows[1]);other['time']+=1e-15;other['wallHeat[1]']+=1e-8
    rows.insert(2,other);path=tmp_path/'event.csv';save(path,rows)
    assert bench.load(path,native)==encode(native)

@pytest.mark.parametrize('damage',['missing','pressure','wall','face','energy','entropy'])
def test_comparison_fails_for_bad_external_results(native,damage):
    rows=encode(native)
    if damage=='missing':
        with pytest.raises(ValueError):bench.compare(native,[])
        return
    if damage=='pressure':rows[-1]['pressure[1]']+=1.
    elif damage=='wall':rows[-1]['wallTemperature[1]']+=.001
    elif damage=='face':rows[-1]['faceMass[1]']+=.001
    elif damage=='energy':rows[-1]['energy[1]']+=1.
    else:rows[-1]['externalEntropy']+=.001
    assert not bench.compare(native,rows)['passed']
