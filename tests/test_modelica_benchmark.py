"""Native CSV admission and model export; fixtures are not peer measurements."""
from copy import deepcopy
import csv
import math
import pytest
from scripts import benchmark_modelica_cycles as bench
from scripts.phase_wall_case import execute as wall_execute
from scripts.verify_phase_cycles import execute


def external_rows():
    return [dict(time=t,pressure=1e6,temperature=450.,mass=1.,energy=1e6,specificEnergy=1e6,phase=2.) for t in bench.TIMES]


def write_csv(path,rows):
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def test_complete_csv_and_consistent_event_duplicates_admitted(tmp_path):
    rows=external_rows();rows.insert(3,dict(rows[2]))
    path=tmp_path/'native.csv';write_csv(path,rows)
    assert len(bench.load_csv(path))==5


@pytest.mark.parametrize('damage',['prefix','nan','ordering','domain','event','phase'])
def test_native_csv_cannot_promote_incomplete_or_inconsistent_output(tmp_path,damage):
    rows=external_rows()
    if damage=='prefix':rows.pop()
    elif damage=='nan':rows[2]['energy']=float('nan')
    elif damage=='ordering':rows.reverse()
    elif damage=='domain':rows[2]['pressure']=20e6
    elif damage=='phase':rows[2]['phase']=7.
    else:
        duplicate=dict(rows[2]);duplicate['energy']+=10*bench.BUDGETS['energy_j'];rows.insert(3,duplicate)
    path=tmp_path/'native.csv';write_csv(path,rows)
    with pytest.raises(ValueError):bench.load_csv(path)


def test_native_model_exports_actual_volume_balances_and_declared_sources():
    result=execute('vapor_exit',2,2e-5)
    initial=result['observations'][0]['cells'][0]
    text=bench.render_model(initial,-50.)
    assert 'Modelica.Fluid.Vessels.ClosedVolume' in text
    assert 'volume.m=' in text and 'volume.U=' in text and 'connect(heater.port' in text
    assert 'WaterIF97_base' in text and 'IndependentVariables.dTX' in text
    bad=deepcopy(initial);bad['mass_kg']=float('nan')
    with pytest.raises(ValueError):bench.render_model(bad,-50.)


def test_explicit_stock_state_selection_changes_only_solver_coordinate_contract():
    result=wall_execute('vapor_exit')
    initial=result['observations'][0]['cells'][0];wall=result['request']['model']['walls'][0]
    text=bench.render_model(initial,-50.,wall=wall,stock_states=True)
    assert 'preferredMediumStates=false' in text and 'u(stateSelect=StateSelect.always)' in text
    assert 'volume.m=' in text and 'volume.U=' in text
    assert 'HeatCapacitor' in text and 'WaterIF97_base' in text
    assert 'preferredMediumStates=false' not in bench.render_model(initial,-50.,wall=wall)


def test_empty_wall_only_campaign_refused_before_output(tmp_path):
    with pytest.raises(ValueError):bench.run(tmp_path/'out',omc=tmp_path/'missing',library=tmp_path,wall_only=True)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('case',['liquid_entry','liquid_exit','vapor_entry','vapor_exit'])
def test_actual_closed_wall_exchange_crosses_phase_and_restarts(case):
    result=wall_execute(case)
    assert result['record']['passed'],result['record']
    text=bench.render_model(result['observations'][0]['cells'][0],-50.,wall=result['request']['model']['walls'][0])
    assert 'HeatCapacitor' in text and 'ThermalConductor' in text
    assert 'heater.Q_flow' not in text and 'PrescribedHeatFlow' not in text
    assert 'V=0.02' in text


def test_common_grid_required_for_comparison():
    with pytest.raises(ValueError):bench.compare([],external_rows())


def test_nonfinite_comparison_time_cannot_pass_tolerance_check():
    native=[dict(time_s=float('nan'))]*5
    with pytest.raises(ValueError):bench.compare(native,external_rows())


@pytest.mark.parametrize('damage',['phase','wall'])
def test_duplicate_event_cannot_hide_phase_or_wall_disagreement(tmp_path,damage):
    rows=external_rows()
    for row in rows:row['wallTemperature']=460.
    duplicate=dict(rows[2])
    duplicate['phase' if damage=='phase' else 'wallTemperature']=1. if damage=='phase' else 470.
    rows.insert(3,duplicate)
    path=tmp_path/'native.csv';write_csv(path,rows)
    with pytest.raises(ValueError):bench.load_csv(path,wall=True)


@pytest.mark.parametrize('side',['atomic','peer'])
def test_nonfinite_wall_cannot_disappear_from_maximum_error(side):
    result=wall_execute('vapor_exit')
    rows=[]
    for t,obs in zip(bench.WALL_TIMES,result['observations'],strict=True):
        c=obs['cells'][0]
        rows.append(dict(time=t,pressure=c['pressure_pa'],temperature=c['temperature_k'],
                         specificEnergy=c['internal_energy_j_kg'],mass=c['mass_kg'],energy=c['total_energy_j'],
                         wallTemperature=obs['walls'][0]['temperature_k']))
    if side=='atomic':result['observations'][2]['walls'][0]['temperature_k']=float('nan')
    else:rows[2]['wallTemperature']=float('nan')
    with pytest.raises(ValueError):bench.compare(result['observations'],rows,times=bench.WALL_TIMES,wall=True)


@pytest.mark.parametrize('tolerance',[0.,float('nan'),1e-3,1e-15])
def test_invalid_precision_refused_before_output(tmp_path,tolerance):
    with pytest.raises(ValueError):bench.run(tmp_path/'refused',omc=tmp_path/'omc',library=tmp_path, tolerance=tolerance)
    assert not (tmp_path/'refused').exists()
