"""Independent wall balance and invalid-data admission checks."""
import pytest
from scripts import verify_phase_wall as bench
from scripts.phase_wall_case import execute


def test_forward_energy_reference_matches_actual_vapor_exit_and_restart():
    native=execute('vapor_exit',diameter_m=.08)
    refs=bench.reference('vapor_exit',[0.,.0025,.005,.0075,.01])
    assert native['record']['passed']
    assert {r['method'] for r in refs}=={'DOP853','Radau'}
    for ref in refs:
        errors=bench.errors(native['observations'],ref['observations'])
        assert all(errors[k]<=bench.BUDGETS[k] for k in errors)


@pytest.mark.parametrize('damage',['nan','prefix','time','nan_time','empty_cells'])
def test_wall_comparison_rejects_corrupt_states(damage):
    obs=[dict(time_s=0.,cells=[dict(pressure_pa=1e6,temperature_k=450.,internal_energy_j_kg=1e6)],
              walls=[dict(temperature_k=460.)])]
    ref=[dict(time_s=0.,pressure_pa=1e6,temperature_k=450.,specific_energy_j_kg=1e6,wall_temperature_k=460.)]
    if damage=='nan':obs[0]['walls'][0]['temperature_k']=float('nan')
    elif damage=='prefix':ref=[]
    elif damage=='time':ref[0]['time_s']=1.
    elif damage=='nan_time':ref[0]['time_s']=float('nan')
    else:obs[0]['cells']=[];obs[0]['walls']=[]
    with pytest.raises(ValueError):bench.errors(obs,ref)


def test_four_cell_mesh_uses_same_fixed_geometry_and_valid_wall_capacities():
    result=execute('vapor_entry',cells=4,diameter_m=.08)
    assert result['record']['passed']
    assert all(e['diameter_m']==.08 for e in result['request']['model']['edges'])
    assert min(w['capacity_j_k'] for w in result['request']['model']['walls'])>=100.
