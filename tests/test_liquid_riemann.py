"""Physical wave invariants, independent acoustic limit and oracle refusals."""
from dataclasses import replace
import math
import numpy as np
import pytest
from atomic_model_hem_qualified import eos
from scripts.liquid_riemann import solve, WaveCurve

BRACKET=(1e6,15e6)

@pytest.mark.parametrize('velocity',[-20.,0.,20.])
def test_exact_material_contact(velocity):
    left,right=eos.pt(4e6,441.),eos.pt(4e6,439.)
    result=solve(left,right,velocity,velocity,pressure_bracket=BRACKET)
    assert result.pressure_pa==4e6 and result.velocity_m_s==velocity
    assert result.left==left and result.right==right
    assert result.left_wave==result.right_wave=='contact'
    rho=(left if velocity>=0 else right).density_kg_m3
    assert result.interface_flux()[0]==rho*velocity

@pytest.mark.parametrize('pressures',[(4.02e6,3.98e6),(14e6,2e6),(2e6,14e6)])
def test_reflection_and_wave_laws(pressures):
    left,right=eos.pt(pressures[0],441.),eos.pt(pressures[1],439.)
    a=solve(left,right,pressure_bracket=BRACKET)
    b=solve(right,left,pressure_bracket=BRACKET)
    assert abs(a.pressure_pa-b.pressure_pa)<1e-5
    assert abs(a.velocity_m_s+b.velocity_m_s)<1e-10
    np.testing.assert_allclose(b.interface_flux(),np.array(a.interface_flux())*[-1,1,-1],rtol=1e-10,atol=1e-6)
    for initial,star,kind in [(left,a.left,a.left_wave),(right,a.right,a.right_wave)]:
        if kind=='rarefaction':
            assert abs(initial.entropy_j_kg_k-star.entropy_j_kg_k)<2e-8
        else:
            assert kind=='shock'
            assert star.entropy_j_kg_k>=initial.entropy_j_kg_k-2e-8
            v0=1/initial.density_kg_m3;v=1/star.density_kg_m3
            assert abs(star.internal_energy_j_kg-initial.internal_energy_j_kg+.5*(star.pressure_pa+initial.pressure_pa)*(v-v0))<2e-6
            # Independent three Rankine–Hugoniot flux jumps in the shock frame.
            sign=-1 if initial==left else 1
            speed=sign*math.sqrt((a.pressure_pa-initial.pressure_pa)/(v0-v))/initial.density_kg_m3
            j0=initial.density_kg_m3*(-speed);j1=star.density_kg_m3*(a.velocity_m_s-speed)
            assert abs(j0-j1)<2e-5
            assert abs(initial.pressure_pa+j0*(-speed)-a.pressure_pa-j1*(a.velocity_m_s-speed))<.05
            assert abs(initial.enthalpy_j_kg+.5*speed**2-star.enthalpy_j_kg-.5*(a.velocity_m_s-speed)**2)<2e-5

@pytest.mark.parametrize('velocity,kind',[(-1.,'rarefaction'),(1.,'shock')])
def test_symmetric_expansion_and_compression(velocity,kind):
    state=eos.pt(8e6,440.)
    result=solve(state,state,velocity,-velocity,pressure_bracket=BRACKET)
    assert abs(result.velocity_m_s)<1e-9
    assert result.left_wave==result.right_wave==kind
    assert (result.pressure_pa>8e6)==(velocity>0)

@pytest.mark.parametrize('pressure,temperature',[(1e6,350.),(4e6,440.),(14e6,550.)])
def test_small_wave_matches_linear_acoustic_impedance(pressure,temperature):
    state=eos.pt(pressure,temperature);impedance=state.density_kg_m3*eos.characteristic_speeds(state)[1]
    # Pressure perturbation follows an isentrope, not an isothermal shortcut.
    delta=100.
    left=WaveCurve(state).isentrope(pressure+delta)
    right=WaveCurve(state).isentrope(pressure-delta)
    result=solve(left,right,pressure_bracket=(pressure-1000.,pressure+1000.))
    assert abs(result.velocity_m_s/(delta/impedance)-1)<2e-5
    assert abs(result.pressure_pa-pressure)<.02

@pytest.mark.parametrize('damage',['nan','bracket','unbracketed','vapor','inconsistent'])
def test_refuses_unqualified_input(damage):
    l=r=eos.pt(4e6,440.);vl=0.;bounds=BRACKET
    if damage=='nan':vl=float('nan')
    elif damage=='bracket':bounds=(15e6,1e6)
    elif damage=='unbracketed':bounds=(5e6,6e6)
    elif damage=='vapor':r=eos.pt(1e6,600.)
    else:r=replace(r,density_kg_m3=r.density_kg_m3+1)
    with pytest.raises(ValueError):solve(l,r,vl,pressure_bracket=bounds)


def test_supersonic_interface_requires_separate_sampler():
    state=eos.pt(4e6,440.)
    solution=solve(state,state,2000.,2000.,pressure_bracket=BRACKET)
    with pytest.raises(ValueError,match='Interface not between'):solution.interface_flux()


def test_saturation_crossing_is_refused():
    curve=WaveCurve(eos.pt(14e6,600.))
    with pytest.raises(ValueError,match='No admissible liquid'):curve.isentrope(1e6)


def test_oracle_does_not_import_approximate_flux_or_time_stepper():
    import ast
    from pathlib import Path
    from scripts import liquid_riemann
    tree=ast.parse(Path(liquid_riemann.__file__).read_text())
    imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any('hem_contact' in (m or '') for m in imports)
    assert 'scipy.integrate' in imports and 'scipy.optimize' in imports


def test_benchmark_reports_errors_without_claiming_hllc_accuracy(tmp_path):
    from scripts.benchmark_liquid_riemann import run
    from atomic_model_multiphysics.provenance import unseal
    import json
    result=run(tmp_path/'benchmark')
    assert result['passed'] and result['oracle_cases']==15
    assert not result['validated'] and not result['local_envelope_graph_qualified']
    assert not result['hllc_accuracy_gate']
    assert unseal(json.loads((tmp_path/'benchmark'/'report.json').read_text()))==result
    row=next(r for r in result['cases'] if r['case']=='wide')
    assert .4<row['candidates'][0]['relative_mass_flux_error']<.44
    assert row['candidates'][1]['relative_mass_flux_error']<.03
    with pytest.raises(FileExistsError):run(tmp_path/'benchmark')


def test_benchmark_rejects_symlink_output(tmp_path):
    from scripts.benchmark_liquid_riemann import run
    target=tmp_path/'target';target.mkdir()
    link=tmp_path/'link';link.symlink_to(target,target_is_directory=True)
    with pytest.raises(ValueError,match='Symlink'):run(link/'out')
    assert not (target/'out').exists()
