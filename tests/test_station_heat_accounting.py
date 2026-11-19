from copy import deepcopy
import pytest
from atomic_model_station_env import StationEnv,EnvConfig
from atomic_model_station_heat_accounting import read_heat,assess_heat,run_heat_window


def sample(t,heat=100.,cooling=60.,pump=0.,gross=40.,own=10.,stored_rate=10.):
    export=gross-own;boundary=heat-cooling-pump-export
    return dict(energy=dict(time_s=t,counters=dict(gross_generated_j=gross*t,own_needs_j=own*t,
        net_generated_j=export*t,net_exported_j=export*t,backup_used_j=0.),
        powers=dict(gross_electrical_w=gross,own_needs_w=own,net_generation_w=export,net_export_w=export),
        grid_available=True,trip=False,energy_relative=0.),stored_j=1000.+stored_rate*t,boundary_j=boundary*t,
        fluxes=dict(nuclear_heat_w=heat,primary_pump_loss_out_w=pump,external_cooling_heat_out_w=cooling,external_net_w=boundary))


def test_constant_heat_independent_balance_and_ratios():
    r=assess_heat([sample(0.),sample(2.)])
    assert r['passed'] and r['sampled_integrals_j']['nuclear_heat_w']==200.
    assert r['nuclear_heat_from_boundary_j']==200.
    assert r['finite_window_net_electric_to_nuclear_heat_ratio']==.3
    assert r['finite_window_gross_electric_to_nuclear_heat_ratio']==.4
    assert r['storage_change_to_nuclear_heat']==.1
    assert not r['stationary_efficiency_certified']


def test_quadratic_heat_converges_at_second_order_and_retains_defect():
    def curve(n):
        rows=[]
        for i in range(n+1):
            t=i/n;s=sample(t)
            s['fluxes']['nuclear_heat_w']=100.+3*t*t
            s['fluxes']['external_net_w']=10.+3*t*t
            s['boundary_j']=10*t+t**3;s['stored_j']=1000.+s['boundary_j']
            rows.append(s)
        return assess_heat(rows)
    coarse,fine=curve(4),curve(8)
    assert not coarse['passed'] and 'sampled_heat_closure' in coarse['reasons']
    assert coarse['heat_closure_j']==pytest.approx(4*fine['heat_closure_j'])
    assert coarse['finite_window_net_electric_to_nuclear_heat_ratio'] is None
    assert fine['sampled_integrals_j']['nuclear_heat_w']-101==pytest.approx(1/(2*8**2))


def test_storage_release_can_make_ratio_above_one_without_efficiency_claim():
    r=assess_heat([sample(0.,heat=10.,cooling=0.,stored_rate=-20.),sample(1.,heat=10.,cooling=0.,stored_rate=-20.)])
    assert r['passed'] and r['finite_window_net_electric_to_nuclear_heat_ratio']==3.
    assert r['storage_change_to_nuclear_heat']==-2. and not r['stationary_efficiency_certified']


def test_zero_heat_gives_no_ratio():
    r=assess_heat([sample(0.,heat=0.,cooling=0.,stored_rate=-30.),sample(1.,heat=0.,cooling=0.,stored_rate=-30.)])
    assert r['passed'] and r['finite_window_net_electric_to_nuclear_heat_ratio'] is None


@pytest.mark.parametrize('field',['stored_j','boundary_j'])
def test_inconsistent_integrated_or_stored_energy_refused(field):
    a,b=sample(0.),sample(1.);b[field]+=1.
    r=assess_heat([a,b]);assert not r['passed'] and r['finite_window_net_electric_to_nuclear_heat_ratio'] is None


def test_wrong_flux_sign_or_missing_flux_is_not_hidden():
    a,b=sample(0.),sample(1.);b['fluxes']['external_cooling_heat_out_w']=-60.
    assert 'instantaneous_boundary_identity' in assess_heat([a,b])['reasons']
    del b['fluxes']['nuclear_heat_w']
    with pytest.raises(ValueError):assess_heat([a,b])


def test_empty_window_is_not_qualified():
    assert not assess_heat([sample(0.)])['passed']


def test_privileged_sampling_does_not_modify_station():
    env=StationEnv(EnvConfig(max_steps=2,tick_s=.01,automatic=False));env.reset();before=env.checkpoint()
    sample=read_heat(env)
    assert env.checkpoint()==before
    assert sample['fluxes']['nuclear_heat_w']>0 and sample['stored_j']>0


def test_real_window_and_restricted_reference_scope():
    r=run_heat_window(EnvConfig(max_steps=3,tick_s=.01,automatic=False))
    assert r['complete'] and r['passed'] and len(r['samples'])==4
    assert not r['industrial_thermal_efficiency_claimed']
    with pytest.raises(ValueError):run_heat_window(EnvConfig(max_steps=1,tick_s=.01,automatic=True))


def test_incomplete_refinement_cannot_publish_success(tmp_path,monkeypatch):
    from scripts import qualify_station_heat_accounting as driver
    from scripts.qualify_station_inventory import read
    monkeypatch.setattr(driver,'run_heat_window',lambda config:dict(complete=False,passed=False,accounting=None,failure={'reason':'numerical refusal'}))
    out=tmp_path/'study'
    with pytest.raises(ValueError,match='qualification failed'):
        driver.run(out)
    r=read(out/'report.json')
    assert not r['passed'] and r['completed_paths']==0 and r['comparisons']=={}
    assert len(list(out.glob('*.json')))==5


def test_source_change_prevents_refinement_receipt(tmp_path,monkeypatch):
    from scripts import qualify_station_heat_accounting as driver
    versions=iter([{'v':1},{'v':2}])
    monkeypatch.setattr(driver,'identity',lambda:next(versions))
    monkeypatch.setattr(driver,'run_heat_window',lambda config:dict(complete=False,passed=False,accounting=None))
    with pytest.raises(ValueError,match='source changed'):
        driver.run(tmp_path/'study')
    assert not (tmp_path/'study/report.json').exists()
