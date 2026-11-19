import copy
import json
import numpy as np
import pytest
from scipy.special import jv, jn_zeros

from atomic_model_multiphysics.provenance import seal
from atomic_model_station_thermal_resolved import resolved_velocity_mode,resolved_scalar_mode,ResolvedThermalSystem,resolve_thermal,compare_thermal
from atomic_model_station_thermal_path import ThermalTrajectory,TrajectoryAccuracyError


def system(order=8):
    m=resolved_velocity_mode(0,0,.628319,order=order)
    s=resolved_scalar_mode(0,0,.628319,order=order)
    return ResolvedThermalSystem([(m,np.zeros(m.dimension))],[(s,s.fit(.1*jv(0,jn_zeros(0,1)[0]*s.radius)))])


def test_checkpoint_and_next_boundary_continuation_are_exact():
    a=ThermalTrajectory(system(),max_segment=.1)
    a.advance_to(.2)
    checkpoint=json.loads(json.dumps(a.checkpoint()))
    b=ThermalTrajectory.restore(checkpoint)
    assert b.checkpoint()==checkpoint
    a.advance_to(.4);b.advance_to(.4)
    assert a.checkpoint()==b.checkpoint()
    assert a.status()['origin_time']==0 and a.status()['accepted_segments']==4


def test_segmented_solution_matches_uninterrupted_ode_within_tolerance():
    s=system();path=ThermalTrajectory(s,max_segment=.05)
    path.advance_to(.2)
    reference=resolve_thermal(s,end_time=.2)
    difference=compare_thermal(path.endpoint(),reference,s.radius,s.weights)
    assert max(difference['velocity_l2'],difference['temperature_l2'])<1e-8
    status=path.status()
    assert status['sampled_global_defect_upper_bounds'][2]<1e-10
    assert not status['continuous_error_bound'] and not status['bdf_history_serialized']


def test_two_locally_passing_heat_defects_fail_global_gate_atomically(monkeypatch):
    from atomic_model_station_thermal_path import session as module
    original=module.resolve_thermal
    def biased(*args,**kwargs):
        result=original(*args,**kwargs)
        result.report['integrated_rates'][5]+=8e-11
        result.report['maximum_physical_heat_budget_residual']=8.1e-11
        return result
    monkeypatch.setattr(module,'resolve_thermal',biased)
    path=ThermalTrajectory(system(),max_segment=.1);before=path.checkpoint()
    with pytest.raises(TrajectoryAccuracyError) as error:path.advance_to(.2)
    assert error.value.receipt['sampled_global_defect_upper_bounds'][2]>1e-10
    assert path.checkpoint()==before and path.time==0


def test_partial_interval_exhaustion_is_atomic():
    path=ThermalTrajectory(system(),max_segment=.05);before=path.checkpoint()
    with pytest.raises(TrajectoryAccuracyError):path.advance_to(.2,max_segments=1)
    assert path.checkpoint()==before


def test_backend_failure_is_atomic(monkeypatch):
    from atomic_model_station_thermal_path import session as module
    path=ThermalTrajectory(system());before=path.checkpoint()
    def fail(*args,**kwargs):raise ValueError('injected solver failure')
    monkeypatch.setattr(module,'resolve_thermal',fail)
    with pytest.raises(ValueError,match='injected'):path.advance_to(.1)
    assert path.checkpoint()==before


def test_decimal_endpoint_is_solved_at_target_without_extra_tiny_segment():
    path=ThermalTrajectory(system(),max_segment=.1)
    report=path.advance_to(1.,max_segments=10)
    assert report['accepted_segments']==10 and path.time==1.
    assert path.segments[-1]['time']==1.


def test_sources_and_global_ledgers_survive_restore():
    m=resolved_velocity_mode(0,0,.628319,order=4)
    s=resolved_scalar_mode(0,0,.628319,order=4);r=s.radius
    u=np.zeros((3,len(r)),complex);u[2]=.2*.1/(16*.05)*(3-4*r*r+r**4)
    model=ResolvedThermalSystem([(m,m.fit(u))],[(s,s.fit(.1*(1-r*r)))],sources={(0,0):[.012]})
    path=ThermalTrajectory(model);path.advance_to(.2)
    restored=ThermalTrajectory.restore(path.checkpoint());restored.advance_to(.3)
    status=restored.status();integrals=status['integrated_rates']
    assert integrals[4]==pytest.approx(.006*.3,abs=1e-12)
    assert integrals[5]==pytest.approx(integrals[4],abs=1e-12)
    assert max(status['sampled_global_defect_upper_bounds'])<1e-10
    assert restored.audit()['maximum_wall_thermal_residual']<1e-10


@pytest.mark.parametrize('change',[dict(max_segment=0),dict(max_segment=True),dict(rtol=0),dict(atol=float('nan')),dict(heat_tolerance=0),dict(wall_tolerance=-1)])
def test_invalid_policy_rejected(change):
    with pytest.raises(ValueError):ThermalTrajectory(system(),**change)


@pytest.mark.parametrize('target',[0.,-1.,float('nan'),True])
def test_invalid_targets_preserve_state(target):
    path=ThermalTrajectory(system());before=path.checkpoint()
    with pytest.raises(ValueError):path.advance_to(target)
    assert path.checkpoint()==before


@pytest.mark.parametrize('name',['identity','integrated_rates','global_bounds','history','current','policy'])
def test_resealed_checkpoint_inconsistencies_are_rejected(name):
    path=ThermalTrajectory(system());path.advance_to(.1)
    p=copy.deepcopy(path.checkpoint()['payload'])
    if name=='identity':p[name]['python']='different'
    elif name=='integrated_rates':p[name][0]+=.1
    elif name=='global_bounds':p[name][2]+=.1
    elif name=='history':p[name][0]['start_time']=.01
    elif name=='current':p[name]['real'][0]+=.1
    elif name=='policy':p[name]['rtol']=0
    with pytest.raises(ValueError):ThermalTrajectory.restore(seal(p))


def test_envelope_and_unknown_fields_are_rejected():
    checkpoint=ThermalTrajectory(system()).checkpoint()
    broken=copy.deepcopy(checkpoint);broken['payload']['time']=.1
    with pytest.raises(ValueError):ThermalTrajectory.restore(broken)
    p=copy.deepcopy(checkpoint['payload']);p['unknown']=1
    with pytest.raises(ValueError):ThermalTrajectory.restore(seal(p))


def test_returned_checkpoint_history_and_endpoint_are_detached():
    path=ThermalTrajectory(system());path.advance_to(.1);original=path.checkpoint()
    checkpoint=path.checkpoint();checkpoint['payload']['identity']['python']='changed'
    history=path.segments;history[0]['integrals'][0]=100
    endpoint=path.endpoint();endpoint.temperature[0][1][:]=100
    status=path.status();status['integrated_rates'][0]=100
    assert path.checkpoint()==original


def test_coarse_physical_heat_failure_does_not_modify_session():
    path=ThermalTrajectory(system(3));before=path.checkpoint()
    with pytest.raises(TrajectoryAccuracyError):path.advance_to(.1)
    assert path.checkpoint()==before


def test_cumulative_bound_covers_all_recorded_endpoint_defects():
    path=ThermalTrajectory(system(),max_segment=.05)
    origin=path._origin_observables.copy();ledger=np.zeros(6)
    for t in (.05,.1,.15,.2):
        path.advance_to(t)
        status=path.status();bound=np.array(status['sampled_global_defect_upper_bounds'])
        actual=abs(np.array(status['endpoint_global_defects']))
        assert np.all(bound>=actual)
    assert path.status()['accepted_segments']==4


def test_complex_checkpoint_decode_preserves_signed_zero_bits():
    from atomic_model_station_thermal_path.session import _decode,_encode
    q=np.empty(2,complex);q.real=[-0.0,0.0];q.imag=[-0.0,-0.0]
    restored=_decode(json.loads(json.dumps(_encode(q))),2)
    assert restored.tobytes()==q.tobytes()
