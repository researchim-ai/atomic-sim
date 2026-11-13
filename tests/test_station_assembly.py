import json
import numpy as np
import pytest
from atomic_model_station import (StationConfig,StationResidual,StationSession,Inventory,Boundary,PowerBlock,
                                  AffineAssembly,build_r0,AssembledStationSession)


def test_compiled_operators_match_separate_monolithic_r0():
    c=StationConfig()
    for power,connected in [(0.,True),(1.05e6,True),(1e6,False)]:
        a,b=build_r0(c,power,connected).operators();old_a,old_b=StationResidual(c).operators(power,connected)
        np.testing.assert_allclose(a,old_a,rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(b,old_b,rtol=1e-14,atol=1e-14)


def test_assembled_trajectory_and_serialized_restart():
    old=StationSession();s=AssembledStationSession()
    for duration,power,connected in [(20.,1.05e6,True),(5.,1e6,False),(30.,1e6,True)]:
        a=s.advance(duration,source_power_w=power,generator_connected=connected)
        b=old.advance(duration,source_power_w=power,generator_connected=connected)
        np.testing.assert_allclose(a['state'],b['state'],rtol=1e-11)
    restored=AssembledStationSession.restore(json.loads(json.dumps(s.checkpoint())))
    assert restored.advance(1.)==s.advance(1.)
    with pytest.raises(ValueError):StationSession.restore(s.checkpoint())


def test_global_boundary_power_and_owned_state_manifest():
    c=StationConfig();g=build_r0(c,1e6,True);y,_=g.steady([1.,1.,1.,1.,1e6]);a,b=g.operators()
    capacities=np.array([c.primary_capacity_j_k,c.wall_capacity_j_k,c.secondary_capacity_j_k,c.cold_capacity_j_k,1.])
    assert abs(capacities@(a@y+b)+sum(g.boundary_power(x,y) for x in ('heat_source','ambient','grid')))<1e-7
    m=g.manifest();assert len(m['states'])==5 and len(m['connections'])==16
    m['states'][0]['owner']='damaged';assert g.manifest()['states'][0]['owner']=='primary'
    a[:]=0.;assert np.any(g.operators()[0]!=0.)


def fixture():
    i=[Inventory('store','temperature','thermal',10.,('in',))]
    b=[Boundary('outside','thermal')]
    law=PowerBlock('link',{'body':('thermal',{'temperature':-1.},300.),'outside':('thermal',{'temperature':1.},-300.)})
    links=[(('link','body'),('store','in')),(('link','outside'),('outside','port'))]
    return i,b,[law],links


def test_independent_one_inventory_assembly_and_initialization():
    g=AffineAssembly(*fixture());y,res=g.steady([1.])
    np.testing.assert_allclose(y,[300.]);assert res<1e-8
    np.testing.assert_allclose(g.residual([310.],[-1.]),[0.])


@pytest.mark.parametrize('damage',['owners','states','dangling','wrong_family','hidden_state'])
def test_invalid_assembly_rejected(damage):
    i,b,l,links=fixture()
    if damage=='owners':b=[Boundary('store','thermal')]
    elif damage=='states':i.append(Inventory('second','temperature','thermal',1.,('in',)))
    elif damage=='dangling':links.pop()
    elif damage=='wrong_family':b=[Boundary('outside','electrical_power')]
    else:l=[PowerBlock('link',{'body':('thermal',{'unknown':-1.},0.),'outside':('thermal',{'unknown':1.},0.)})]
    with pytest.raises(ValueError):AffineAssembly(i,b,l,links)


def test_nonconservative_and_nonfinite_blocks_refused():
    with pytest.raises(ValueError):PowerBlock('x',{'out':('thermal',{},1.)})
    with pytest.raises(ValueError):PowerBlock('x',{'out':('thermal',{},float('nan'))})


def test_rank_deficiency_not_silently_initialized():
    i,b,l,links=fixture();l=[PowerBlock('link',{'body':('thermal',{},0.),'outside':('thermal',{},0.)})]
    with pytest.raises(ValueError):AffineAssembly(i,b,l,links).steady([1.])


def test_assembled_domain_failure_rolls_back_everything():
    s=AssembledStationSession(StationConfig(rotor_inertia_kg_m2=100.));before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(60.,generator_connected=False)
    assert s.checkpoint()==before


def test_scheduled_events_match_explicit_segments_and_json_restart():
    a=AssembledStationSession();b=AssembledStationSession()
    events=[dict(at_s=0.,commands={'source_power_w':1.05e6}),
            dict(at_s=3.,commands={'generator_connected':False}),dict(at_s=5.,commands={'generator_connected':True})]
    result=a.advance_schedule(10.,events)
    b.advance(3.,source_power_w=1.05e6);b.advance(2.,generator_connected=False);b.advance(5.,generator_connected=True)
    assert a.checkpoint()==b.checkpoint() and len(result['segments'])==3
    assert [r['time_s'] for r in a.snapshot()['command_transitions']]==[0.,3.,5.]
    assert AssembledStationSession.restore(json.loads(json.dumps(a.checkpoint()))).snapshot()==a.snapshot()


@pytest.mark.parametrize('events',[
    [dict(at_s=1.,commands={'generator_connected':False}),dict(at_s=1.,commands={'generator_connected':True})],
    [dict(at_s=10.,commands={'generator_connected':False})],
    [dict(at_s=True,commands={'generator_connected':False})],
    [dict(at_s=1.,commands={'unknown':1.})],
    [dict(at_s=1.,commands={'source_power_w':float('nan')})],
])
def test_invalid_schedule_is_atomic(events):
    s=AssembledStationSession();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance_schedule(10.,events)
    assert s.checkpoint()==before


def test_failure_after_earlier_schedule_segments_rolls_back_whole_call():
    s=AssembledStationSession(StationConfig(rotor_inertia_kg_m2=100.));before=s.checkpoint()
    events=[dict(at_s=0.,commands={'source_power_w':1.05e6}),dict(at_s=1.,commands={'generator_connected':False})]
    with pytest.raises(ValueError):s.advance_schedule(60.,events)
    assert s.checkpoint()==before


def test_component_and_connection_order_has_exact_canonical_operators():
    inventory=[Inventory('store','t','thermal',10.,('a','b','c'))]
    boundaries=[Boundary('outside_'+k,'thermal') for k in ('a','b','c')]
    blocks=[];links=[]
    for label,coefficient in [('a',.1),('b',.2),('c',.3)]:
        blocks.append(PowerBlock(label,{'body':('thermal',{'t':-coefficient},300.*coefficient),
                                        'outside':('thermal',{'t':coefficient},-300.*coefficient)}))
        links.extend([((label,'body'),('store',label)),((label,'outside'),('outside_'+label,'port'))])
    a=AffineAssembly(inventory,boundaries,blocks,links)
    b=AffineAssembly(inventory,boundaries,list(reversed(blocks)),list(reversed(links)))
    assert a.manifest()==b.manifest()
    for x,y in zip(a.operators(),b.operators(),strict=True):np.testing.assert_array_equal(x,y)
