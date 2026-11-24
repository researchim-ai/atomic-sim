from dataclasses import replace
import json

import numpy as np
import pytest
from scipy.linalg import expm
from core.status import DomainError
from atomic_model_multiphysics.provenance import seal,unseal
from coupling.conservative import ConservativeMap
from coupling.coolant_channel import ChannelConfig,CoolantChannel,Port,contact
from physics.hydraulics.network import BoundaryState


def test_nonmatching_contact_retains_opposite_heat_flows_with_zero_net_wall_power():
    exchange=ConservativeMap([0.,2.],[0.,1.,2.])
    r=contact(exchange,[550.],[540.,560.],[1000.])
    np.testing.assert_array_equal(r['coolant_heat_in_w'],[5000.,-5000.])
    np.testing.assert_array_equal(r['wall_heat_out_w'],[0.])
    assert r['entropy_w_k'].sum()>0
    # Simply mapping the net wall power would return [0,0] and lose both flows.
    assert not np.array_equal(exchange.integrals(r['wall_heat_out_w']),r['coolant_heat_in_w'])


def test_contact_is_pairwise_conservative_and_entropy_matches_reservoir_balances():
    op=ConservativeMap([0.,.7,2.2,4.],[0.,1.,2.,4.]);tw=np.array([545.,560.,550.]);tc=np.array([555.,548.,570.])
    r=contact(op,tw,tc,[100.,500.,1000.])
    assert r['wall_heat_out_w'].sum()==pytest.approx(r['coolant_heat_in_w'].sum(),abs=1e-10)
    entropy=r['coolant_heat_in_w']@(1/tc)-r['wall_heat_out_w']@(1/tw)
    assert r['entropy_w_k'].sum()==pytest.approx(entropy,abs=1e-12)
    assert np.min(r['entropy_w_k'])>=0


@pytest.mark.parametrize('wall,coolant,ua',[([0],[550],[1]),([True],[550],[1]),([550],[np.nan],[1]),([550],[550],[-1]),([550+1j],[550],[1]),([550],[550],[np.inf])])
def test_contact_refuses_invalid_physical_inputs(wall,coolant,ua):
    with pytest.raises((ValueError,DomainError)):contact(ConservativeMap([0,1],[0,1]),wall,coolant,ua)


@pytest.mark.parametrize('patch',[dict(faces_m=[0,4]),dict(faces_m=(0,1,1,4)),dict(wall_faces_m=(0,3)),
    dict(area_m2=True),dict(loss_coefficient=-1.),dict(initial_flow_kg_s=float('nan')),
    dict(initial_pressure_pa=1e6,initial_temperature_k=550.),dict(initial_wall_temperature_k=700.)])
def test_invalid_configurations_are_rejected(patch):
    with pytest.raises((ValueError,DomainError)):ChannelConfig(**patch)


@pytest.mark.parametrize('flow',[0.,5.,-5.])
def test_uniform_isothermal_channel_preserves_exact_rest_or_lossless_throughflow(flow):
    s=CoolantChannel(ChannelConfig(initial_flow_kg_s=flow,loss_coefficient=0.));initial=s.state.copy()
    r=s.advance(.005)
    np.testing.assert_allclose(s.state[:s.ledger_index],initial[:s.ledger_index],rtol=1e-10,atol=1e-7)
    assert max(r['balance_relative'])<1e-10
    assert r['terms']['conductive_entropy_w_k']==pytest.approx(0.,abs=1e-12)
    assert r['distributed_fluid_inventories'] and r['distributed_momentum'] and not r['pwr_connected']


@pytest.mark.parametrize('flow,inlet_fraction',[(5.,.0008),(-5.,.0002)])
def test_local_boron_transport_matches_exact_compartment_matrix_exponential(flow,inlet_fraction):
    c=ChannelConfig(initial_flow_kg_s=flow,loss_coefficient=0.)
    left=Port(boron_fraction=inlet_fraction) if flow>0 else Port()
    right=Port(boron_fraction=inlet_fraction) if flow<0 else Port()
    s=CoolantChannel(c,left=left,right=right);mass=s.state[:3*s.cells:3].copy()
    initial=s.state[2:3*s.cells:3].copy();order=list(range(s.cells))
    if flow<0:order.reverse()
    matrix=np.zeros((s.cells+1,s.cells+1));q=abs(flow)
    for k,i in enumerate(order):
        matrix[i,i]=-q/mass[i]
        if k==0:matrix[i,-1]=q*inlet_fraction
        else:matrix[i,order[k-1]]=q/mass[order[k-1]]
    expected=(expm(matrix*.01)@np.r_[initial,1.])[:-1]
    r=s.advance(.01)
    np.testing.assert_allclose(s.state[2:3*s.cells:3],expected,rtol=1e-8,atol=1e-11)
    assert max(r['balance_relative'])<1e-9
    assert r['terms']['ports']['left']['donor']==('boundary' if flow>0 else 'cell0')
    assert r['terms']['ports']['right']['donor']==(f'cell{s.cells-1}' if flow>0 else 'boundary')


def test_momentum_pressure_and_signed_friction_match_independent_face_equations():
    s=CoolantChannel(ChannelConfig(initial_flow_kg_s=-2.),left=Port(15.501e6),right=Port(15.499e6))
    s._rhs(0.,s.state)
    for i,(name,j) in enumerate(s.network.junctions.items()):
        up=s.network._node(j.upstream);down=s.network._node(j.downstream)
        donor=down
        rho=donor.density_kg_m3 if isinstance(donor,BoundaryState) else donor.thermo.rho_kg_m3
        expected=(up.pressure_pa-down.pressure_pa-j.loss_coefficient*(-2)*2/(2*rho*j.area_m2**2))*j.area_m2/j.length_m
        assert s.terms['face_acceleration_kg_s2'][i]==pytest.approx(expected,rel=1e-12,abs=1e-10)


def test_heating_drives_local_fluid_and_wall_energy_without_duplicate_storage():
    s=CoolantChannel(ChannelConfig(initial_wall_temperature_k=552.))
    r=s.advance(.003,[10000.,20000.,5000.])
    assert np.ptp(r['terms']['coolant_temperature_k'])>1e-5
    assert r['terms']['conductive_entropy_w_k']>0
    assert max(r['balance_relative'])<1e-9
    assert abs(r['terms']['instantaneous_residual'][1])<1e-6
    assert r==s.snapshot()


def test_identical_inventory_rhs_is_independent_of_previous_thermodynamic_probes():
    s=CoolantChannel();state=s.state.copy();expected=s._rhs(0.,state)
    probe=state.copy();probe[1]+=1000.;s._rhs(0.,probe)
    np.testing.assert_array_equal(s._rhs(0.,state),expected)
    assert np.max(abs(expected))==0.


@pytest.mark.parametrize('kind',['mass','boron','phase','wall'])
def test_local_domain_failure_is_not_hidden_by_aggregate(kind):
    s=CoolantChannel();y=s.state.copy()
    if kind=='mass':y[0]=-1.
    elif kind=='boron':y[2]=-1e-8
    elif kind=='phase':
        v=s.network.volumes['cell0'];v.set_quality(15.5e6,.01);y[0]=v.mass_kg;y[1]=v.internal_energy_j
    else:y[s.wall_index]=650.
    with pytest.raises(DomainError):s._admit(y)


def test_json_restart_preserves_changed_ports_and_exact_heated_suffix():
    s=CoolantChannel();s.advance(.002,[10000.]*3,left=Port(15.501e6,551.,.0007))
    saved=json.loads(json.dumps(s.checkpoint()));t=CoolantChannel.restore(saved)
    assert t.checkpoint()==saved
    assert s.advance(.002,[5000.,10000.,2000.],right=Port(15.499e6))==t.advance(.002,[5000.,10000.,2000.],right=Port(15.499e6))
    assert s.checkpoint()==t.checkpoint()


@pytest.mark.parametrize('kind',['source','extra','format','config','port','history','snapshot'])
def test_resealed_checkpoint_corruption_is_rejected(kind):
    s=CoolantChannel();p=unseal(s.checkpoint())
    if kind=='source':p['source']['eos_sha256']='0'*64
    elif kind=='extra':p['bad']=True
    elif kind=='format':p['format']='other'
    elif kind=='config':p['config']['bad']=True
    elif kind=='port':p['initial_ports']['left'].pop('pressure_pa')
    elif kind=='history':p['history']=[{'duration_s':.01,'bad':True}]
    else:p['snapshot']['state'][0]+=1.
    with pytest.raises(ValueError):CoolantChannel.restore(seal(p))


def test_invalid_command_and_late_failure_do_not_commit_states_or_ports(monkeypatch):
    s=CoolantChannel();before=s.checkpoint()
    with pytest.raises(ValueError):s.advance(.001,[-1.,0.,0.],left=Port(15.501e6))
    assert s.checkpoint()==before
    original=CoolantChannel._admit
    def reject(self,y):
        original(self,y)
        if self.time>0:raise DomainError('deliberate endpoint rejection')
    monkeypatch.setattr(CoolantChannel,'_admit',reject)
    with pytest.raises(DomainError,match='deliberate'):s.advance(.001,[1000.]*3,left=Port(15.501e6))
    assert s.checkpoint()==before
