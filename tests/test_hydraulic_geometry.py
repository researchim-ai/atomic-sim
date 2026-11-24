import json
import math

import numpy as np
import pytest

from core.status import DomainError
from coupling.hydraulic_geometry import dual_geometry,combine_boundary
from coupling.pwr_channel import ChannelPWRSession,FluidGrid


def boundary(**overrides):
    return combine_boundary(**dict(dict(area_m2=.25,length_m=8.,loss_coefficient=3.,reference_density=700.,
        channel_area_m2=.1,channel_length_m=.5,channel_loss=.2,channel_reference_density=800.),**overrides))


def test_nonuniform_dual_grid_includes_both_boundary_half_cells():
    g=dual_geometry((0.,.3,1.8,4.),.2,2.)
    np.testing.assert_allclose(g['lengths_m'],[.15,.9,1.85,1.1],rtol=1e-14)
    assert sum(g['inertance_m_inv'])==pytest.approx(20.)
    assert sum(g['loss_coefficients'])==pytest.approx(2.)


@pytest.mark.parametrize('cells',[1,2,4,8])
def test_refinement_keeps_total_inertance_loss_and_uniform_flow_kinetic_energy(cells):
    g=dual_geometry(np.linspace(0.,4.,cells+1),.2,3.)
    assert sum(g['inertance_m_inv'])==pytest.approx(4/.2)
    assert sum(g['loss_coefficients'])==pytest.approx(3.)
    assert sum(.5*i*13.**2/800. for i in g['inertance_m_inv'])==pytest.approx(.5*4/.2*13.**2/800.)
    # Series pressure pulses give the same uniform acceleration on each mesh.
    acceleration=200./sum(g['inertance_m_inv'])
    assert acceleration==pytest.approx(10.)


def test_combined_link_preserves_momentum_energy_and_loss_with_unequal_areas_and_densities():
    b=boundary();old=8/.25;half=.5/.1;flow=13.
    assert b['total_inertance_m_inv']==old+half
    assert .5*b['total_inertance_m_inv']*flow**2/b['reference_density_kg_m3']==pytest.approx(.5*flow**2*(old/700+half/800))
    assert b['loss_coefficient']/(.25**2)==pytest.approx(3./.25**2+.2/.1**2)
    assert b['reference_density_kg_m3']!=800. and b['reference_density_kg_m3']!=700.
    json.dumps(b,allow_nan=False)


@pytest.mark.parametrize('value',[True,0.,-1.,math.inf,math.nan])
def test_bad_hydraulic_scalars_rejected(value):
    with pytest.raises(ValueError):dual_geometry((0.,4.),value)
    with pytest.raises(ValueError):boundary(channel_reference_density=value)


@pytest.mark.parametrize('faces',[(0.,0.,4.),(0.,np.nextafter(0.,1.),4.),(0.,math.inf)])
def test_unresolved_or_invalid_geometry_is_rejected(faces):
    with pytest.raises((ValueError,DomainError)):dual_geometry(faces,.2)


def test_invalid_mode_and_legacy_one_cell_loss_are_rejected():
    for mode in ('unknown',True):
        with pytest.raises(ValueError):FluidGrid(momentum_geometry=mode)
    with pytest.raises(ValueError):FluidGrid((0.,4.),1.)
    assert FluidGrid((0.,4.),1.,'complete_dual').internal_loss_coefficient==1.


@pytest.mark.parametrize('cells',[1,2,8])
def test_native_complete_geometry_has_consistent_face_momentum_and_instantaneous_energy(cells):
    s=ChannelPWRSession(fluid_grid=FluidGrid(tuple(np.linspace(0.,4.,cells+1)),1.,'complete_dual'))
    y=s.state.copy();y[s.fluid_index+1]+=1e5
    s.state=s._reconstruct_state(y);s._rhs(0.,s.state);r=s.snapshot();m=r['momentum_geometry']
    p=s.plant;links=[p.network.junctions['pump'],*s.channel.junctions.values(),p.network.junctions['core_out']]
    inertance=np.array([j.length_m/j.area_m2 for j in links])
    f=r['distributed_fluid']
    np.testing.assert_allclose(inertance*np.asarray(f['face_acceleration_kg_s2']),f['face_driving_pressure_pa'],rtol=1e-8,atol=1e-6)
    legacy=sum(v['legacy_inertance_m_inv'] for v in m['boundaries'].values())
    assert sum(inertance)-legacy==pytest.approx(4/s.channel_area)
    assert abs(r['instantaneous_energy_defect_w'])<1e-3


def test_complete_checkpoint_restores_geometry_and_rejects_tampering():
    from atomic_model_multiphysics.provenance import seal,unseal
    s=ChannelPWRSession(fluid_grid=FluidGrid((0.,1.,4.),1.,'complete_dual'))
    s.advance(.005);saved=json.loads(json.dumps(s.checkpoint()))
    restored=ChannelPWRSession.restore(saved)
    assert restored.advance(.005)==s.advance(.005)
    changed=unseal(saved);changed['fluid_grid']['momentum_geometry']='legacy'
    with pytest.raises(ValueError):ChannelPWRSession.restore(seal(changed))
