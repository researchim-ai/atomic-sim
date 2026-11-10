from copy import deepcopy
import numpy as np
import pytest
from atomic_model_hem_heated import HeatedHEMNetwork
from atomic_model_hem_heated.model import canonical
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from atomic_model_hem_contact import ContactHEMNetwork
from atomic_model_workspace.common import digest
from scripts import verify_heated_hllc as heated

@pytest.mark.parametrize('scheme',['rusanov','hllc','hllc-local'])
@pytest.mark.parametrize('case',heated.CASES)
def test_heat_and_entropy_source_identities(case,scheme):
    model=HeatedHEMNetwork(heated.request_for(case,heated.CAPS[-1]),scheme=scheme)
    assert heated.source_identity(model)['passed']


def test_rusanov_rhs_preserves_qualified_source_assembly():
    request=heated.request_for('branched_pulse',heated.CAPS[-1])
    a=HeatedHEMNetwork(request,scheme='rusanov');b=QualifiedHEMNetwork(request)
    ra,aa=a.evaluate(a._y,a._controls);rb,ab=b.evaluate(b._y,b._controls)
    np.testing.assert_array_equal(ra.view(np.uint64),rb.view(np.uint64))
    for key in aa:np.testing.assert_array_equal(aa[key],ab[key])

@pytest.mark.parametrize('scheme',['hllc','hllc-local'])
def test_uniform_isochoric_heat_matches_exact_inventory_rate(scheme):
    model=HeatedHEMNetwork(heated.request_for('uniform_heat',heated.CAPS[-1]),scheme=scheme)
    rhs,_=model.evaluate(model._y,model._controls)
    np.testing.assert_allclose(rhs[:2*model.n],0.,atol=1e-11,rtol=0)
    np.testing.assert_allclose(rhs[2*model.n:3*model.n],model.volumes*1e5,rtol=0,atol=1e-9)
    before=model.snapshot();after=model.advance(1e-5)
    for a,b,power in zip(before['cells'],after['cells'],model._controls['fluid'],strict=True):
        assert abs((b['total_energy_j']-a['total_energy_j'])-power*1e-5)<1e-8

@pytest.mark.parametrize('scheme',['rusanov','hllc','hllc-local'])
def test_prescribed_heat_command_and_restart(scheme):
    request=heated.request_for('wall_relaxation',heated.CAPS[-1]);model=HeatedHEMNetwork(request,scheme=scheme)
    controls=deepcopy(model.snapshot()['controls']);controls['fluid']['a']=2000.;controls['walls']['wall']=100.
    result=model.step(controls,1e-5,expected_revision=0,controller_state={'power':2000})
    assert abs(result['global_ledger']['external_fluid_heat_j']-.02)<1e-12
    assert abs(result['global_ledger']['energy_residual_j'])<1e-8
    other=HeatedHEMNetwork(request,scheme=scheme);other.restore(model.checkpoint())
    assert canonical(other.checkpoint())==canonical(model.checkpoint())
    assert canonical(other.advance(1e-5))==canonical(model.advance(1e-5))

@pytest.mark.parametrize('damage',['friction','boiling','correlation','phase','scheme'])
def test_unsupported_physics_rejected(damage):
    request=heated.request_for('wall_relaxation',heated.CAPS[-1]);scheme='hllc-local'
    if damage=='friction':request['model']['edges'][0]['friction']='darcy'
    elif damage in ('boiling','correlation'):request['model']['walls'][0][damage]={}
    elif damage=='phase':request['model']['nodes'][0]['initial']=dict(kind='px',pressure_pa=4e6,quality=.5,velocity_m_s=0.)
    else:scheme='unknown'
    with pytest.raises(ValueError):HeatedHEMNetwork(request,scheme=scheme)


def test_rejected_hot_wall_command_rolls_back():
    model=HeatedHEMNetwork(heated.request_for('wall_relaxation',heated.CAPS[-1]))
    checkpoint=model.checkpoint();controls=deepcopy(model.snapshot()['controls']);controls['walls']['wall']=1e12
    with pytest.raises(ValueError):model.step(controls,.001,expected_revision=0,controller_state={'attempt':1})
    assert canonical(model.checkpoint())==canonical(checkpoint)
    model.advance(1e-5)


def test_signed_zero_tamper_and_cross_mode_checkpoint_refused():
    request=heated.request_for('uniform_heat',heated.CAPS[-1]);model=HeatedHEMNetwork(request)
    original=model.checkpoint();bad=deepcopy(original);bad['payload']['coordinates'][0]=-0.
    bad['sha256']=digest(bad['payload'])
    with pytest.raises(ValueError):model.restore(bad)
    assert canonical(model.checkpoint())==canonical(original)
    from scripts.verify_hllc_branching import request_for,CAPS
    cold=ContactHEMNetwork(request_for('branch_reversal',2,CAPS[-1]),scheme='hllc-local')
    with pytest.raises(ValueError):model.restore(cold.checkpoint())


def test_heat_still_rejected_by_original_contact_mode():
    with pytest.raises(ValueError):ContactHEMNetwork(heated.request_for('branched_pulse',heated.CAPS[-1]))


def test_time_comparison_refuses_nan_and_misaligned_times():
    request=heated.request_for('uniform_heat',heated.CAPS[-1]);model=HeatedHEMNetwork(request)
    packet=dict(observations=[model.snapshot()]);ref=deepcopy(packet)
    ref['observations'][0]['cells'][0]['pressure_pa']=float('nan')
    with pytest.raises(ValueError):heated.compare(packet,ref)
    ref=deepcopy(packet);ref['observations'][0]['time_s']=1.
    with pytest.raises(ValueError):heated.compare(packet,ref)

@pytest.mark.parametrize('damage',['nan_wall','missing_wall','nan_time','nan_ledger'])
def test_heated_comparison_rejects_corrupt_wall_or_entropy_evidence(damage):
    model=HeatedHEMNetwork(heated.request_for('wall_relaxation',heated.CAPS[-1]))
    packet=dict(observations=[model.snapshot()]);ref=deepcopy(packet)
    if damage=='nan_wall':ref['observations'][0]['walls'][0]['temperature_k']=float('nan')
    elif damage=='missing_wall':ref['observations'][0]['walls']=[]
    elif damage=='nan_time':ref['observations'][0]['time_s']=float('nan')
    else:ref['observations'][0]['global_ledger']['external_entropy_j_k']=float('nan')
    with pytest.raises(ValueError):heated.compare(packet,ref)
