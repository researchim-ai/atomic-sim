import numpy as np
import pytest
from core.status import DomainError
from coupling import ConservativeMap,conserve_integral
from atomic_model_multiphysics.coupled import map_power,map_temperature,overlaps


def test_exact_nonmatching_mesh_example_and_duality():
    op=ConservativeMap([0,1,2],[0,.5,1.5,2])
    np.testing.assert_allclose(op.powers([10,30]),[5,20,15])
    np.testing.assert_allclose(op.feedback_temperature([500,600,700]),[550,650])
    assert op.audit([10,30],[500,600,700])['pairing_relative_defect']<1e-15
    np.testing.assert_allclose(op.averages([10,30]),[10,20,30])
    np.testing.assert_allclose(map_power([0,1,2],[0,.5,1.5,2],[10,30]),[5,20,15])
    np.testing.assert_allclose(map_temperature([0,.5,1.5,2],[0,1,2],[500,600,700]),[550,650])


def test_power_and_density_interfaces_cannot_be_confused():
    op=ConservativeMap([0,.25,1],[0,.5,1])
    assert op.powers([1,3]).sum()==pytest.approx(4)
    np.testing.assert_allclose(op.averages([4,4]),[4,4])
    np.testing.assert_allclose(conserve_integral([4,4],[.25,.75],[.5,.5]),[4,4])


def test_random_heterogeneous_meshes_signed_fields_and_pairing():
    rng=np.random.default_rng(384)
    for n,m in [(2,3),(3,2),(8,13),(31,7),(64,128)]:
        a=np.r_[0.,np.cumsum(rng.uniform(.1,1,n))];a/=a[-1]
        b=np.r_[0.,np.cumsum(rng.uniform(.1,1,m))];b/=b[-1]
        op=ConservativeMap(a,b);q=rng.normal(size=n)*1e6;t=rng.uniform(400,900,m)
        report=op.audit(q,t)
        assert report['power_relative_defect']<1e-14 and report['pairing_relative_defect']<1e-14
        np.testing.assert_allclose(op.averages(np.ones(n)),1.,rtol=1e-14)
        assert np.min(op.powers(abs(q)))>=0


def test_operators_own_immutable_arrays_and_do_not_alias_inputs():
    a=np.array([0.,1.,2.]);b=a.copy();op=ConservativeMap(a,b);a[1]=.5
    assert op.source_faces[1]==1.
    for field in ('source_faces','target_faces','power_matrix','average_matrix','feedback_matrix','overlap'):
        with pytest.raises(ValueError):getattr(op,field).setflags(write=True)
    with pytest.raises(AttributeError):op.power_matrix=np.eye(2)
    result=op.powers([1.,2.]);result[0]=99
    np.testing.assert_array_equal(op.powers([1.,2.]),[1.,2.])


@pytest.mark.parametrize('a,b',[([0,1],[0,2]),([0,1],[1,2]),([0,1,1],[0,1]),
    ([0,float('nan'),1],[0,1]),([0,float('inf')],[0,1]),([0,1j],[0,1]),
    ([False,1],[0,1]),([[0,1]],[0,1]),([0,1e-20,1],[0,1]),([0],[0,1])])
def test_invalid_geometries_fail_closed(a,b):
    with pytest.raises(DomainError):ConservativeMap(a,b)


@pytest.mark.parametrize('source,sw,tw',[(np.array([np.nan,1]),[.5,.5],[1]),([1,2],[0,1],[1]),
    ([1,2],[-1,2],[1]),([1,2],[.5,.5],[1.000000001]),([1],[np.inf],[1]),
    ([1,2],[[.5,.5]],[1]),([True,1],[.5,.5],[1]),([1,2],[.5,.5],[0,1])])
def test_legacy_silent_nan_and_bad_widths_are_now_refused(source,sw,tw):
    with pytest.raises(DomainError):conserve_integral(source,np.asarray(sw),np.asarray(tw))


def test_roundoff_in_width_sums_is_allowed_without_rescaling_values():
    sw=np.full(10,.1);tw=np.array([.5,.5])
    values=np.linspace(1.,2.,10)
    mapped=conserve_integral(values,sw,tw)
    assert mapped@tw==pytest.approx(values@sw,rel=1e-14)


def test_malformed_fields_negative_fission_and_nonpositive_temperature_refused():
    op=ConservativeMap([0,1,2],[0,.5,2])
    for q in ([1],[[1,2]],[1,float('nan')],[1+0j,2]):
        with pytest.raises(DomainError):op.integrals(q)
    with pytest.raises(DomainError):op.powers([1,-1])
    with pytest.raises(DomainError):op.feedback_temperature([0,500])
    with pytest.raises(ValueError):map_power([0,1,2],[0,2],[1,-1])
    with pytest.raises(ValueError):map_temperature([0,1,2],[0,2],[0,500])


def test_overflowing_transfer_cannot_return_infinity():
    op=ConservativeMap([0,1,2],[0,2])
    with pytest.raises(DomainError):op.integrals([1.7e308,1.7e308])


def test_deep_copy_preserves_readonly_operator_ownership():
    from copy import deepcopy
    op=ConservativeMap([0,1,2],[0,.5,2])
    assert deepcopy(op) is op
    with pytest.raises(ValueError):deepcopy(op).power_matrix.setflags(write=True)


def test_unrepresentable_legacy_integral_is_refused_even_when_density_is_finite():
    with pytest.raises(DomainError):conserve_integral([1.7e308,1.7e308],[1,1],[2])


def test_existing_spatial_restart_binds_shared_exchange_dependency(monkeypatch):
    from pathlib import Path
    from atomic_model_multiphysics import provenance
    from atomic_model_multiphysics.coupled import SpatialSession
    from atomic_model_multiphysics.feedback import numerical_table
    from atomic_model_spatial.cases import slab
    from coupling import conservative
    session=SpatialSession(slab(6),numerical_table());saved=session.checkpoint()
    original=provenance.sha;target=Path(conservative.__file__)
    def changed(path):return '0'*64 if Path(path)==target else original(path)
    monkeypatch.setattr(provenance,'sha',changed)
    with pytest.raises(ValueError,match='source/history'):SpatialSession.restore(saved)
