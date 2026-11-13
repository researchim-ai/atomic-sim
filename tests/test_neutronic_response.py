from copy import deepcopy
import math
import numpy as np
import pytest
from atomic_model_spatial import Model
from atomic_model_spatial.cases import slab
from atomic_model_spatial.reference import dense_k
from atomic_model_multiphysics.absorber import insert
from atomic_model_neutronic_response import absorption_temperature_response as response


def derivative(cells):
    d=np.zeros((2,cells));d[1]=np.linspace(1e-5,4e-5,cells);return d


def test_adjoint_temperature_response_matches_independent_dense_difference():
    spec=insert(slab(12,heterogeneous=True),.37,.2);d=derivative(12);original=deepcopy(spec)
    r=response(spec,d,derivative_provenance='Synthetic verification fixture')
    h=.05;plus=deepcopy(spec);minus=deepcopy(spec)
    plus['absorption_per_m']=(np.asarray(spec['absorption_per_m'])+h*d).tolist()
    minus['absorption_per_m']=(np.asarray(spec['absorption_per_m'])-h*d).tolist()
    fd=(1./dense_k(Model(minus))-1./dense_k(Model(plus)))/(2*h)
    assert math.isclose(fd,r['temperature_coefficient_per_k'],rel_tol=1e-6)
    assert r['forward_relative_residual']<1e-11 and r['adjoint_relative_residual']<1e-11
    assert spec==original and not r['validated']


def test_cell_gradient_matches_local_perturbation():
    spec=slab(8);r=response(spec,derivative(8),derivative_provenance='synthetic')
    h=1e-6;plus=deepcopy(spec);minus=deepcopy(spec)
    plus['absorption_per_m'][1][3]+=h;minus['absorption_per_m'][1][3]-=h
    fd=(1/dense_k(Model(minus))-1/dense_k(Model(plus)))/(2*h)
    assert math.isclose(fd,r['absorption_gradient_m'][1][3],rel_tol=1e-6)


def test_absorber_changes_temperature_response_through_spatial_flux():
    spec=slab(12,heterogeneous=True);d=derivative(12)
    a=response(spec,d,derivative_provenance='synthetic')
    b=response(insert(spec,.37,.2),d,derivative_provenance='synthetic')
    assert b['k']<a['k'] and abs(a['temperature_coefficient_pcm_per_k']-b['temperature_coefficient_pcm_per_k'])>1e-3


def test_zero_and_signed_derivatives_are_not_clipped():
    spec=slab(8)
    assert response(spec,np.zeros((2,8)),derivative_provenance='zero fixture')['temperature_coefficient_per_k']==0.
    assert response(spec,-derivative(8),derivative_provenance='negative fixture')['temperature_coefficient_per_k']>0.


@pytest.mark.parametrize('damage',['shape','nan','complex','bool','provenance','budget','exhausted'])
def test_bad_response_inputs_or_unconverged_adjoint_refused(damage):
    spec=slab(8);d=derivative(8);p='synthetic';budget=10000
    if damage=='shape':d=d[0]
    elif damage=='nan':d[0,0]=float('nan')
    elif damage=='complex':d=d.astype(complex)
    elif damage=='bool':d=np.zeros((2,8),dtype=bool)
    elif damage=='provenance':p=''
    elif damage=='budget':budget=True
    else:budget=1
    with pytest.raises(ValueError):response(spec,d,derivative_provenance=p,max_iterations=budget)
