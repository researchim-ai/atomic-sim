"""Differentiate A phi = F phi / k; A/F already contain FV cell volumes.

This is the adjoint of the assembled discrete problem, with unchanged charts,
coefficients and boundary conditions. It is not evaluated nuclear data.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import math
import numpy as np
from scipy.sparse.csgraph import connected_components
from atomic_model_spatial import Model,eigenvalue
from atomic_model_spatial.solvers import factor,relative_residual
from atomic_model_spatial.workflow import identity as spatial_identity
from atomic_model_workspace.common import digest

SOURCE='https://digital.library.unt.edu/ark:/67531/metadc870144/m1/291/ocr/'


def identity():
    return dict(spatial=spatial_identity(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob('*.py'))})


def _model(model):
    if type(model) is not Model:raise ValueError('Immutable slab Model required')
    graph=(model.fission-model.loss).tocsc();graph.setdiag(0.);graph.eliminate_zeros()
    if np.any(graph.data<0) or connected_components(graph,directed=True,connection='strong',return_labels=False)!=1:
        raise ValueError('Irreducible coupled neutron operator required; degenerate/disconnected modes unsupported')


def _pair(model,tolerance,max_iterations,power_weights_w,power_w):
    _model(model)
    if type(tolerance) not in (int,float) or not math.isfinite(tolerance) or not 1e-13<=tolerance<=1e-6:
        raise ValueError('Eigen tolerance in [1e-13,1e-6] required')
    direct=eigenvalue(model,tolerance=tolerance,max_iterations=max_iterations)
    phi=np.asarray(direct['flux']).ravel();k=direct['k'];fphi=model.fission@phi
    psi=np.ones_like(phi);psi/=float(psi@fphi);solve=factor(model.loss).solve
    for iteration in range(1,max_iterations+1):
        raw=solve(model.fission.T@psi,trans='T');scale=float(raw@fphi)
        if not np.isfinite(raw).all() or np.any(raw<0) or not math.isfinite(scale) or scale<=0:
            raise ValueError('Invalid nonnegative adjoint iteration')
        new=raw/scale
        residual=relative_residual(model.loss.T,new,model.fission.T@new/k)
        change=float(np.max(abs(new-psi))/max(float(np.max(abs(new))),1e-300));psi=new
        if residual<=tolerance and change<=tolerance and abs(scale-k)/k<=10*tolerance:break
    else:raise ValueError('Adjoint iteration budget exhausted; no accepted response')
    if np.any(phi<=0) or np.any(psi<=0):raise ValueError('Strictly positive fundamental direct/adjoint pair required')
    normalization=dict(kind='integrated-neutron-birth',target=1.,value=float(np.sum(model.fission@phi)))
    if (power_weights_w is None)!=(power_w is None):raise ValueError('Both declared power weights and target required')
    if power_weights_w is not None:
        if np.iscomplexobj(power_weights_w):raise ValueError('Real power weights required')
        weights=np.array(power_weights_w,dtype=float,copy=True)
        if weights.shape!=(model.g,model.n) or not np.isfinite(weights).all() or np.any(weights<0) or not np.any(weights>0):
            raise ValueError('Finite nonnegative group/cell power weights required')
        if type(power_w) not in (int,float) or not math.isfinite(power_w) or power_w<=0:
            raise ValueError('Positive finite declared power target required')
        value=float(weights.ravel()@phi)
        if not math.isfinite(value) or value<=0:raise ValueError('Invalid declared power projection')
        multiplier=power_w/value;phi=phi*multiplier;psi=psi/multiplier
        normalization=dict(kind='declared-power-projection',target_w=power_w,value_w=float(weights.ravel()@phi),
            weights_w_per_flux=weights.tolist(),evaluated_energy_per_fission_inferred=False)
    denominator=float(psi@(model.fission@phi))
    if not np.isfinite(phi).all() or not np.isfinite(psi).all() or np.any(phi<=0) or np.any(psi<=0) or not math.isfinite(denominator) or denominator<=0:
        raise ValueError('Nonfinite or singular pair normalization')
    if power_weights_w is not None and abs(normalization['value_w']-power_w)>1e-12*power_w:
        raise ValueError('Declared power normalization failed')
    psi=psi/denominator
    direct_residual=relative_residual(model.loss,phi,model.fission@phi/k)
    adjoint_residual=relative_residual(model.loss.T,psi,model.fission.T@psi/k)
    pairing=float(psi@(model.fission@phi))
    if max(direct_residual,adjoint_residual)>tolerance or abs(pairing-1.)>1e-12:
        raise ValueError('Direct/adjoint residual or pairing normalization failed')
    return dict(k=k,flux=phi.reshape(model.g,model.n).tolist(),adjoint=psi.reshape(model.g,model.n).tolist(),
        direct_relative_residual=direct_residual,adjoint_relative_residual=adjoint_residual,
        direct_iterations=direct['iterations'],adjoint_iterations=iteration,fission_pairing=pairing,
        normalization=normalization,operator_scope='discrete FV transpose; volumes already included in A/F')


def solve_pair(model,*,tolerance=1e-11,max_iterations=10000,power_weights_w=None,power_w=None):
    source=identity();before=digest(model.config()) if type(model) is Model else None
    result=_pair(model,tolerance,max_iterations,power_weights_w,power_w)
    if source!=identity() or before!=digest(model.config()):raise ValueError('Model or response source changed')
    return dict(source=source,model_sha256=before,**result,model_unchanged=True,plant_validated=False)


def sensitivities(model,*,tolerance=1e-11,max_iterations=10000,power_weights_w=None,power_w=None):
    pair=solve_pair(model,tolerance=tolerance,max_iterations=max_iterations,
        power_weights_w=power_weights_w,power_w=power_w)
    phi=np.asarray(pair['flux']);psi=np.asarray(pair['adjoint']);k=pair['k'];width=model.width
    chi=(1-model.beta.sum(axis=0))*model.chi_prompt+model.beta.sum(axis=0)*model.chi_delayed
    birth=k*phi*width*np.sum(psi*chi,axis=0)
    absorption=-k*k*psi*phi*width
    scatter=np.zeros((model.g,model.g,model.n));diffusion=np.zeros((model.g,model.n))
    for origin in range(model.g):
        for target in range(model.g):
            if target!=origin:scatter[target,origin]=-k*k*width*phi[origin]*(psi[origin]-psi[target])
        d=model.diffusion_m[origin]
        conductance=1/(width[:-1]/(2*d[:-1])+width[1:]/(2*d[1:]))
        response=-k*k*(psi[origin,:-1]-psi[origin,1:])*(phi[origin,:-1]-phi[origin,1:])
        diffusion[origin,:-1]+=response*conductance**2*width[:-1]/(2*d[:-1]**2)
        diffusion[origin,1:]+=response*conductance**2*width[1:]/(2*d[1:]**2)
        for side,cell in ((0,0),(1,model.n-1)):
            if model.boundaries[side]=='zero_flux':diffusion[origin,cell]+=-k*k*psi[origin,cell]*phi[origin,cell]*2/width[cell]
    gradients=dict(nu_fission_per_m=birth,absorption_per_m=absorption,scatter_per_m=scatter,diffusion_m=diffusion)
    if any(not np.isfinite(v).all() for v in gradients.values()):raise ValueError('Nonfinite eigenvalue gradient')
    if pair['source']!=identity() or pair['model_sha256']!=digest(model.config()):raise ValueError('Response source or model changed')
    return dict(**pair,gradients={key:value.tolist() for key,value in gradients.items()},
        elasticities={key:(value*getattr(model,'scatter' if key=='scatter_per_m' else key)/k).tolist() for key,value in gradients.items()},
        scatter_convention='[to,from,cell]; diagonal entries are placeholders, not parameters',
        held_fixed=['mesh','boundaries','spectra','beta','decay','speed'],
        formula='dk = psi.T (k*dF - k**2*dA) phi / (psi.T F phi)',
        response_scope='Local first derivative of a simple static k eigenvalue; not transient or nonlinear uncertainty propagation')


def station_core_response(core,*,power_w=None,tolerance=1e-11):
    from atomic_model_station.core import DiffusionCore
    if type(core) is not DiffusionCore:raise ValueError('Existing station DiffusionCore required')
    if core.config.axial_bias_reactivity!=0.:raise ValueError('Static zero-bias core reference required')
    power=float(core.kinetics.power_nominal_w) if power_w is None else power_w
    weights=(core.kinetics.power_nominal_w*core.weights).reshape(1,core.cells)
    result=sensitivities(core.model,tolerance=tolerance,power_weights_w=weights,power_w=power)
    state=np.r_[np.asarray(result['flux']).ravel(),np.zeros(6*core.cells)]
    profile=core.fission_by_cell(state)
    defect=abs(float(profile.sum())-power)/max(abs(power),1.)
    if defect>1e-12:raise ValueError('Station declared power projection failed')
    return dict(**result,station_core_config=asdict(core.config),fission_by_cell_w=profile.tolist(),
        relative_power_projection_defect=defect,reference_scope='zero bulk reactivity and zero axial bias; static only',
        nuclear_data_family='illustrative',local_thermal_feedback=False,closed_tasks=[])
