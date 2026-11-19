"""Positive, minimum-norm corrections for radial quadrature moments.

Inputs contain no velocity, temperature, Reynolds number or experimental fit.
Integral P_j(2r^2-1) r dr is 1/2 for j=0 and zero otherwise.
"""
import numpy as np
from numpy.polynomial.legendre import legvander

POLICY='radial-even-legendre-svd-minimum-norm/v1'


def _vector(value,name):
    a=np.asarray(value)
    if a.dtype.kind not in 'iuf' or a.ndim!=1 or not 16<=a.size<=1024:
        raise ValueError('Finite numeric vector16..1024 required: '+name)
    a=a.astype(float)
    if not np.all(np.isfinite(a)):raise ValueError('Nonfinite '+name)
    return a


def correct_quadrature(radius,weights,*,degree=10):
    """Use SVD, retain all nodes and reject a nonpositive corrected rule.

    The L2 correction lies in the constraint row space. No clipping or
    normalization is applied after solving the moment constraints.
    Bounds and positivity are study admission guards, not a general theorem
    that every input mesh admits this particular positive correction.
    """
    r=_vector(radius,'radius');w=_vector(weights,'weights')
    if w.shape!=r.shape or np.any(np.diff(r)<=0) or r[0]<0 or r[-1]>1+5e-14:
        raise ValueError('Ordered common radial mesh in[0,1] required')
    if np.any(w<=0) or abs(float(np.sum(w))-.5)>.01:raise ValueError('Positive near-normalized native weights required')
    if type(degree) is not int or not 0<=degree<=16 or degree+1>=len(r):raise ValueError('Bounded integral moment degree required')
    a=legvander(2*r*r-1,degree).T;target=np.zeros(degree+1);target[0]=.5
    left,singular,right=np.linalg.svd(a,full_matrices=False)
    condition=float(singular[0]/singular[-1]) if singular[-1]>0 else float('inf')
    if not np.isfinite(condition) or condition>1e8:raise ValueError('Radial moment system rank/conditioning limit')
    delta=right.T@((left.T@(target-a@w))/singular);new=w+delta
    residual=float(np.max(np.abs(a@new-target)))
    if not np.all(np.isfinite(new)) or np.min(new)<=0:raise ValueError('Minimum-norm corrected rule is not positive')
    if residual>5e-14:raise ArithmeticError('Radial moment residual exceeds predeclared limit')
    # Independent KKT check: correction orthogonal to the constraint nullspace.
    row_residual=float(np.max(np.abs(delta-right.T@(right@delta))))
    if row_residual>5e-14:raise ArithmeticError('Radial minimum-norm optimality residual')
    return dict(policy=POLICY,degree_in_radius_squared=degree,radius=r.tolist(),original_weights=w.tolist(),
        corrected_weights=new.tolist(),original_moments=(a@w).tolist(),corrected_moments=(a@new).tolist(),
        target_moments=target.tolist(),maximum_moment_residual=residual,minimum_weight=float(np.min(new)),
        correction_l2=float(np.linalg.norm(delta)),condition=condition,optimality_residual=row_residual,
        model_coefficients_fitted=False,flow_data_used=False)
