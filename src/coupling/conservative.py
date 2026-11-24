"""One overlap operator for conservative power and dual temperature exchange.

Cell-integrated powers and cell-average densities are distinct interfaces.
The transpose identity is algebraic power/temperature pairing, not a new
thermodynamic conservation law. Mesh endpoints must match exactly.
"""
import numpy as np
from core.status import DomainError


def vector(value, size=None, *, positive=False):
    if isinstance(value,(list,tuple)) and any(isinstance(v,(bool,np.bool_)) for v in value):
        raise DomainError('Boolean values are not physical scalars')
    try:
        raw=np.asarray(value)
        if raw.dtype.kind not in 'fiu':raise ValueError('real numeric vector required')
        a=np.array(raw,dtype=float,copy=True)
    except (TypeError,ValueError,OverflowError) as error:
        raise DomainError('Finite real vector required') from error
    if a.ndim!=1 or not a.size or (size is not None and a.shape!=(size,)) or not np.isfinite(a).all():
        raise DomainError('Finite one-dimensional vector with exact cell count required')
    if positive and np.any(a<=0):raise DomainError('Strictly positive vector required')
    return a


def faces(value):
    a=vector(value)
    if not 2<=len(a)<=513:raise DomainError('One to 512 cells required')
    width=np.diff(a)
    if not np.isfinite(width).all() or np.any(width<=0):raise DomainError('Strictly increasing mesh required')
    scale=max(float(np.max(abs(a))),float(a[-1]-a[0]),1.)
    if np.any(width<=32*np.finfo(float).eps*scale):raise DomainError('Mesh cells are not sufficiently resolved in float64')
    return a


def _frozen(array):
    a=np.asarray(array,dtype=float)
    return np.frombuffer(a.tobytes(),dtype=float).reshape(a.shape)


def _matvec(matrix,values):
    try:
        with np.errstate(over='raise',invalid='raise'):
            result=matrix@values
    except FloatingPointError as error:
        raise DomainError('Transfer exceeds finite float64 range') from error
    if not np.isfinite(result).all():raise DomainError('Nonfinite transfer result')
    return result


class ConservativeMap:
    """Immutable dense overlaps for two matching one-dimensional domains."""
    def __setattr__(self,name,value):
        if getattr(self,'_sealed',False):raise AttributeError('ConservativeMap is immutable')
        object.__setattr__(self,name,value)

    def __copy__(self):return self

    def __deepcopy__(self,memo):
        memo[id(self)]=self
        return self

    def __init__(self,source_faces,target_faces):
        a,b=faces(source_faces),faces(target_faces)
        if a[0]!=b[0] or a[-1]!=b[-1]:raise DomainError('Meshes must have exactly matching physical endpoints')
        overlap=np.maximum(0.,np.minimum(b[1:,None],a[None,1:])-np.maximum(b[:-1,None],a[None,:-1]))
        sw,tw=np.diff(a),np.diff(b)
        if not np.allclose(overlap.sum(axis=0),sw,rtol=1e-12,atol=0.) or not np.allclose(overlap.sum(axis=1),tw,rtol=1e-12,atol=0.):
            raise DomainError('Mesh overlap does not cover both domains')
        self.source_faces=_frozen(a);self.target_faces=_frozen(b)
        self.source_widths=_frozen(sw);self.target_widths=_frozen(tw)
        self.overlap=_frozen(overlap)
        self.power_matrix=_frozen(overlap/sw[None,:])
        self.average_matrix=_frozen(overlap/tw[:,None])
        self.feedback_matrix=_frozen(self.power_matrix.T)
        self._sealed=True

    def integrals(self,values):
        q=vector(values,len(self.source_widths))
        return _matvec(self.power_matrix,q)

    def powers(self,values):
        q=vector(values,len(self.source_widths))
        if np.any(q<0):raise DomainError('Cell-integrated fission powers must be nonnegative')
        return self.integrals(q)

    def averages(self,values):
        return _matvec(self.average_matrix,vector(values,len(self.source_widths)))

    def feedback_temperature(self,values):
        return _matvec(self.feedback_matrix,vector(values,len(self.target_widths),positive=True))

    def audit(self,powers,temperature):
        q=vector(powers,len(self.source_widths));t=vector(temperature,len(self.target_widths),positive=True)
        self.integrals(q);self.feedback_temperature(t)
        # Normalize diagnostics only, leaving transferred states untouched.
        qs=q/max(float(np.max(abs(q))),1.);ts=t/max(float(np.max(abs(t))),1.)
        mapped=self.integrals(qs);feedback=self.feedback_temperature(ts)
        total_scale=max(float(np.sum(abs(qs))),1e-300)
        pairing_scale=max(float(abs(qs)@abs(feedback)),float(abs(mapped)@abs(ts)),1e-300)
        return dict(power_relative_defect=float(abs(mapped.sum()-qs.sum())/total_scale),
            pairing_relative_defect=float(abs(qs@feedback-mapped@ts)/pairing_scale),
            source_cells=len(q),target_cells=len(t),state_projection_used=False)
