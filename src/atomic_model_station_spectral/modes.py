"""Openpipeflow half-plane packing and source-independent modal invariants."""
import numpy as np
from scipy.sparse import csr_matrix


def layout(k_count,m_count,mp=1):
    if any(type(v) is not int for v in (k_count,m_count,mp)) or not 1<=k_count<=128 or not 1<=m_count<=128 or not 1<=mp<=8:
        raise ValueError('Bounded integer Fourier dimensions required')
    pairs=[(k,0) for k in range(k_count)]
    pairs.extend((k,m*mp) for m in range(1,m_count) for k in range(1-k_count,k_count))
    k,m=np.array(pairs,dtype=int).T;factor=np.full(len(k),2.);factor[0]=1.
    return k,m,factor


def field_energy(values,weights,k,m,factor):
    a=np.asarray(values);w=np.asarray(weights,dtype=float)
    if a.ndim!=2 or a.shape!=(len(k),len(w)) or not np.iscomplexobj(a) or not np.all(np.isfinite(a)):
        raise ValueError('Complete finite complex spectral field required')
    if np.any(w<=0) or not np.all(np.isfinite(w)):raise ValueError('Positive finite radial weights required')
    energy=(np.real(a)**2+np.imag(a)**2)@w*factor
    axial=np.bincount(abs(k),weights=energy);azimuthal=np.bincount(m,weights=energy)
    def tail(s):
        nonzero=s[1:];total=float(np.sum(nonzero));peak=float(np.max(nonzero)) if len(nonzero) else 0.
        band=s[max(1,int(.8*len(s))):]
        return dict(terminal_amplitude_ratio=float(np.sqrt(s[-1]/peak)) if peak>0 else None,
                    upper_band_energy_fraction=float(np.sum(band)/total) if total>0 else None)
    return dict(radial_energy_per_mode=energy,total=float(np.sum(energy)),axial=axial,azimuthal=azimuthal,
        axial_nonzero=float(np.sum(energy[k!=0])),azimuthal_nonzero=float(np.sum(energy[m!=0])),
        axial_tail=tail(axial),azimuthal_tail=tail(azimuthal))


def derivative_matrices(radius):
    r=np.asarray(radius,dtype=float);n=len(r)
    if r.ndim!=1 or not 8<=n<=256 or not np.all(np.isfinite(r)) or r[0]<=0 or np.any(np.diff(r)<=0) or abs(r[-1]-1)>5e-14:
        raise ValueError('Regular positive native radial mesh required')
    extended=np.r_[-r[:4][::-1],r];positive=np.zeros((n,n));ghost=np.zeros((n,n))
    for i in range(n):
        indices=np.arange(max(0,i),min(n+4,i+9));nodes=extended[indices]
        scale=float(np.max(abs(nodes-r[i])));x=(nodes-r[i])/scale
        matrix=np.vander(x,N=len(x),increasing=True).T;rhs=np.zeros(len(x));rhs[1]=1/scale
        coefficients=np.linalg.solve(matrix,rhs)
        for index,c in zip(indices,coefficients):
            if index<4:ghost[i,3-index]+=c
            else:positive[i,index-4]+=c
    return csr_matrix(positive),csr_matrix(ghost)


def divergence(ur,ut,uz,radius,k,m,alpha):
    r=np.asarray(radius);positive,ghost=derivative_matrices(r)
    parity=(1-2*((m+1)%2))[:,None]
    dr=positive.dot(ur.T).T+parity*ghost.dot(ur.T).T
    return dr+ur/r+1j*m[:,None]*ut/r+1j*alpha*k[:,None]*uz
