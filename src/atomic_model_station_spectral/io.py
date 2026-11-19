"""Bounded NetCDF reader and cylindrical spectral-state audit."""
from pathlib import Path
import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import CubicSpline
from .modes import layout,field_energy,divergence


def read_state(path):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>200000000:raise ValueError('Bounded ordinary spectral checkpoint required')
    with netcdf_file(path,'r',mmap=False) as f:
        meta={}
        for name in ('t','Re','Pr','C','alpha'):
            value=np.asarray(getattr(f,name))
            if value.ndim or value.dtype.kind not in 'iuf' or not np.isfinite(value):raise ValueError('Finite scalar spectral metadata: '+name)
            meta[name]=float(value)
        if meta['t']<0 or min(meta['Re'],meta['Pr'],meta['alpha'])<=0:raise ValueError('Spectral metadata domain')
        r=np.array(f.variables['r'][:],dtype=float)
        if r.ndim!=1 or not 8<=len(r)<=256 or not np.all(np.isfinite(r)) or r[0]<=0 or np.any(np.diff(r)<=0) or abs(r[-1]-1)>5e-14:raise ValueError('Native spectral radial mesh required')
        fields={};dimensions=None
        for name in ('Ur','Ut','Uz','T'):
            v=f.variables[name];attrs=[]
            for key in ('K','M','Mp'):
                value=np.asarray(getattr(v,key))
                if value.ndim or value.dtype.kind not in 'iu':raise ValueError('Integer Fourier metadata required')
                attrs.append(int(value))
            if dimensions is None:dimensions=attrs
            elif dimensions!=attrs:raise ValueError('Fourier layouts of state fields differ')
            k,m,factor=layout(*attrs);raw=np.asarray(v[:])
            if raw.shape!=(2,len(k),len(r)) or raw.size*4>16000000 or not np.all(np.isfinite(raw)):raise ValueError('Complete finite bounded Fourier state required')
            if np.max(abs(raw[1,0]))>1e-12:raise ValueError('Mean Fourier coefficient must be real')
            fields[name]=raw[0].astype(float)+1j*raw[1].astype(float)
    return dict(metadata=meta,radius=r,dimensions=dimensions,k=k,m=m,factor=factor,fields=fields)


def audit_state(path,weights):
    s=read_state(path);w=np.asarray(weights,dtype=float);r=s['radius'];k=s['k'];m=s['m'];f=s['factor'];fields=s['fields'];meta=s['metadata']
    if w.shape!=r.shape or not np.all(np.isfinite(w)) or np.any(w<=0) or abs(sum(w)-.5)>1e-12:raise ValueError('Positive unit-radius integration rule required')
    spectra=[field_energy(fields[name],w,k,m,f) for name in ('Ur','Ut','Uz')]
    energy=sum(v['total'] for v in spectra);axial=sum((v['axial'] for v in spectra));azimuthal=sum((v['azimuthal'] for v in spectra))
    three_d=sum(v['axial_nonzero'] for v in spectra);nonaxis=sum(v['azimuthal_nonzero'] for v in spectra)
    def tail(values):
        peak=float(max(values[1:])) if len(values)>1 else 0.;total=float(sum(values[1:]))
        return dict(terminal_amplitude_ratio=float(np.sqrt(values[-1]/peak)) if peak>0 else None,
            upper_band_energy_fraction=float(sum(values[max(1,int(.8*len(values))):])/total) if total>0 else None)
    div=divergence(fields['Ur'],fields['Ut'],fields['Uz'],r,k,m,meta['alpha'])
    div_l2=float(np.sqrt(2*np.sum((abs(div)**2)@w*f)))
    u=1-r*r+np.real(fields['Uz'][0]);temperature=r*r+np.real(fields['T'][0]);volume=2*np.pi**2/meta['alpha']
    thermal=field_energy(fields['T'],w,k,m,f)
    wall=float(max(np.max(abs(fields[name][:,-1])) for name in ('Ur','Ut','Uz')))
    return dict(time=meta['t'],time_unit='R/(2Ub)',metadata=meta,radial_nodes=len(r),axial_modes=s['dimensions'][0],azimuthal_modes=s['dimensions'][1],rotational_stride=s['dimensions'][2],stored_modes=len(k),
        perturbation_kinetic_per_volume=energy,perturbation_kinetic_integral=volume*energy,
        axial_nonzero_energy_integral=volume*three_d,azimuthal_nonzero_energy_integral=volume*nonaxis,
        temperature_perturbation_energy_integral=volume*thermal['total'],
        axial_spatial_energy_fraction=three_d/energy if energy>0 else 0.,azimuthal_spatial_energy_fraction=nonaxis/energy if energy>0 else 0.,
        axial_spectrum=axial.tolist(),azimuthal_spectrum=azimuthal.tolist(),axial_tail=tail(axial),azimuthal_tail=tail(azimuthal),
        divergence_l2=div_l2,no_slip_mode_maximum=wall,temperature_wall_mode_maximum=float(np.max(abs(fields['T'][:,-1]))),
        area_mean_velocity=float(2*w@u),area_mean_temperature=float(2*w@temperature),
        mean_velocity=u.tolist(),mean_temperature=temperature.tolist(),radius=r.tolist(),
        spatial_modes_present=bool(three_d>1e-12 and nonaxis>1e-12),turbulent_statistics_qualified=False,station_usable=False,validated=False)


def compare_states(first,second,weights):
    a=read_state(first);b=read_state(second);w=np.asarray(weights)
    if any(abs(a['metadata'][name]-b['metadata'][name])>1e-12*max(1,abs(a['metadata'][name])) for name in a['metadata']) or a['dimensions'][2]!=b['dimensions'][2]:
        raise ValueError('Same epoch and physics required')
    if w.shape!=a['radius'].shape or np.any(w<=0) or abs(sum(w)-.5)>1e-12:raise ValueError('Valid comparison quadrature required')
    lookup={(int(k),int(m)):i for i,(k,m) in enumerate(zip(b['k'],b['m']))}
    ia=[];ib=[]
    for i,pair in enumerate(zip(a['k'],a['m'])):
        if tuple(pair) in lookup:ia.append(i);ib.append(lookup[tuple(pair)])
    same_grid=np.array_equal(a['radius'],b['radius'])
    if not same_grid and (min(a['radius'])<min(b['radius'])-5e-14 or max(a['radius'])>max(b['radius'])+5e-14):
        raise ValueError('Radial comparison must not extrapolate')
    result={};bitwise=a['dimensions']==b['dimensions'] and same_grid
    for name in ('Ur','Ut','Uz','T'):
        x=a['fields'][name][ia];y=b['fields'][name][ib]
        if not same_grid:y=CubicSpline(b['radius'],y,axis=1)(a['radius'])
        d=x-y
        result[name]=float(np.sqrt(2*np.sum((abs(d)**2)@w*a['factor'][ia])))
        bitwise=bool(bitwise and np.array_equal(x,y))
    return dict(common_modes=len(ia),rms_difference=result,maximum_rms=max(result.values()),physical_fields_bitwise_equal=bitwise,
                radial_interpolation='none' if same_grid else 'cubic spline from second grid to first, without extrapolation',
                note='Common Fourier modes only; omitted-mode content is assessed by separate tail diagnostics')
