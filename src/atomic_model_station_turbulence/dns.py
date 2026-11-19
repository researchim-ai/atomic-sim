"""Pinned, attributed El Khoury et al. profiles from Zenodo 7352880."""
import hashlib
from importlib.resources import files
import io
import json
import re
import numpy as np


def load_profile(label):
    root=files(__package__).joinpath('data')
    manifest=json.loads(root.joinpath('sources.json').read_text())
    if type(label) is not int or str(label) not in manifest['profiles']:
        raise ValueError('DNS case must be 180, 360, 550 or 1000')
    entry=manifest['profiles'][str(label)];raw=root.joinpath(entry['file']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('DNS source checksum differs')
    text=raw.decode('ascii')
    t=float(re.search(r'^Re_tau = ([^\n]+)',text,re.M)[1])
    ratio=float(re.search(r'^u_tau/U_b = ([^\n]+)',text,re.M)[1])
    reb=float(re.search(r'Re_b = (\d+);',text)[1])
    numeric='\n'.join(line for line in text.splitlines() if line and line[0] not in '%Ru')
    a=np.loadtxt(io.StringIO(numeric))
    if a.ndim!=2 or a.shape[1]!=9 or not 100<=a.shape[0]<=5000 or not np.all(np.isfinite(a)):
        raise ValueError('Malformed DNS columns')
    r=1-a[:,0];u=a[:,2]
    if r[0]!=0 or r[-1]!=1 or not np.all(np.diff(r)>0) or np.any(u<0) or abs(u[-1])>1e-10:
        raise ValueError('DNS radius ordering or no-slip limit differs')
    if not np.allclose(a[:,1],a[:,0]*t,rtol=1e-12,atol=1e-12):raise ValueError('DNS wall-distance units differ')
    if not np.allclose(a[:,3],u*ratio,rtol=1e-12,atol=1e-12):raise ValueError('DNS velocity units differ')
    if abs(2*t/ratio/reb-1)>1e-10:raise ValueError('DNS radius/diameter Reynolds convention differs')
    return dict(label=label,re_tau=t,re_bulk=reb,friction_velocity_over_bulk=ratio,
                darcy_friction=8*ratio**2,radius=r.tolist(),velocity_plus=u.tolist(),
                velocity_over_bulk=a[:,3].tolist(),sha256=entry['sha256'])
