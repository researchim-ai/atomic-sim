"""Liquid HLLC with qualified heat/entropy ledgers; no boiling or friction."""
import hashlib
import json
from pathlib import Path
import numpy as np
from atomic_model_hem_qualified.model import QualifiedHEMNetwork
from atomic_model_hem_contact.model import signature as contact_signature
from atomic_model_hem_contact.flux import _FluxEvaluation, SOURCE
from atomic_model_hem_system.model import MAX_CALLS
from atomic_model_workspace.common import digest, keys

MODEL='heated-liquid-hllc-graph-v1'


def signature():
    return dict(contact=contact_signature(),heated={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob('*.py'))})


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


class HeatedHEMNetwork(QualifiedHEMNetwork):
    """Constant wall conductance, ambient UA and externally prescribed powers.

    The qualified source assembly is retained. Only conservative face transport
    and its stationary-node reactions are replaced for HLLC schemes.
    """
    def __init__(self,request,*,scheme='hllc-local'):
        if scheme not in ('rusanov','hllc','hllc-local'):raise ValueError('Declared heated scheme required')
        self._scheme=scheme;self._heated_ready=False
        if request['friction'] or any(e['friction']!='none' for e in request['model']['edges']):
            raise ValueError('Heated HLLC friction not supported')
        if any('boiling' in w or 'correlation' in w for w in request['model']['walls']):
            raise ValueError('Heated HLLC requires constant linear wall conductance')
        super().__init__(request)
        self.identity=digest(dict(model=MODEL,scheme=scheme,request=self.request))
        self.source=signature();self._heated_ready=True
        self.evaluate(self._y,self._controls)

    @property
    def scheme(self):return self._scheme

    def _sources(self):
        if not self._heated_ready:return QualifiedHEMNetwork._sources(self)
        if signature()!=self.source:raise ValueError('Heated HLLC source changed')

    def primitive(self,y):
        point=super().primitive(y)
        if not self._initializing and any(s.region!=1 or s.phase!='liquid' for s in point[4]):
            raise ValueError('Heated HLLC leaves single-phase liquid domain')
        return point

    def _evaluate_point(self,y,controls,point):
        rhs,aux=super()._evaluate_point(y,controls,point)
        if self._initializing or self.scheme=='rusanov':return rhs,aux
        _,_,_,velocity,states,_,speeds=point
        evaluation=_FluxEvaluation(states,speeds)
        face_flux=evaluation.local_hllc if self.scheme=='hllc-local' else evaluation.hllc
        cache={};columns=[]
        for (a,b),area in zip(self.faces,self.face_areas):
            key=(states[a],states[b],float(velocity[a]),float(velocity[b]))
            if key not in cache:cache[key]=face_flux(*key,wave_bound=self.wave,energy_reference=self.eref)
            columns.append(cache[key]*area)
        flux=np.column_stack(columns);net=np.zeros((3,self.n))
        for k,(a,b) in enumerate(self.faces):net[:,a]-=flux[:,k];net[:,b]+=flux[:,k]
        support=np.zeros(self.n);support[:self.node_count]=-net[1,:self.node_count]
        net[1]+=support
        net[2]+=controls['fluid']
        np.add.at(net[2],self.ports,aux['wall_to_fluid_w'])
        rhs[:3*self.n]=net.reshape(-1)
        physical=flux.copy();physical[2]+=self.eref*physical[0]
        rhs[self.face_slice]=physical.reshape(-1);rhs[self.support_slice]=support
        aux['shared_flux']=flux;aux['node_support_force_n']=support
        if not np.isfinite(rhs).all():raise ValueError('Invalid heated HLLC rates')
        return rhs,aux

    def contract(self):
        value=super().contract()
        value.update(model=MODEL,numerical_flux=self.scheme,flux_source=SOURCE,
                     heated_scope='Region1 liquid, constant wall conductance, ambient UA, prescribed powers; no friction or phase change',
                     transport_assembly='Qualified source/entropy assembly followed by face transport replacement',
                     fallback='None; rejected calls roll back',validated=False,plant_referenced=False)
        value['friction_scope']='None; friction rejected'
        return value

    def _checkpoint(self,*args):
        cp=super()._checkpoint(*args);cp['payload'].update(model=MODEL,scheme=self.scheme)
        cp['sha256']=digest(cp['payload']);return cp

    def restore(self,checkpoint):
        with self._lock:
            self._sources();keys(checkpoint,('payload','sha256'));p=checkpoint['payload']
            encoded=canonical(checkpoint)
            if (len(encoded.encode())>4*1024**2 or digest(p)!=checkpoint['sha256'] or p.get('model')!=MODEL or
                p.get('scheme')!=self.scheme or p.get('source')!=self.source or p.get('input_sha256')!=self.identity):
                raise ValueError('Heated checkpoint identity/size')
            calls=p.get('calls')
            if not isinstance(calls,list) or len(calls)>MAX_CALLS:raise ValueError('Heated replay budget')
            other=type(self)(self.request,scheme=self.scheme)
            for call in calls:
                keys(call,('duration_s','command'));command=call['command']
                if command is None:other.advance(call['duration_s'])
                else:
                    keys(command,('controls','expected_revision','controller_state'))
                    other.step(command['controls'],call['duration_s'],expected_revision=command['expected_revision'],controller_state=command['controller_state'])
            if canonical(other.checkpoint())!=encoded:raise ValueError('Heated replay differs')
            self._sources()
            self._y,self._time,self._controls,self._controller,self._calls=other._y,other._time,other._controls,other._controller,other._calls
            return self.snapshot()
