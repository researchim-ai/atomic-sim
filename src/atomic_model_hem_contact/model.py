"""Opt-in HLLC graph with the qualified inventory, ledger and RK contracts.

The existing qualified/Rusanov family is not modified. Phase changes, heat,
friction and automatic fallback are deliberately outside this release's scope.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np

from atomic_model_hem_qualified.model import QualifiedHEMNetwork,signature as foundation_signature
from atomic_model_hem_system.model import MAX_CALLS
from atomic_model_workspace.common import digest,keys
from .flux import _FluxEvaluation,SOURCE

MODEL = 'contact-preserving-liquid-graph-v1'


def signature():
    return dict(foundation=foundation_signature(),contact={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob('*.py'))})


class ContactHEMNetwork(QualifiedHEMNetwork):
    def __init__(self,request,*,scheme='hllc'):
        if scheme not in ('hllc','hllc-local','rusanov'):raise ValueError('Declared contact graph scheme required')
        self._scheme=scheme;self._contact_ready=False
        spec=request['model']
        if request['friction'] or any(e['friction']!='none' for e in spec['edges']):
            raise ValueError('Contact graph friction not qualified')
        powers=[n['heat_power_w'] for n in spec['nodes']]+[p for e in spec['edges'] for p in e['heat_power_w']]
        for wall in spec['walls']:
            powers.extend(wall.get(k) for k in ('conductance_w_k','ambient_ua_w_k','heat_power_w'))
            if 'boiling' in wall:raise ValueError('Contact graph boiling not qualified')
        if any(p!=0 for p in powers):raise ValueError('Contact graph must be adiabatic and unforced')
        super().__init__(request)
        self.identity=digest(dict(model=MODEL,scheme=scheme,request=self.request))
        self.source=signature();self._contact_ready=True
        self.evaluate(self._y,self._controls)

    @property
    def scheme(self):return self._scheme

    def _sources(self):
        if not self._contact_ready:return QualifiedHEMNetwork._sources(self)
        if signature()!=self.source:raise ValueError('Contact graph source changed')

    def primitive(self,y):
        point=super().primitive(y)
        if not self._initializing and any(s.region!=1 or s.phase!='liquid' for s in point[4]):
            raise ValueError('Contact graph leaves single-phase liquid domain')
        return point

    def _evaluate_point(self,y,controls,point):
        if self._initializing:return super()._evaluate_point(y,controls,point)
        if np.any(controls['fluid']!=0) or np.any(controls['walls']!=0):
            raise ValueError('Contact graph forced controls not qualified')
        if self.scheme=='rusanov':return super()._evaluate_point(y,controls,point)
        _,momentum,_,velocity,states,tw,speeds=point
        if np.any(momentum[:self.node_count]!=0):raise ValueError('Stationary node acquired momentum')
        # The constructor admits no drag, boiling or external heating. Keep
        # source/entropy ledgers, but do not build a discarded Rusanov flux.
        temperature=np.array([state.temperature_k for state in states])
        conductance=self._thermal_conductance(point)
        exchange=conductance*(tw-temperature[self.ports])
        external=controls['walls']+self.ua*(self.ambient-tw)
        drag=np.zeros(self.n);rhs=np.zeros_like(y)
        rhs[self.wall_slice]=(external-exchange)/self.capacity
        rhs[self.cell_heat_slice]=controls['fluid']
        rhs[self.wall_exchange_slice]=exchange
        rhs[self.wall_external_slice]=external
        rhs[self.drag_slice]=drag
        rhs[self.entropy_index]=sum(controls['fluid']/temperature)+sum(controls['walls']/tw+self.ua*(self.ambient-tw)/self.ambient)
        rhs[self.production_index]=sum(conductance*(tw-temperature[self.ports])**2/(tw*temperature[self.ports])+self.ua*(self.ambient-tw)**2/(self.ambient*tw))
        rhs[self.drag_entropy_index]=sum(-drag*velocity/temperature)+0.0
        aux=dict(wall_to_fluid_w=exchange,drag_force_n=drag)
        evaluation=_FluxEvaluation(states,speeds)
        face_flux=evaluation.local_hllc if self.scheme=='hllc-local' else evaluation.hllc
        # Exact per-stage memoization only: no rounded keys or persistent state.
        cache={};columns=[]
        for (a,b),area in zip(self.faces,self.face_areas):
            key=(states[a],states[b],float(velocity[a]),float(velocity[b]))
            if key not in cache:
                cache[key]=face_flux(*key,wave_bound=self.wave,energy_reference=self.eref)
            columns.append(cache[key]*area)
        flux=np.column_stack(columns)
        net=np.zeros((3,self.n))
        for k,(a,b) in enumerate(self.faces):net[:,a]-=flux[:,k];net[:,b]+=flux[:,k]
        support=np.zeros(self.n);support[:self.node_count]=-net[1,:self.node_count]
        net[1]+=support
        rhs[:3*self.n]=net.reshape(-1)
        physical=flux.copy();physical[2]+=self.eref*physical[0]
        rhs[self.face_slice]=physical.reshape(-1);rhs[self.support_slice]=support
        aux['shared_flux']=flux;aux['node_support_force_n']=support
        if not np.isfinite(rhs).all() or rhs[self.drag_entropy_index]<0:raise ValueError('Invalid graph rates')
        return rhs,aux

    def contract(self):
        value=super().contract()
        value.update(model=MODEL,numerical_flux=self.scheme,flux_source=SOURCE,
            contact_scope='Single-phase liquid; adiabatic, no friction or forcing; bounded admitted star states',
            signal_envelope=('1.01 * max(abs(v)+c), both admitted stars checked; global CFL cap retained'
                             if self.scheme=='hllc-local' else 'Fixed global wave cap'),
            fallback='None; rejected stages roll back',validated=False,plant_referenced=False)
        return value

    def _checkpoint(self,*args):
        cp=super()._checkpoint(*args)
        cp['payload'].update(model=MODEL,scheme=self.scheme)
        cp['sha256']=digest(cp['payload']);return cp

    def restore(self,checkpoint):
        with self._lock:
            self._sources();keys(checkpoint,('payload','sha256'));p=checkpoint['payload']
            if (len(json.dumps(checkpoint,allow_nan=False).encode())>4*1024**2 or digest(p)!=checkpoint['sha256'] or
                p.get('model')!=MODEL or p.get('scheme')!=self.scheme or p.get('source')!=self.source or
                p.get('input_sha256')!=self.identity):raise ValueError('Contact checkpoint identity/size')
            calls=p.get('calls')
            if not isinstance(calls,list) or len(calls)>MAX_CALLS:raise ValueError('Contact replay budget')
            other=ContactHEMNetwork(self.request,scheme=self.scheme)
            for call in calls:
                keys(call,('duration_s','command'));command=call['command']
                if command is None:other.advance(call['duration_s'])
                else:
                    keys(command,('controls','expected_revision','controller_state'))
                    other.step(command['controls'],call['duration_s'],expected_revision=command['expected_revision'],controller_state=command['controller_state'])
            if other.checkpoint()!=checkpoint:raise ValueError('Contact replay differs')
            self._sources()
            self._y,self._time,self._controls,self._controller,self._calls=other._y,other._time,other._controls,other._controller,other._calls
            return self.snapshot()
