"""Finite mass dosing through real boron-flow commands, with atomic execution.

The inventory forecast includes the entire eventual closing tail, but only the
specified finite tail is simulated. It does not predict thermal admissibility.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
import hashlib
import json
import math
from pathlib import Path

from scipy.optimize import brentq
from .pwr_boron import BoronExchangePWRSession


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _planner_source():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _owner(session):
    if type(session) is not BoronExchangePWRSession:
        raise ValueError('Exact finite boron session required')


def _mass(open_s,tail_s,flow,initial_flow,tau):
    opened=-math.expm1(-open_s/tau);closed=-math.expm1(-tail_s/tau)
    closing_flow=initial_flow+(flow-initial_flow)*opened
    return flow*open_s+(initial_flow-flow)*tau*opened+closing_flow*tau*closed


@dataclass(frozen=True)
class BoronDosePlan:
    requested_mass_kg:float
    target_flow_kg_s:float
    tail_time_constants:float
    opening_duration_s:float
    closing_duration_s:float
    residual_flow_kg_s:float
    future_tail_mass_kg:float
    eventual_transfer_mass_kg:float
    origin_sha256:str
    planner_sha256:str

    def __post_init__(self):
        bounds=dict(requested_mass_kg=(.001,1e5),target_flow_kg_s=(1e-300,100.),
            tail_time_constants=(5.,20.),opening_duration_s=(0.,3600.),closing_duration_s=(.25,100.),
            residual_flow_kg_s=(0.,100.),future_tail_mass_kg=(0.,500.),eventual_transfer_mass_kg=(.001,100500.))
        for name,(lo,hi) in bounds.items():
            value=getattr(self,name)
            if type(value) not in (float,int) or not math.isfinite(value) or not lo<=value<=hi:
                raise ValueError('Bounded finite dose plan required: '+name)
        for value in (self.origin_sha256,self.planner_sha256):
            if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Dose plan SHA256 identity required')

    def steps(self):
        """Fresh explicit native commands, each within the 60-second API limit."""
        result=[]
        for duration,target in ((self.opening_duration_s,self.target_flow_kg_s),(self.closing_duration_s,0.)):
            left=duration;first=True
            while left>0:
                dt=min(left,60.);result.append(dict(duration_s=dt,
                    patch={'boron_flow_target_kg_s':target} if first else None))
                left-=dt;first=False
        return result


def plan_boron_dose(session,*,mass_kg,flow_kg_s,tail_time_constants=5.):
    """Plan a mass transfer including lag; refuse insufficient eventual reserve.

    Uses the current actual flow, including a previous unfinished closing tail.
    Planning is read-only. Thermodynamics and all physical ledgers are checked
    by the native solver when executing the plan.
    """
    _owner(session);c=session.boron_config
    for v in (mass_kg,flow_kg_s,tail_time_constants):
        if type(v) not in (int,float) or not math.isfinite(v):raise ValueError('Finite dose inputs required')
    if not .001<=mass_kg<=1e5 or not 0<flow_kg_s<=c.maximum_flow_kg_s or not 5<=tail_time_constants<=20:
        raise ValueError('Dose 0.001..100000 kg, bounded positive flow and 5..20 closing time constants required')
    if not session.controls['grid_available']:raise ValueError('Dose requires external grid')
    tau=c.actuator_tau_s;q0=float(session.state[session.boron_index+6]);tail_s=tau*tail_time_constants
    horizon=session._maximum_time_s()-session.time-tail_s
    if horizon<0:raise ValueError('Dose exceeds physical time budget')
    minimum=_mass(0.,tail_s,flow_kg_s,q0,tau)
    maximum=_mass(horizon,tail_s,flow_kg_s,q0,tau)
    if not minimum<=mass_kg<=maximum:raise ValueError('Dose outside achievable finite-horizon mass range')
    opening=0. if mass_kg==minimum else brentq(lambda t:_mass(t,tail_s,flow_kg_s,q0,tau)-mass_kg,
        0.,horizon,xtol=1e-12,rtol=1e-14)
    qclose=q0+(flow_kg_s-q0)*(-math.expm1(-opening/tau));residual=qclose*math.exp(-tail_time_constants)
    future=tau*residual;eventual=mass_kg+future
    tanks=session._tanks(session.state);supply=tanks['supply'];receiver=tanks['receiver']
    if supply['mass_kg']-eventual<c.minimum_mass_kg:raise ValueError('Dose including future tail exhausts supply reserve')
    gas=receiver['gas_volume_m3']-eventual/session.boron_rho
    if gas<=0 or receiver['pressure_pa']*(receiver['gas_volume_m3']/gas)**1.4>2e6:
        raise ValueError('Dose including future tail exceeds receiver gas/pressure domain')
    plan=BoronDosePlan(float(mass_kg),float(flow_kg_s),float(tail_time_constants),float(opening),float(tail_s),
        float(residual),float(future),float(eventual),_digest(session.checkpoint()),_planner_source())
    if len(session.history)+len(plan.steps())>session._maximum_history_steps():raise ValueError('Dose exceeds replay history budget')
    return plan


def execute_boron_dose(session,plan):
    """Commit the entire finite dose only if every native interval succeeds.

    Stale or altered plans are refused. No stock, concentration, heat or energy
    is assigned directly. On failure the original session remains unchanged.
    """
    _owner(session)
    if type(plan) is not BoronDosePlan:raise ValueError('BoronDosePlan required')
    if plan.origin_sha256!=_digest(session.checkpoint()) or plan.planner_sha256!=_planner_source():
        raise ValueError('Stale dose plan or planner source')
    expected=plan_boron_dose(session,mass_kg=plan.requested_mass_kg,flow_kg_s=plan.target_flow_kg_s,
        tail_time_constants=plan.tail_time_constants)
    if plan!=expected:raise ValueError('Altered dose plan')
    trial=deepcopy(session);initial=trial._tanks(trial.state);rows=[]
    for step in plan.steps():rows.append(trial.advance(**step))
    final=trial._tanks(trial.state);transferred=initial['supply']['mass_kg']-final['supply']['mass_kg']
    received=final['receiver']['mass_kg']-initial['receiver']['mass_kg']
    q=float(trial.state[trial.boron_index+6])
    if (not math.isclose(transferred,plan.requested_mass_kg,rel_tol=1e-8,abs_tol=1e-7)
            or not math.isclose(received,plan.requested_mass_kg,rel_tol=1e-8,abs_tol=1e-7)
            or not math.isclose(q,plan.residual_flow_kg_s,rel_tol=1e-10,abs_tol=1e-12)):
        raise ValueError('Dose mass/actuator postcondition failed')
    result=dict(plan=asdict(plan),steps=plan.steps(),snapshots=rows,transferred_mass_kg=transferred,
        received_mass_kg=received,residual_flow_kg_s=q,whole_dose_committed=True,
        thermal_admissibility_predicted=False,experimental_validation=False)
    session.__dict__.update(trial.__dict__)
    return result
