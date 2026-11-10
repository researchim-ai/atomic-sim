"""Three-wave HLLC with bounded signals and explicit IF97 star admission.

Toro/Spruce/Speares1994, doi:10.1007/BF01414629. No ideal-gas gamma,
state clipping, entropy fix or silent Rusanov fallback is used here.
"""
import math
import numpy as np
from atomic_model_hem_qualified import eos
from .inverse import liquid_first_rho_u as _inverse

SOURCE = 'https://doi.org/10.1007/BF01414629'


def physical(state, velocity, energy_reference):
    rho = state.density_kg_m3
    centered = rho * (state.internal_energy_j_kg - energy_reference + .5*velocity**2)
    conserved = np.array([rho, rho*velocity, centered])
    flux = np.array([rho*velocity, rho*velocity**2+state.pressure_pa,
                     velocity*(centered+state.pressure_pa)])
    return conserved, flux


def _hllc(left, right, velocity_left, velocity_right, *, wave_bound, energy_reference, require_star_envelope=False, sound):
    values = (velocity_left, velocity_right, wave_bound, energy_reference,
              left.pressure_pa, right.pressure_pa, left.density_kg_m3, right.density_kg_m3,
              left.internal_energy_j_kg, right.internal_energy_j_kg)
    if any(not math.isfinite(v) for v in values) or wave_bound<=0:
        raise ValueError('Finite positive HLLC envelope required')
    if any(s.region!=1 or s.phase!='liquid' or s.density_kg_m3<=0 for s in (left,right)):
        raise ValueError('HLLC release admits single-phase liquid only')
    for state,velocity in ((left,velocity_left),(right,velocity_right)):
        c=sound(state)
        if abs(velocity)+c>wave_bound:raise ValueError('HLLC wave bound too small')
    ul,fl=physical(left,velocity_left,energy_reference)
    ur,fr=physical(right,velocity_right,energy_reference)
    # This is an exact equality, not a tolerance-based projection onto a contact.
    if left.pressure_pa==right.pressure_pa and velocity_left==velocity_right:
        return fl if velocity_left>=0 else fr
    sl,sr=-wave_bound,wave_bound
    dl=left.density_kg_m3*(sl-velocity_left)
    dr=right.density_kg_m3*(sr-velocity_right)
    denominator=dl-dr
    if denominator==0:raise ValueError('Degenerate HLLC contact speed')
    sm=(right.pressure_pa-left.pressure_pa+dl*velocity_left-dr*velocity_right)/denominator
    if not sl<sm<sr:raise ValueError('HLLC contact outside signal fan')
    stars=[]
    for state,v,s,u,f in ((left,velocity_left,sl,ul,fl),(right,velocity_right,sr,ur,fr)):
        pressure=state.pressure_pa+state.density_kg_m3*(s-v)*(sm-v)
        rho=state.density_kg_m3*(s-v)/(s-sm)
        specific=u[2]/state.density_kg_m3+(sm-v)*(sm+state.pressure_pa/(state.density_kg_m3*(s-v)))
        if not math.isfinite(rho) or rho<=0 or not eos.P_MIN<=pressure<=eos.P_MAX:
            raise ValueError('Inadmissible HLLC star density/pressure')
        star=np.array([rho,rho*sm,rho*specific])
        internal=energy_reference+specific-.5*sm**2
        admitted=_inverse(float(rho),float(internal),float(pressure),state.temperature_k)
        if admitted.region!=1 or admitted.phase!='liquid':raise ValueError('HLLC star leaves liquid chart')
        if require_star_envelope and abs(sm)+sound(admitted)>wave_bound:
            raise ValueError('Local HLLC star exceeds signal envelope')
        stars.append(f+s*(star-u))
    result=stars[0] if sm>=0 else stars[1]
    if not np.isfinite(result).all():raise ValueError('Nonfinite HLLC flux')
    return result


def _local_envelope(left,right,velocity_left,velocity_right,*,maximum,sound):
    """1% margin over input acoustic signals, bounded by the global CFL cap.

    This is not a universal nonlinear wave bound. Both EOS-admitted stars
    are checked by local_hllc; unsupported fans are refused, never expanded.
    """
    velocities=(velocity_left,velocity_right)
    if not math.isfinite(maximum) or maximum<=0 or any(not math.isfinite(v) for v in velocities):
        raise ValueError('Finite local HLLC envelope required')
    for state in (left,right):
        if state.region!=1 or state.phase!='liquid':raise ValueError('Local HLLC requires liquid')
    bound=1.01*max(abs(v)+sound(s) for s,v in zip((left,right),velocities))
    if bound>maximum:raise ValueError('Local HLLC exceeds global wave cap')
    return bound



def _sound_speed(state):return eos.characteristic_speeds(state)[1]


class _FluxEvaluation:
    """Exact sound cache scoped to one RHS stage or one public face call.

    The graph seeds it only from its just-validated primitive states/sounds.
    Full immutable State keys: no rounded keys, cross-step or persistent cache.
    Star EOS inversions and both star-envelope tests remain unconditional.
    """
    def __init__(self,states=(),speeds=()):
        self._sounds={state:float(speed) for state,speed in zip(states,speeds,strict=True)}

    def sound(self,state):
        if state not in self._sounds:self._sounds[state]=_sound_speed(state)
        return self._sounds[state]

    def hllc(self,left,right,velocity_left,velocity_right,*,wave_bound,energy_reference):
        return _hllc(left,right,velocity_left,velocity_right,wave_bound=wave_bound,
                     energy_reference=energy_reference,sound=self.sound)

    def local_hllc(self,left,right,velocity_left,velocity_right,*,wave_bound,energy_reference):
        bound=_local_envelope(left,right,velocity_left,velocity_right,maximum=wave_bound,sound=self.sound)
        return _hllc(left,right,velocity_left,velocity_right,wave_bound=bound,
                     energy_reference=energy_reference,require_star_envelope=True,sound=self.sound)


def hllc(left,right,velocity_left,velocity_right,*,wave_bound,energy_reference,require_star_envelope=False):
    return _hllc(left,right,velocity_left,velocity_right,wave_bound=wave_bound,
                 energy_reference=energy_reference,require_star_envelope=require_star_envelope,sound=_sound_speed)


def local_envelope(left,right,velocity_left,velocity_right,*,maximum):
    return _local_envelope(left,right,velocity_left,velocity_right,maximum=maximum,sound=_sound_speed)


def local_hllc(left,right,velocity_left,velocity_right,*,wave_bound,energy_reference):
    return _FluxEvaluation().local_hllc(left,right,velocity_left,velocity_right,
                                       wave_bound=wave_bound,energy_reference=energy_reference)
