"""Scientific episode execution, causal command schedule and replay anchors."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_station_faults import FaultSession
from atomic_model_station_faults.model import signature as fault_signature
from .spec import Episode


def signature():
    return dict(faults=fault_signature(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob('*.py'))})


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def run_episode(episode, resume=None):
    if type(episode) is not Episode:raise ValueError('Episode required')
    # Revalidate mutable nested command dictionaries at the execution boundary.
    episode=Episode.from_payload(episode.payload())
    owner=FaultSession(episode.faults,tick_s=episode.tick_s,seed=episode.seed,
        react_on_alarm=episode.react_on_alarm,max_step_s=episode.max_step_s,rtol=episode.rtol,
        tracking_tau_s=episode.tracking_tau_s,history_budget_steps=max(256,episode.ticks))
    source=signature();commands={c['tick']:{k:deepcopy(v) for k,v in c.items() if k!='tick'} for c in episode.commands}
    chain=digest(dict(episode=episode.payload(),source=source))
    samples=[];anchors={};completed=0;rejection=None;rollback=None
    maxima=dict(energy_relative=0.,mass_relative=0.,boron_relative=0.)
    resume_payload=unseal(deepcopy(resume)) if resume is not None else None
    if resume_payload is not None:
        if set(resume_payload)!={'episode_sha256','completed_ticks','checkpoint','trajectory_sha256','maxima'}:
            raise ValueError('Campaign anchor fields')
        if resume_payload['episode_sha256']!=digest(episode.payload()):raise ValueError('Anchor belongs to another episode')
        tick=resume_payload['completed_ticks']
        if type(tick) is not int or not 0<=tick<=episode.ticks:raise ValueError('Anchor epoch')
        checkpoint=unseal(resume_payload['checkpoint'])
        if checkpoint.get('source')!=source['faults'] or len(checkpoint.get('history',[]))!=tick:
            raise ValueError('Anchor source/history')

    def anchor():
        return seal(dict(episode_sha256=digest(episode.payload()),completed_ticks=completed,
            checkpoint=owner.checkpoint(),trajectory_sha256=chain,maxima=deepcopy(maxima)))

    def at_boundary():
        if completed in episode.checkpoint_ticks or (resume_payload is not None and completed==resume_payload['completed_ticks']):
            value=anchor()
            if resume_payload is not None and completed==resume_payload['completed_ticks'] and unseal(value)!=resume_payload:
                raise ValueError('Restart anchor state/trajectory differs')
            if completed in episode.checkpoint_ticks:anchors[str(completed)]=value

    at_boundary()
    for tick in range(episode.ticks):
        kwargs=deepcopy(commands.get(tick,{}))
        if tick==0 and episode.automatic:
            kwargs['modes']=dict({k:'auto' for k in owner.control.loops},**(kwargs.get('modes') or {}))
        # A costly whole-session checkpoint is needed only for an actual rejected
        # input. Preserve a cheap structural transaction clone before the call.
        before=deepcopy(owner)
        try:result=owner.step(**kwargs)
        except ValueError as error:
            rejection=str(error);rollback=owner.checkpoint()==before.checkpoint()
            if resume_payload is not None and tick<resume_payload['completed_ticks']:
                raise ValueError('Restart prefix rejected before saved anchor') from error
            break
        completed=tick+1
        row=owner.control.plant.snapshot()
        for name in maxima:maxima[name]=max(maxima[name],row[name])
        chain=digest(dict(previous=chain,result=result,physical_state=owner.control.plant.state.tolist()))
        if completed%episode.record_every==0 or completed==episode.ticks:
            samples.append(dict(tick=completed,result=result,physical=dict(time_s=row['time_s'],
                powers=row['powers'],levels=row['level_readouts'],turbine_speed_rad_s=row['turbine_speed_rad_s'],
                primary_pump_speed_rad_s=row['primary_pump_speed_rad_s'],
                pressure_pa=row['steam_header']['pressure_pa'],energy_relative=row['energy_relative'],
                mass_relative=row['mass_relative'],boron_relative=row['boron_relative'])))
        at_boundary()
    final=owner.checkpoint()
    unchanged=source==signature()
    success=bool(unchanged and (completed==episode.ticks if not episode.expected_rejection else rejection is not None and rollback))
    if any(maxima[n]>(1e-6 if n=='energy_relative' else 1e-10) for n in maxima):success=False
    return seal(dict(episode=episode.payload(),source=source,source_unchanged=unchanged,
        accepted=success,status='completed' if rejection is None else 'rejected',completed_ticks=completed,
        completed_time_s=owner.control.plant.time,trajectory_sha256=chain,maxima=maxima,
        rejection=rejection,rollback_exact=rollback,samples=samples,checkpoints=anchors,final=final,
        physical_state=owner.control.plant.state.tolist(),full_r1_matrix=False,validated=False))
