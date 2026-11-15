"""Opt-in equipment faults at explicit control-clock boundaries.

Requested controls remain operator/controller intent. Effective controls are
used inside physical balances. A seized bypass stem defeats closing as well
as opening, including protective requests; no safety action repairs a fault.
"""
from copy import deepcopy
from dataclasses import asdict

from atomic_model_multiphysics.provenance import seal, unseal, sha
from atomic_model_station_faults.spec import Fault, canonical, finite
from atomic_model_station_faults import spec as fault_spec
from .journal import history_budget
from .bypass import PWRBypassSession, BypassConfig, bypass_signature
from .pwr_train import TrainConfig
from .cooling import CoolingConfig
from .feed import FeedConfig
from .power import BackupConfig
from .core import CoreConfig
from .primary import PrimaryConfig
from .film import FilmConfig
from .steam import SteamConfig
from .stages import StageConfig


class PWRFaultSession(PWRBypassSession):
    def __init__(self, *args, faults=(), fault_tick_s=1., history_budget_steps=256, **kwargs):
        self.history_budget_steps = history_budget(history_budget_steps)
        self.equipment_faults = canonical(faults, domain='equipment')
        if not finite(fault_tick_s) or not .01 <= fault_tick_s <= 60:
            raise ValueError('Equipment fault clock')
        self.fault_tick_s = float(fault_tick_s)
        self.active_faults = ()
        self.stuck_positions = {}
        self.fault_events = []
        super().__init__(*args, **kwargs)

    def _maximum_history_steps(self):
        return self.history_budget_steps

    def _bypass_target(self):
        stuck = next((f for f in self.active_faults if f.kind == 'bypass_stuck'), None)
        if stuck is not None:
            return self.stuck_positions[stuck.name]
        return super()._bypass_target()

    def _effective_controls(self):
        effective = dict(self.controls)
        for fault in self.active_faults:
            if fault.kind == 'pump_off':
                effective[fault.target + '_pump_on'] = False
            elif fault.kind == 'drive_limit':
                key = fault.target + '_pump_fraction'
                effective[key] = min(effective[key], fault.value)
            elif fault.kind == 'cooling_loss':
                effective['external_water_fraction'] *= fault.value
        return effective

    def _rhs(self, t, y):
        if not self.active_faults or getattr(self, '_fault_controls_active', False):
            return super()._rhs(t, y)
        requested = self.controls
        self.controls = self._effective_controls()
        self._fault_controls_active = True
        try:
            return super()._rhs(t, y)
        finally:
            self.controls = requested
            self._fault_controls_active = False

    def _admit(self, y):
        if not self.active_faults or getattr(self, '_fault_controls_active', False):
            return super()._admit(y)
        requested = self.controls
        self.controls = self._effective_controls()
        self._fault_controls_active = True
        try:
            return super()._admit(y)
        finally:
            self.controls = requested
            self._fault_controls_active = False

    def _select_faults(self):
        epoch = round(self.time / self.fault_tick_s)
        selected = tuple(f for f in self.equipment_faults if f.active(epoch))
        old = {f.name for f in self.active_faults}
        new = {f.name for f in selected}
        for name in sorted(old - new):
            self.fault_events.append(dict(time_s=self.time, name=name, transition='ended'))
        for fault in selected:
            if fault.name not in old:
                if fault.kind == 'bypass_stuck':
                    self.stuck_positions[fault.name] = float(self.state[self.bypass_index])
                self.fault_events.append(dict(time_s=self.time, name=fault.name, transition='started'))
        self.active_faults = selected

    def advance(self, duration_s, patch=None):
        if getattr(self, '_fault_advancing', False):
            return super().advance(duration_s, patch)
        if not finite(duration_s) or duration_s != self.fault_tick_s:
            raise ValueError('Equipment fault advance must match declared tick')
        clock = self.time / self.fault_tick_s
        if abs(clock - round(clock)) > 1e-8:
            raise ValueError('Equipment fault clock alignment')
        trial = deepcopy(self)
        trial._select_faults()
        trial._fault_advancing = True
        try:
            result = super(PWRFaultSession, trial).advance(duration_s, patch)
        finally:
            trial._fault_advancing = False
        self.__dict__.update(trial.__dict__)
        return result

    def snapshot(self):
        result = super().snapshot()
        if getattr(self, 'bypass_ready', False):
            result['equipment_fault_truth'] = dict(
                active=[f.name for f in self.active_faults],
                effective_controls=self._effective_controls(),
                stuck_positions=deepcopy(self.stuck_positions), events=deepcopy(self.fault_events))
        return result

    def checkpoint(self):
        payload = unseal(super().checkpoint())
        payload['equipment_faults'] = [asdict(f) for f in self.equipment_faults]
        payload['fault_tick_s'] = self.fault_tick_s
        payload['history_budget_steps'] = self.history_budget_steps
        payload['fault_spec_sha256'] = sha(fault_spec.__file__)
        payload['fault_state'] = dict(active=[f.name for f in self.active_faults],
                                     stuck_positions=deepcopy(self.stuck_positions), events=deepcopy(self.fault_events))
        return seal(payload)

    @classmethod
    def restore(cls, saved):
        p = unseal(deepcopy(saved))
        if p['fault_spec_sha256'] != sha(fault_spec.__file__) or p['bypass_source'] != bypass_signature() or type(p['history']) is not list or len(p['history']) > history_budget(p['history_budget_steps']):
            raise ValueError('Fault plant source/history')
        owner = cls(TrainConfig(**p['config']), cooling=CoolingConfig(**p['cooling_config']),
                    feed=FeedConfig(**p['feed_config']), backup=BackupConfig(**p['backup_config']),
                    core=CoreConfig(**p['core_config']), primary=PrimaryConfig(**p['primary_config']),
                    film=FilmConfig(**p['film_config']), steam=SteamConfig(**p['steam_config']),
                    stages=StageConfig(**p['stage_config']), bypass=BypassConfig(**p['bypass_config']),
                    max_step_s=p['max_step_s'], rtol=p['rtol'],
                    faults=[Fault(**f) for f in p['equipment_faults']], fault_tick_s=p['fault_tick_s'], history_budget_steps=p['history_budget_steps'])
        for row in p['history']:
            owner.advance(**row)
        if owner.checkpoint() != saved:
            raise ValueError('Fault plant checkpoint replay differs')
        return owner
