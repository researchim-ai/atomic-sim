"""Atomic scenario owner; only observation/detection/reaction receipts are public."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path

from atomic_model_multiphysics.provenance import seal, unseal
from atomic_model_station.faults import PWRFaultSession
from atomic_model_station.journal import Journal,history_budget
from atomic_model_station_control import ControlSession, feed_control_session, control_signature
from atomic_model_station_signals import StationSignals, Sensor
from atomic_model_station_signals.model import CATALOG
from .spec import Fault, canonical
from .monitor import Monitor, MonitorConfig


def signature():
    from atomic_model_station_equilibrium import signature as equilibrium_signature
    return dict(control=control_signature(),equilibrium=equilibrium_signature(), files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob('*.py'))})


class FaultSession:
    def __init__(self, faults=(), *, tick_s=1., seed=0, react_on_alarm=False,
                 monitor=None, max_step_s=1., rtol=1e-7, history_budget_steps=256, tracking_tau_s=10.,inventory_deadband_m=0.,equilibrium_start=False,film_model="dittus_boelter"):
        from atomic_model_station.film import FilmConfig
        from atomic_model_station.film_correlations import validate_model
        validate_model(film_model)
        self.faults = canonical(faults)
        if type(react_on_alarm) is not bool:
            raise ValueError('Boolean alarm reaction')
        if type(equilibrium_start) is not bool:raise ValueError('Boolean equilibrium initialization')
        self.react_on_alarm = react_on_alarm
        self.monitor = Monitor(monitor)
        self.reacted = False
        self.history = Journal()
        history_budget(history_budget_steps)
        self.initial = dict(faults=[asdict(f) for f in self.faults], tick_s=tick_s, seed=seed,
                            react_on_alarm=react_on_alarm, monitor=asdict(self.monitor.config),
                            max_step_s=max_step_s, rtol=rtol, history_budget_steps=history_budget_steps, tracking_tau_s=tracking_tau_s,inventory_deadband_m=inventory_deadband_m,equilibrium_start=equilibrium_start,film_model=film_model)
        reference = feed_control_session(tick_s=tick_s, seed=seed, integration_step_s=max_step_s, rtol=rtol, tracking_tau_s=tracking_tau_s,inventory_deadband_m=inventory_deadband_m)
        from atomic_model_station_equilibrium import EquilibriumConfig
        plant = PWRFaultSession(faults=[f for f in self.faults if f.domain == 'equipment'],
                                fault_tick_s=tick_s, max_step_s=max_step_s, rtol=rtol, history_budget_steps=history_budget_steps,
                                equilibrium=EquilibriumConfig() if equilibrium_start else None,film=FilmConfig(correlation=film_model))
        loop_configs=[v.config for v in reference.loops.values()]
        if equilibrium_start:
            from atomic_model_station_equilibrium.control import references
            loop_configs=references(loop_configs,plant)
        instruments = StationSignals([*reference.instruments.sensors,
            Sensor('pressure_reference', 'steam_pressure_pa', lag_s=.2)], tick_s=tick_s, seed=seed,
            faults=[f for f in self.faults if f.domain == 'sensor'])
        names = {s.name for s in instruments.sensors}
        if self.monitor.config.sensor not in names or self.monitor.config.reference not in names:
            raise ValueError('Monitor channel not installed')
        quantities={s.name:s.quantity for s in instruments.sensors}
        c=self.monitor.config
        if quantities[c.sensor]!=quantities[c.reference] or CATALOG[quantities[c.sensor]][1]!=c.unit:
            raise ValueError('Monitor channels require the same quantity and declared unit')
        self.control = ControlSession(plant, instruments=instruments,
                                      loops=loop_configs)
        self.detection = self.monitor.advance(self.control.instruments.operator_frame())

    def step(self, operator=None, *, modes=None, targets=None, reset=False):
        trial = deepcopy(self)
        command = deepcopy(operator) if operator is not None else {}
        if type(command) is not dict:
            raise ValueError('Operator command dictionary')
        reaction = None
        if trial.react_on_alarm and trial.detection['alarms'] and not trial.reacted:
            command.update(scram=True, turbine_trip=True)
            trial.reacted = True
            reaction = dict(kind='stop_requested', based_on=deepcopy(trial.detection['alarms']),
                            observed_at_s=trial.detection['time_s'])
        result = trial.control.step(command or None, modes=modes, targets=targets, reset=reset)
        trial.detection = trial.monitor.advance(result['observation'])
        trial.history.append(dict(operator=deepcopy(operator), modes=deepcopy(modes),
                                  targets=deepcopy(targets), reset=reset))
        self.__dict__.update(trial.__dict__)
        return dict(**result, detection=deepcopy(self.detection), reaction=reaction)

    def checkpoint(self):
        return seal(dict(format='station-fault-scenario/v1', source=signature(),
                         initial=deepcopy(self.initial), history=list(self.history),
                         control=self.control.checkpoint(), monitor=self.monitor.state(),
                         detection=deepcopy(self.detection), reacted=self.reacted))

    @classmethod
    def restore(cls, saved):
        p = unseal(deepcopy(saved))
        if p['format'] != 'station-fault-scenario/v1' or p['source'] != signature():
            raise ValueError('Fault scenario source/format')
        if type(p['history']) is not list or len(p['history']) > history_budget(p['initial']['history_budget_steps']):
            raise ValueError('Fault scenario replay budget')
        args = deepcopy(p['initial'])
        args['faults'] = [Fault(**f) for f in args['faults']]
        args['monitor'] = MonitorConfig(**args['monitor'])
        obj = cls(**args)
        for row in p['history']:
            obj.step(**row)
        if obj.checkpoint() != saved:
            raise ValueError('Fault scenario checkpoint replay differs')
        return obj
