"""Bounded exact-argument memoization of immutable strict EOS results.

No input rounding, changed seed, stock projection or relaxed EOS tolerance.
Only successful evaluations are retained; failures still use the original kernel.
Caches contain only immutable EOS values; physical state and solvers are never shared.
The frozen core source remains untouched.
"""

from collections import OrderedDict
from threading import RLock

from atomic_model_plant_extension.volume import StrictVolume
from components.condenser import DynamicCondenser


class ExactMemo:
    def __init__(self):
        self.values = OrderedDict()
        self.lock = RLock()

    def __len__(self):
        with self.lock:
            return len(self.values)

    def lookup(self, key):
        with self.lock:
            value = self.values.get(key)
            if value is not None:
                self.values.move_to_end(key)
            return value

    def remember(self, key, value):
        with self.lock:
            self.values[key] = value
            self.values.move_to_end(key)
            if len(self.values) > 128:
                self.values.popitem(last=False)


class CachedVolume(StrictVolume):
    def sync(self):
        if self.mass_kg <= 0 or not hasattr(self, "_inverse_seed"):
            return super().sync()
        density = float(self.mass_kg / self.volume_m3)
        energy = float(self.internal_energy_j / self.mass_kg)
        key = (density, energy, *self._inverse_seed)
        cache = self._exact_cache
        cached = cache.lookup(key)
        if cached is not None:
            point, energy_error, volume_error = cached
            self.thermo = point
            self.energy_mismatch_j_kg = energy_error
            self.volume_mismatch = volume_error
            self.pressure_pa = point.p_pa
            self.temperature_k = point.T_k
            self._thermo_key = (density, energy)
            return point
        point = super().sync()
        cache.remember(key, (point, self.energy_mismatch_j_kg, self.volume_mismatch))
        return point


class CachedCondenser(DynamicCondenser):
    def state(self):
        if self.mass_kg <= 0:
            return super().state()
        key = (
            self.mass_kg / self.volume_m3,
            self.energy_j / self.mass_kg,
            self._pressure_guess_pa,
            self._temperature_guess_k,
        )
        cache = self._exact_cache
        point = cache.lookup(key)
        if point is not None:
            self._pressure_guess_pa = point.p_pa
            self._temperature_guess_k = point.T_k
            return point
        point = super().state()
        cache.remember(key, point)
        return point


def enable(plant, memo_bank):
    for name, volume in plant.network.volumes.items():
        if type(volume) is not StrictVolume:
            raise ValueError("Exact cache supports only the frozen strict volume type")
        volume.__class__ = CachedVolume
        volume._exact_cache = memo_bank.setdefault(name, ExactMemo())

    if type(plant.condenser) is not DynamicCondenser:
        raise ValueError("Exact cache requires the frozen condenser type")
    plant.condenser.__class__ = CachedCondenser
    plant.condenser._exact_cache = memo_bank.setdefault("__condenser__", ExactMemo())
