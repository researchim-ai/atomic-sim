"""Measured station environment; Gym adapter is loaded only when requested."""
from .model import EnvConfig,StationEnv,ActionError

__all__=['EnvConfig','StationEnv','ActionError','StationGymEnv']


def __getattr__(name):
    if name=='StationGymEnv':
        from .gym import StationGymEnv
        return StationGymEnv
    raise AttributeError(name)
