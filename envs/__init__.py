"""Gymnasium environments for reactor control"""

try:
    from .gym_reactor import ReactorEnv, ReactorEnvContinuous
    __all__ = ['ReactorEnv', 'ReactorEnvContinuous']
except ImportError:
    # Gymnasium не установлен
    __all__ = []

