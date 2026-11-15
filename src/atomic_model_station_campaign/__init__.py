"""Local deterministic station episodes and resource-isolated campaigns."""
from .spec import Episode,Limits

__all__=['Episode','Limits','run_campaign']


def __getattr__(name):
    if name=='run_campaign':
        from .runner import run_campaign
        return run_campaign
    raise AttributeError(name)
