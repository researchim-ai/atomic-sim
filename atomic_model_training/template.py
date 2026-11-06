"""Copy only a pristine t=0 plant, never a live solver or callback context."""

from copy import deepcopy
from threading import RLock

from atomic_model_controlled_plant.session import Session, identity
from atomic_model_workspace.common import digest
from plants.ledger import HorizonLedger

from .cache import enable


class Template:
    def __init__(self):
        self._source = digest(identity())
        self._pristine = Session()
        self._checkpoint = self._pristine.checkpoint()
        self._memo_bank = {}

    def new(self):
        p = self._pristine.plant
        if (
            digest(identity()) != self._source
            or self._pristine.checkpoint() != self._checkpoint
            or p.time_s != 0
            or self._pristine.history
            or p._hybrid is not None
            or p._continuation is not None
        ):
            raise ValueError("Template source or pristine state changed")
        result = object.__new__(Session)
        result.config = deepcopy(self._pristine.config)
        result.initialization = deepcopy(self._pristine.initialization)
        result.plant = deepcopy(p)
        enable(result.plant, self._memo_bank)
        result.ledger = HorizonLedger()
        result.ledger.attach(result.plant)
        result.history = []
        result.revision = 0
        result.lock = RLock()
        if result.checkpoint() != self._checkpoint:
            raise ValueError("Template clone differs from direct initialization")
        return result
