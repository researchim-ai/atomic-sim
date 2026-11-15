"""Persistent append journal with caller-owned rows and constant-cost cloning."""
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class _Node:
    previous: object
    value: object


class Journal:
    """Mutable head, immutable shared prefix; never hand internal rows to callers.

    Transaction copies share a prefix, while appends affect only their own head.
    Serialization materializes ordinary lists. This is not a disk checkpoint or
    a replacement for replay validation.
    """
    def __init__(self, rows=()):
        self._head = None
        self._length = 0
        for row in rows:
            self.append(row)

    def append(self, row):
        self._head = _Node(self._head, deepcopy(row))
        self._length += 1

    def __len__(self):
        return self._length

    def __iter__(self):
        nodes = []
        current = self._head
        while current is not None:
            nodes.append(current)
            current = current.previous
        for node in reversed(nodes):
            yield deepcopy(node.value)

    def __deepcopy__(self, memo):
        result = type(self)()
        result._head = self._head
        result._length = self._length
        memo[id(self)] = result
        return result


def history_budget(value):
    if type(value) is not int or not 256 <= value <= 8192:
        raise ValueError('History budget must be an integer in 256..8192')
    return value
