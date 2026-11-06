"""Точный вывод предписанного линейного движения между событиями упора."""

import numpy as np


class LinearComponentOutput:
    def __init__(self, dense, index, t0, position, speed, final_position):
        self.dense = dense
        self.ts = dense.ts
        self.index, self.t0, self.position, self.speed = index, t0, position, speed
        self.final_position = final_position

    def __call__(self, time):
        result = np.asarray(self.dense(time), dtype=float).copy()
        result[self.index] = np.clip(
            self.position + self.speed * (np.asarray(time) - self.t0), 0.0, 1.0
        )
        result[self.index] = np.where(
            np.asarray(time) == self.ts[-1], self.final_position, result[self.index]
        )
        return result
