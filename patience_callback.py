"""Callback classes which terminate solving when no better solution was found in given time."""

from time import time

import gurobipy as gp
from gurobipy import GRB


class PatienceOutsideLocalSearch:
    """Terminates solving when no solution is found in given time after first was found."""

    def __init__(self, patience: float | int):
        self.time_last_sol_found: float | None = None
        self.patience = patience

    def __call__(self, model: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL:
            self.time_last_sol_found = time()

        elif self.time_last_sol_found is None:
            return

        elif time() - self.time_last_sol_found > self.patience:
            model.terminate()


class PatienceInsideLocalSearch:
    """Terminates solving when no solution is found in given time."""

    def __init__(self, patience: float | int):
        self.reference_time = time()
        self.patience = patience

    def __call__(self, model: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL:
            self.reference_time = time()

        elif time() - self.reference_time > self.patience:
            model.terminate()
