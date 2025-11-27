"""Callback classes which terminate solving when no better solution was found in given time."""

from time import time

import gurobipy as gp
from gurobipy import GRB


class PatienceOutsideLocalSearch:
    """Terminates solving when no solution is found in given time after first was found."""

    def __init__(self, patience: float | int):
        self.time_last_sol_found: float | None = None
        self.patience = patience

    def __call__(self, model: gp.Model, where: int):  # intenum oder typedef z.B gurobidef
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


class GurobiAloneProgressTracker:
    """Tracks the progress of Gurobi"""

    def __init__(self, solution_summaries: list[dict[str, int | float]]):
        self.best_obj = -GRB.MAXINT
        self.best_bound = GRB.MAXINT
        self.solution_summaries = solution_summaries

    def __call__(self, model: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL:
            best_objective = int(model.cbGet(GRB.Callback.MIPSOL_OBJBST) + 1e-6)
            best_bound = int(model.cbGet(GRB.Callback.MIPSOL_OBJBND) + 1e-6)

            if best_objective > self.best_obj or best_bound < self.best_bound:
                self.best_obj = best_objective
                self.best_bound = best_bound

                summary: dict[str, int | float] = {
                    "objective": best_objective,
                    "bound": best_bound,
                    "runtime": model.cbGet(GRB.Callback.RUNTIME),
                }
                self.solution_summaries.append(summary)


class InitialOptimizationTracker:

    def __init__(self, solution_summaries: list[dict[str, int | float | str]], start_time: float):
        self.best_obj = -GRB.MAXINT
        self.solution_summaries = solution_summaries
        self.start_time = start_time

    def __call__(self, model: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL:
            best_objective = int(model.cbGet(GRB.Callback.MIPSOL_OBJBST) + 1e-6)

            if best_objective > self.best_obj:
                self.best_obj = best_objective
                summary: dict[str, int | float | str] = {
                    "objective": best_objective,
                    "runtime": time() - self.start_time,
                    "station": "initial_optimization",
                }
                self.solution_summaries.append(summary)
