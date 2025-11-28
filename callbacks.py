"""Callback classes which terminate solving when no better solution was found in given time."""

from time import time

import gurobipy as gp
from gurobipy import GRB


class PatienceShake:
    """Terminates solving when no solution is found in given time after first was found."""

    def __init__(
        self,
        patience: float | int,
        start_time: float,
        best_obj: int,
        solution_summaries: list[dict[str, int | float | str]],
    ):
        self.time_last_sol_found: float | None = None
        self.patience = patience
        self.start_time = start_time
        self.best_obj = best_obj
        self.solution_summaries = solution_summaries

    def __call__(self, model: gp.Model, where: int):  # intenum oder typedef z.B gurobidef
        if where == GRB.Callback.MIPSOL:
            self.time_last_sol_found = time()

            current_objective = int(model.cbGet(GRB.Callback.MIPSOL_OBJ) + 1e-6)

            if current_objective > self.best_obj:
                self.best_obj = current_objective

                summary: dict[str, int | float | str] = {
                    "objective": current_objective,
                    "runtime": self.time_last_sol_found - self.start_time,
                    "station": "shake",
                }
                self.solution_summaries.append(summary)

        elif self.time_last_sol_found is None:
            return

        elif time() - self.time_last_sol_found > self.patience:
            model.terminate()


class PatienceVND:
    """Terminates solving when no solution is found in given time."""

    def __init__(
        self,
        patience: float | int,
        start_time: float,
        best_obj: int,
        solution_summaries: list[dict[str, int | float | str]],
    ):
        self.reference_time = time()
        self.patience = patience
        self.start_time = start_time
        self.best_obj = best_obj
        self.solution_summaries = solution_summaries

    def __call__(self, model: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL:
            self.reference_time = time()

            current_objective = int(model.cbGet(GRB.Callback.MIPSOL_OBJ) + 1e-6)

            if current_objective > self.best_obj:
                self.best_obj = current_objective

                summary: dict[str, int | float | str] = {
                    "objective": current_objective,
                    "runtime": self.reference_time - self.start_time,
                    "station": "vnd",
                }
                self.solution_summaries.append(summary)

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
            current_objective = int(model.cbGet(GRB.Callback.MIPSOL_OBJ) + 1e-6)
            best_bound = int(model.cbGet(GRB.Callback.MIPSOL_OBJBND) + 1e-6)

            if current_objective > self.best_obj or best_bound < self.best_bound:
                self.best_obj = current_objective
                self.best_bound = best_bound

                summary: dict[str, int | float] = {
                    "objective": current_objective,
                    "bound": best_bound,
                    "runtime": model.cbGet(GRB.Callback.RUNTIME),
                }
                self.solution_summaries.append(summary)


class InitialOptimizationTracker:

    def __init__(
        self,
        patience: float | int,
        solution_summaries: list[dict[str, int | float | str]],
        start_time: float,
    ):
        self.patience = patience
        self.time_last_sol_found: float | None = None
        self.best_obj = -GRB.MAXINT
        self.solution_summaries = solution_summaries
        self.start_time = start_time

    def __call__(self, model: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL:
            self.time_last_sol_found = time()
            current_objective = int(model.cbGet(GRB.Callback.MIPSOL_OBJ) + 1e-6)

            if current_objective > self.best_obj:
                self.best_obj = current_objective
                summary: dict[str, int | float | str] = {
                    "objective": current_objective,
                    "runtime": self.time_last_sol_found - self.start_time,
                    "station": "initial optimization",
                }
                self.solution_summaries.append(summary)

        elif self.time_last_sol_found is None:
            return

        elif time() - self.time_last_sol_found > self.patience:
            model.terminate()


class SimpleVNDTracker:

    def __init__(
        self,
        solution_summaries: list[dict[str, int | float | str]],
        start_time: float,
        best_obj: int,
    ):
        self.solution_summaries = solution_summaries
        self.start_time = start_time
        self.best_obj = best_obj

    def __call__(self, model: gp.Model, where: int):
        if where == GRB.Callback.MIPSOL:
            current_objective = int(model.cbGet(GRB.Callback.MIPSOL_OBJ) + 1e-6)

            if current_objective > self.best_obj:
                self.best_obj = current_objective

                summary: dict[str, int | float | str] = {
                    "objective": current_objective,
                    "runtime": time() - self.start_time,
                    "station": "vnd",
                }
                self.solution_summaries.append(summary)
