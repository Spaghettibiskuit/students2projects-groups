import abc
import time

import gurobipy

from modeling.model_components import ModelComponents
from solving_utilities.callbacks import PatienceShake, PatienceVND
from solving_utilities.solution_reminders import SolutionReminderBase
from utilities import Stations, gurobi_round


class ModelWrapper(abc.ABC):

    def __init__(
        self,
        model_components: ModelComponents,
        model: gurobipy.Model,
        start_time: float,
        solution_summaries: list[dict[str, int | float | str]],
        sol_reminder: SolutionReminderBase,
    ):
        self.model_components = model_components
        self.model = model
        self.start_time = start_time
        self.solution_summaries = solution_summaries
        self.current_solution = sol_reminder
        self.best_found_solution = sol_reminder

    @property
    def status(self) -> int:
        return self.model.Status

    @property
    def objective_value(self) -> int:
        return gurobi_round(self.model.ObjVal)

    @property
    def solution_count(self) -> int:
        return self.model.SolCount

    def eliminate_time_limit(self):
        self.model.Params.TimeLimit = float("inf")

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def eliminate_cutoff(self):
        self.model.Params.Cutoff = float("-inf")

    def set_cutoff(self):
        self.model.Params.Cutoff = round(self.current_solution.objective_value) + 1 - 1e-4

    def optimize(self, patience: int | float, shake: bool = False):
        cb_class = PatienceShake if shake else PatienceVND
        callback = cb_class(
            patience=patience,
            start_time=self.start_time,
            best_obj=self.best_found_solution.objective_value,
            solution_summaries=self.solution_summaries,
        )
        self.model.optimize(callback)
        if self.solution_count > 0 and self.objective_value > callback.best_obj:
            summary: dict[str, int | float | str] = {
                "objective": self.objective_value,
                "runtime": time.time() - self.start_time,
                "station": Stations.SHAKE if shake else Stations.VND,
            }
            self.solution_summaries.append(summary)

    def new_best_found(self) -> bool:
        return self.current_solution.objective_value > self.best_found_solution.objective_value

    @abc.abstractmethod
    def store_solution(self):
        pass

    @abc.abstractmethod
    def make_current_solution_best_solution(self):
        pass

    @abc.abstractmethod
    def recover_to_best_found(self):
        pass
