import abc

import gurobipy

from model_components import ModelComponents


class ModelWrapper(abc.ABC):

    def __init__(self, model_components: ModelComponents, model: gurobipy.Model):
        self.model_components = model_components
        self.model = model

    @property
    def status(self) -> int:
        return self.model.Status

    @property
    def objective_value(self) -> int:
        return int(self.model.ObjVal + 1e-6)

    @property
    def solution_count(self) -> int:
        return self.model.SolCount

    def eliminate_time_limit(self):
        self.model.Params.TimeLimit = float("inf")

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def eliminate_cutoff(self):
        self.model.Params.Cutoff = float("-inf")

    @abc.abstractmethod
    def set_cutoff(self):
        pass

    @abc.abstractmethod
    def store_solution(self):
        pass

    @abc.abstractmethod
    def make_current_solution_best_solution(self):
        pass

    @abc.abstractmethod
    def new_best_found(self) -> bool:
        pass

    @abc.abstractmethod
    def recover_to_best_found(self):
        pass
