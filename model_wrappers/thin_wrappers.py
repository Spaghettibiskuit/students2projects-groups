import functools
from time import time

from modeling.base_model_builder import BaseModelBuilder
from modeling.configuration import Configuration
from modeling.derived_modeling_data import DerivedModelingData
from solving_utilities.callbacks import (
    GurobiAloneProgressTracker,
    InitialOptimizationTracker,
)
from solving_utilities.fixing_data import FixingByRankingData
from solving_utilities.solution_reminders import (
    SolutionReminderBranching,
    SolutionReminderDiving,
)
from utilities import Stations, gurobi_round, var_values


class ThinWrapper:

    def __init__(self, config: Configuration, derived: DerivedModelingData):
        self.config = config
        self.derived = derived
        self.model_components, self.model = BaseModelBuilder(
            config=self.config, derived=self.derived
        ).get_base_model()
        self.start_time = time()
        self.solution_summaries: list[dict[str, int | float | str]] = []

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def optimize(self, patience: int | float):
        callback = InitialOptimizationTracker(patience, self.solution_summaries, self.start_time)
        self.model.optimize(callback)
        if (obj := gurobi_round(self.model.ObjVal)) > callback.best_obj:
            summary: dict[str, int | float | str] = {
                "objective": obj,
                "runtime": time() - self.start_time,
                "station": Stations.INITIAL_OPTIMIZATION,
            }
            self.solution_summaries.append(summary)


class ReducedModelInitializer(ThinWrapper):

    @functools.cached_property
    def current_solution(self) -> SolutionReminderDiving:
        variables = self.model_components.variables
        return SolutionReminderDiving(
            variable_values=var_values(self.model.getVars()),
            objective_value=gurobi_round(self.model.ObjVal),
            assign_students_var_values=var_values(variables.assign_students.values()),
            mutual_unrealized_var_values=var_values(variables.mutual_unrealized.values()),
            unassigned_students_var_values=var_values(variables.unassigned_students.values()),
        )

    @functools.cached_property
    def fixing_data(self) -> FixingByRankingData:
        return FixingByRankingData.get(
            config=self.config,
            derived=self.derived,
            variables=self.model_components.variables,
            lin_expressions=self.model_components.lin_expressions,
            model=self.model,
        )


class ConstrainedModelInitializer(ThinWrapper):

    @functools.cached_property
    def current_solution(self):
        variables = self.model_components.variables
        return SolutionReminderBranching(
            variable_values=var_values(self.model.getVars()),
            objective_value=gurobi_round(self.model.ObjVal),
            assign_students_var_values=var_values(list(variables.assign_students.values())),
        )


class GurobiDuck:

    def __init__(self, config: Configuration, derived: DerivedModelingData):
        self.config = config
        self.derived = derived
        self.model_components, self.model = BaseModelBuilder(
            config=self.config, derived=self.derived
        ).get_base_model()
        self.solution_summaries: list[dict[str, int | float]] = []

    @property
    def objective_value(self) -> int:
        return gurobi_round(self.model.ObjVal)

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def optimize(self):
        self.model.optimize(GurobiAloneProgressTracker(self.solution_summaries))
        summary: dict[str, int | float] = {
            "objective": gurobi_round(self.model.ObjVal),
            "bound": gurobi_round(self.model.ObjBound),
            "runtime": self.model.Runtime,
        }
        self.solution_summaries.append(summary)
