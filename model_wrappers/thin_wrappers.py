import functools
from time import time

from modeling.base_model_builder import BaseModelBuilder
from modeling.configuration import Configuration
from modeling.derived_modeling_data import DerivedModelingData
from solving_utilities.assignment_fixing_data import AssignmentFixingData
from solving_utilities.callbacks import (
    GurobiAloneProgressTracker,
    InitialOptimizationTracker,
)
from solving_utilities.solution_reminders import (
    SolutionReminderAssignmentFixing,
    SolutionReminderBranching,
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

    def set_time_limit(self, total_time_limit: int | float, start_time: float):
        self.model.Params.TimeLimit = max(0, total_time_limit - (time() - start_time))

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


class AssignmentFixerInitializer(ThinWrapper):
    @functools.cached_property
    def fixing_data(self) -> AssignmentFixingData:
        return AssignmentFixingData.get(
            config=self.config,
            derived=self.derived,
            variables=self.model_components.variables,
            lin_expressions=self.model_components.lin_expressions,
            model=self.model,
        )

    @functools.cached_property
    def current_solution(self) -> SolutionReminderAssignmentFixing:
        variables = self.model_components.variables
        return SolutionReminderAssignmentFixing(
            variable_values=var_values(self.model.getVars()),
            objective_value=gurobi_round(self.model.ObjVal),
            assign_students_var_values=var_values(variables.assign_students.values()),
            group_size_surplus_var_values=var_values(variables.group_size_surplus.values()),
            group_size_deficit_var_values=var_values(variables.group_size_deficit.values()),
        )


class LocalBrancherInitializer(ThinWrapper):
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
