import functools
from time import time

from base_model import BaseModelBuilder
from callbacks import GurobiAloneProgressTracker, InitialOptimizationTracker
from configuration import Configuration
from derived_modeling_data import DerivedModelingData
from fixing_data import FixingData
from solution_reminder import SolutionReminderBranching, SolutionReminderDiving
from utilities import var_values


class ReducedModelInitializer:

    def __init__(self, config: Configuration, derived: DerivedModelingData):
        self.config = config
        self.derived = derived
        self.variables, self.lin_expressions, self.initial_constraints, self.model = (
            BaseModelBuilder(config=self.config, derived=self.derived).get_base_model()
        )
        self.start_time = time()
        self.solution_summaries: list[dict[str, int | float | str]] = []

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def optimize_initially(self, patience: int | float):
        callback = InitialOptimizationTracker(patience, self.solution_summaries, self.start_time)
        self.model.optimize(callback)
        if (obj := int(self.model.ObjVal + 1e-6)) > callback.best_obj:
            summary: dict[str, int | float | str] = {
                "objective": obj,
                "runtime": time() - self.start_time,
                "station": "initial optimization",
            }
            self.solution_summaries.append(summary)

    @functools.cached_property
    def solution_data(self) -> SolutionReminderDiving:
        return SolutionReminderDiving(
            variable_values=var_values(self.model.getVars()),
            objective_value=int(self.model.ObjVal + 1e-6),
            assign_students_var_values=var_values(self.variables.assign_students.values()),
            mutual_unrealized_var_values=var_values(self.variables.mutual_unrealized.values()),
            unassigned_students_var_values=var_values(self.variables.unassigned_students.values()),
        )

    @functools.cached_property
    def fixing_data(self) -> FixingData:
        # self.solution_data ORDER MATTERS hier. Verbessern!
        return FixingData.get(self.config, self.derived, self.variables)


class ConstrainedModelInitializer:

    def __init__(self, config: Configuration, derived: DerivedModelingData):
        self.config = config
        self.derived = derived
        self.variables, self.lin_expressions, self.initial_constraints, self.model = (
            BaseModelBuilder(config=self.config, derived=self.derived).get_base_model()
        )
        self.start_time = time()
        self.solution_summaries: list[dict[str, int | float | str]] = []

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def optimize_initially(self, patience: int | float):
        callback = InitialOptimizationTracker(patience, self.solution_summaries, self.start_time)
        self.model.optimize(callback)
        if (obj := int(self.model.ObjVal + 1e-6)) > callback.best_obj:
            summary: dict[str, int | float | str] = {
                "objective": obj,
                "runtime": time() - self.start_time,
                "station": "initial optimization",
            }
            self.solution_summaries.append(summary)

    @functools.cached_property
    def current_solution(self):
        return SolutionReminderBranching(
            variable_values=var_values(self.model.getVars()),
            objective_value=int(self.model.ObjVal + 1e-6),
            assign_students_var_values=var_values(list(self.variables.assign_students.values())),
            establish_groups_var_values=var_values(list(self.variables.establish_groups.values())),
        )


class GurobiDuck:

    def __init__(self, config: Configuration, derived: DerivedModelingData):
        self.config = config
        self.derived = derived
        self.variables, self.lin_expressions, self.initial_constraints, self.model = (
            BaseModelBuilder(config=self.config, derived=self.derived).get_base_model()
        )
        self.solution_summaries: list[dict[str, int | float]] = []

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def optimize(self):
        self.model.optimize(GurobiAloneProgressTracker(self.solution_summaries))
        summary: dict[str, int | float] = {
            "objective": int(self.model.ObjVal + 1e-6),
            "bound": int(self.model.ObjBound + 1e-6),
            "runtime": self.model.Runtime,
        }
        self.solution_summaries.append(summary)
