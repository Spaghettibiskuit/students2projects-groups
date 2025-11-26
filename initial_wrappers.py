import functools

from base_model import BaseModelBuilder
from configuration import Configuration
from derived_modeling_data import DerivedModelingData
from fixing_data import FixingData
from patience_callback import PatienceOutsideLocalSearch
from solution_reminder import SolutionReminderDiving
from utilities import var_values


class ReducedModelInitializer:

    def __init__(self, config: Configuration, derived: DerivedModelingData):
        self.config = config
        self.derived = derived
        self.variables, self.lin_expressions, self.initial_constraints, self.model = (
            BaseModelBuilder(config=self.config, derived=self.derived).get_base_model()
        )

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def optimize_initially(self, patience: int | float):
        self.model.optimize(PatienceOutsideLocalSearch(patience))

    @functools.cached_property
    def solution_data(self) -> SolutionReminderDiving:
        return SolutionReminderDiving(
            variable_values=var_values(self.model.getVars()),
            objective_value=self.model.ObjVal,
            assign_students_var_values=var_values(self.variables.assign_students.values()),
            mutual_unrealized_var_values=var_values(self.variables.mutual_unrealized.values()),
            unassigned_students_var_values=var_values(self.variables.unassigned_students.values()),
        )

    @functools.cached_property
    def fixing_data(self) -> FixingData:
        return FixingData.get(self.config, self.derived, self.variables)
