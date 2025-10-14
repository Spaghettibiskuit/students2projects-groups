"""A class that retrieves information regarding the solution within a Gurobi model."""

from functools import cached_property

from configuration import Configuration
from constrained_model import ConstrainedModel
from derived_modeling_data import DerivedModelingData
from model_components import LinExpressions, Variables


class SolutionInformationRetriever:
    """Retrieves information regarding the solution within a Gurobi model."""

    def __init__(
        self,
        config: Configuration,
        derived: DerivedModelingData,
        constrained_model: ConstrainedModel,
        variables: Variables,
        lin_expressions: LinExpressions,
    ):
        self.config = config
        self.derived = DerivedModelingData
        self.model = constrained_model.model
        self.variables = variables
        self.lin_expressions = lin_expressions

    @cached_property
    def objective_value(self):
        return self.model.ObjVal

    @cached_property
    def num_unassigned(self) -> int:
        return round(
            self.lin_expressions.sum_penalties_unassigned.getValue()
            / self.config.penalty_unassigned
        )

    @cached_property
    def unassigned_students(self) -> list[tuple[int, str]]:
        return [
            (student_id, name)
            for name, student_id, is_unassigned in zip(
                self.config.students_info["name"],
                self.derived.student_ids,
                self.variables.unassigned_students.values(),
            )
            if round(is_unassigned.X)
        ]
