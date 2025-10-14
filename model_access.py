"""Class that provides access to information about the model state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import gurobipy as gp

from names import VariableNames

if TYPE_CHECKING:
    from configuration import Configuration
    from constrained_model import ConstrainedModel
    from derived_modeling_data import DerivedModelingData
    from model_components import LinExpressions, Variables


@dataclass(frozen=True)
class ModelAccess:
    variables: Variables
    lin_expressions: LinExpressions

    @classmethod
    def get(
        cls,
        config: Configuration,
        derived: DerivedModelingData,
        constrained_model: ConstrainedModel,
    ):
        gurobi_model = constrained_model.model
        assign_students = gp.tupledict(
            zip(derived.project_group_student_triples, constrained_model.assign_students_vars)
        )
        establish_groups = gp.tupledict(
            zip(derived.project_group_pairs, constrained_model.establish_groups_vars)
        )

        mutual_unrealized_name = VariableNames.MUTUAL_UNREALIZED.value
        mutual_unrealized_vars = tuple(
            cast(
                gp.Var,
                gurobi_model.getVarByName(f"{mutual_unrealized_name}[{first_id},{second_id}]"),
            )
            for first_id, second_id in derived.mutual_pairs
        )


#         assign_students_names = VariableNames.ASSIGN_STUDENTS.value
# assign_students_var_names = tuple(
#     f"{assign_students_names}[{project_id},{group_id},{student_id}]"
#     for project_id, group_id, student_id in derived.project_group_student_triples
# )
# assign_students_vars = tuple(
#     cast(gp.Var, base_model.getVarByName(assign_student_var_name))
#     for assign_student_var_name in assign_students_var_names
# )
