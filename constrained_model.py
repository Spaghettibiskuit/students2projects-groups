"""A class that contains a model further constrained for local branching."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import gurobipy as gp

from base_model import get_base_model
from names import VariableNames

if TYPE_CHECKING:
    from configuration import Configuration
    from derived_modeling_data import DerivedModelingData


class ConstrainedModel:
    """Contains a model further constrained for local branching."""

    def __init__(
        self,
        model: gp.Model,
        assign_students_var_names: tuple[str, ...],
        assign_students_vars: tuple[gp.Var, ...],
        establish_groups_var_names: tuple[str, ...],
        establish_groups_vars: tuple[gp.Var, ...],
        branching_constraints_names: list[str],
        branching_constraints: list[gp.Constr],
        lower_bound_constr_name: str | None,
        lower_bound_constr: gp.Constr | None,
        shake_constraints: tuple[gp.Constr, gp.Constr] | None,
    ):
        self.model = model
        self.assign_students_var_names = assign_students_var_names
        self.assign_students_vars = assign_students_vars
        self.establish_groups_var_names = establish_groups_var_names
        self.establish_groups_vars = establish_groups_vars
        self.branching_constraints_names = branching_constraints_names
        self.branching_constraints = branching_constraints
        self.lower_bound_constr_name = lower_bound_constr_name
        self.lower_bound_constr = lower_bound_constr
        self.shake_constraints = shake_constraints

    def copy(self):
        model = self.model.copy()

        assign_students_vars = tuple(
            cast(gp.Var, model.getVarByName(assign_student_var_name))
            for assign_student_var_name in self.assign_students_var_names
        )

        establish_groups_vars = tuple(
            cast(gp.Var, model.getVarByName(establish_group_var_name))
            for establish_group_var_name in self.establish_groups_var_names
        )

        branching_constraints = [
            cast(gp.Constr, model.getConstrByName(branching_constraint_name))
            for branching_constraint_name in self.branching_constraints_names
        ]
        if self.lower_bound_constr_name:
            lower_bound_constraint = cast(
                gp.Constr, model.getConstrByName(self.lower_bound_constr_name)
            )
        else:
            lower_bound_constraint = None

        return ConstrainedModel(
            model=model,
            assign_students_var_names=self.assign_students_var_names,
            assign_students_vars=assign_students_vars,
            establish_groups_var_names=self.establish_groups_var_names,
            establish_groups_vars=establish_groups_vars,
            branching_constraints_names=self.branching_constraints_names,
            branching_constraints=branching_constraints,
            lower_bound_constr_name=self.lower_bound_constr_name,
            lower_bound_constr=lower_bound_constraint,
            shake_constraints=None,
        )

    @classmethod
    def initial_model(
        cls,
        config: Configuration,
        derived: DerivedModelingData,
    ):
        base_model = get_base_model(config, derived)

        assign_students_name = VariableNames.ASSIGN_STUDENTS.value
        assign_students_var_names = tuple(
            f"{assign_students_name}[{project_id},{group_id},{student_id}]"
            for project_id, group_id, student_id in derived.project_group_student_triples
        )
        assign_students_vars = tuple(
            cast(gp.Var, base_model.getVarByName(assign_student_var_name))
            for assign_student_var_name in assign_students_var_names
        )

        establish_groups_name = VariableNames.ESTABLISH_GROUPS.value
        establish_groups_var_names = tuple(
            f"{establish_groups_name}[{project_id},{group_id}]"
            for project_id, group_id in derived.project_group_pairs
        )
        establish_groups_vars = tuple(
            cast(gp.Var, base_model.getVarByName(establish_group_var_name))
            for establish_group_var_name in establish_groups_var_names
        )
        return cls(
            model=base_model,
            assign_students_var_names=assign_students_var_names,
            assign_students_vars=assign_students_vars,
            establish_groups_var_names=establish_groups_var_names,
            establish_groups_vars=establish_groups_vars,
            branching_constraints_names=[],
            branching_constraints=[],
            lower_bound_constr_name=None,
            lower_bound_constr=None,
            shake_constraints=None,
        )
