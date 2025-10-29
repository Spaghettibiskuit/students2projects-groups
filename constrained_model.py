"""A class that contains a model further constrained for local branching."""

from __future__ import annotations

import copy
import itertools as it
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
        assign_students_vars_values: tuple[int | float, ...],
        establish_groups_var_names: tuple[str, ...],
        establish_groups_vars: tuple[gp.Var, ...],
        establish_groups_vars_values: tuple[int | float, ...],
        branching_constraints_names: list[str],
        branching_constraints: list[gp.Constr],
        counter: it.count[int],
        saved_var_values: tuple[float, ...],
    ):
        self.model = model
        self.assign_students_var_names = assign_students_var_names
        self.assign_students_vars = assign_students_vars
        self.assign_students_vars_values = assign_students_vars_values
        self.establish_groups_var_names = establish_groups_var_names
        self.establish_groups_vars = establish_groups_vars
        self.establish_groups_vars_values = establish_groups_vars_values
        self.branching_constraints_names = branching_constraints_names
        self.branching_constraints = branching_constraints
        self.counter = counter
        self.shake_constraints: tuple[gp.Constr, gp.Constr] | None = None
        self.saved_var_values = saved_var_values

    @property
    def status(self) -> int:
        return self.model.Status

    @property
    def objective_value(self) -> int | float:
        return self.model.ObjVal

    @property
    def solution_count(self) -> int:
        return self.model.SolCount

    def set_solution_limit(self, solution_limit: int):
        self.model.Params.SolutionLimit = solution_limit

    def eliminate_solution_limit(self):
        self.model.Params.SolutionLimit = 2_000_000_000

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def set_cutoff(self, ascending: bool = True):
        self.model.Params.Cutoff = round(self.objective_value) - 0.5 + int(ascending)

    # def recover(self):
    #     self.drop_latest_branching_constraint()
    #     if not self.saved_var_values:
    #         raise ValueError("No variable values available.")

    #     for var, var_value in zip(self.model.getVars(), self.saved_var_values):
    #         var.Start = var_value

    #     self.model.Params.SolutionLimit = 1
    #     self.model.optimize()
    #     self.model.Params.SolutionLimit = 2_000_000_000

    def eliminate_cutoff(self):
        self.model.Params.Cutoff = float("-inf")

    def update(self):
        self.model.update()

    def optimize(self):
        self.model.optimize()

    def save_var_values(self):
        self.saved_var_values = tuple(var.X for var in self.model.getVars())

    def save_decision_var_values(self):
        self.assign_students_vars_values = tuple(var.X for var in self.assign_students_vars)
        self.establish_groups_vars_values = tuple(var.X for var in self.establish_groups_vars)

    def branching_lin_expression(self):
        if not self.establish_groups_vars_values or not self.assign_students_vars_values:
            raise ValueError("Values for decision variables not saved.")
        return gp.quicksum(
            1 - var if var_value > 0.5 else var
            for var, var_value in zip(
                it.chain(self.assign_students_vars, self.establish_groups_vars),
                it.chain(self.assign_students_vars_values, self.establish_groups_vars_values),
            )
        )

    def add_bounding_branching_constraint(self, rhs: int):
        branching_constr_name = f"branching{next(self.counter)}"
        branching_constr = self.model.addConstr(
            self.branching_lin_expression() <= rhs,
            name=branching_constr_name,
        )
        self.branching_constraints.append(branching_constr)
        self.branching_constraints_names.append(branching_constr_name)

    def drop_latest_branching_constraint(self):
        branching_constraint = self.branching_constraints.pop()
        self.model.remove(branching_constraint)
        self.branching_constraints_names.pop()

    def add_excluding_branching_constraint(self, rhs: int):
        branching_constr_name = f"branching{next(self.counter)}"
        branching_constr = self.model.addConstr(
            self.branching_lin_expression() >= rhs + 1,
            name=branching_constr_name,
        )
        self.branching_constraints.append(branching_constr)
        self.branching_constraints_names.append(branching_constr_name)

    def prohibit_last_solution(self):
        branching_constr_name = f"branching{next(self.counter)}"
        branching_constr = self.model.addConstr(
            self.branching_lin_expression() >= 1,
            name=branching_constr_name,
        )
        self.branching_constraints.append(branching_constr)
        self.branching_constraints_names.append(branching_constr_name)

    def drop_all_branching_constraints(self):
        self.model.remove(self.branching_constraints)
        self.branching_constraints_names.clear()

    def add_shaking_constraints(self, k_cur: int, k_step: int):
        if not self.saved_var_values:
            raise ValueError("No variable values available.")

        smaller_radius = self.model.addConstr(
            self.branching_lin_expression() >= k_cur,
        )

        bigger_radius = self.model.addConstr(
            self.branching_lin_expression() <= k_cur + k_step,
        )

        self.shake_constraints = smaller_radius, bigger_radius

    def remove_shaking_constraints(self):
        if self.shake_constraints is None:
            raise TypeError("Cannot remove shake constraints if None.")
        self.model.remove(self.shake_constraints)
        self.shake_constraints = None

    def copy(self):
        model = self.model.copy()

        if not self.saved_var_values:
            raise ValueError("No variable values available.")

        for copied_var, var_value in zip(model.getVars(), self.saved_var_values):
            copied_var.Start = var_value

        model.Params.TimeLimit = float("inf")
        model.Params.Cutoff = round(self.objective_value) - 0.5
        model.Params.SolutionLimit = 1
        model.optimize()
        model.Params.SolutionLimit = 2_000_000_000

        assign_students_vars = tuple(
            cast(gp.Var, model.getVarByName(assign_student_var_name))
            for assign_student_var_name in self.assign_students_var_names
        )

        establish_groups_vars = tuple(
            cast(gp.Var, model.getVarByName(establish_group_var_name))
            for establish_group_var_name in self.establish_groups_var_names
        )

        branching_constraints_names = copy.copy(self.branching_constraints_names)

        branching_constraints = [
            cast(gp.Constr, model.getConstrByName(branching_constraint_name))
            for branching_constraint_name in branching_constraints_names
        ]

        return ConstrainedModel(
            model=model,
            assign_students_var_names=self.assign_students_var_names,
            assign_students_vars=assign_students_vars,
            assign_students_vars_values=self.assign_students_vars_values,
            establish_groups_var_names=self.establish_groups_var_names,
            establish_groups_vars=establish_groups_vars,
            establish_groups_vars_values=self.establish_groups_vars_values,
            branching_constraints_names=branching_constraints_names,
            branching_constraints=branching_constraints,
            counter=self.counter,
            saved_var_values=self.saved_var_values,
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
        counter = it.count()
        return cls(
            model=base_model,
            assign_students_var_names=assign_students_var_names,
            assign_students_vars=assign_students_vars,
            assign_students_vars_values=tuple(),
            establish_groups_var_names=establish_groups_var_names,
            establish_groups_vars=establish_groups_vars,
            establish_groups_vars_values=tuple(),
            branching_constraints_names=[],
            branching_constraints=[],
            counter=counter,
            saved_var_values=tuple(),
        )
