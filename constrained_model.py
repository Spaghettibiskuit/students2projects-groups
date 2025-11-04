"""A class that contains a model further constrained for local branching."""

from __future__ import annotations

import copy
import itertools as it
from typing import TYPE_CHECKING

import gurobipy as gp

from base_model import BaseModelBuilder
from solution_reminder import SolutionReminder

if TYPE_CHECKING:
    from configuration import Configuration
    from derived_modeling_data import DerivedModelingData


class ConstrainedModel:
    """Contains a model further constrained for local branching."""

    def __init__(self, config: Configuration, derived: DerivedModelingData):
        self.variables, self.lin_expressions, self.initial_constraints, self.model = (
            BaseModelBuilder(config=config, derived=derived).get_base_model()
        )
        self.assign_students_vars = list(self.variables.assign_students.values())
        self.assign_students_vars_values: tuple[int | float, ...] | None = None
        self.establish_groups_vars = list(self.variables.establish_groups.values())
        self.establish_groups_vars_values: tuple[int | float, ...] | None = None
        self.branching_constraints: list[gp.Constr] = []
        self.counter = it.count()
        self.shake_constraints: tuple[gp.Constr, gp.Constr] | None = None
        self.last_feasible_solution: SolutionReminder | None = None
        self.incumbent_solution: SolutionReminder | None = None

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

    def eliminate_solution_limit(self) -> None:
        self.model.Params.SolutionLimit = 2_000_000_000

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def set_cutoff(self, ascending: bool = True):
        if self.last_feasible_solution is None:
            raise TypeError("last_feasible_solution should not be None at this point.")
        self.model.Params.Cutoff = (
            round(self.last_feasible_solution.objective_value) - 0.5 + int(ascending)
        )

    def eliminate_cutoff(self):
        self.model.Params.Cutoff = float("-inf")

    def update(self):
        self.model.update()

    def optimize(self):
        self.model.optimize()

    def store_solution(self):
        self.last_feasible_solution = SolutionReminder(
            variable_values=tuple(var.X for var in self.model.getVars()),
            objective_value=self.objective_value,
            assign_students_vars_values=tuple(var.X for var in self.assign_students_vars),
            establish_groups_vars_values=tuple(var.X for var in self.establish_groups_vars),
        )

    def last_feasible_solution_better_than_incumbent(self) -> bool:
        if self.last_feasible_solution is None or self.incumbent_solution is None:
            raise TypeError(
                "last_feasible_solution and incumbent_solution should not be None at this point."
            )
        return (
            self.last_feasible_solution.objective_value > self.incumbent_solution.objective_value
        )

    def store_last_feasible_solution_as_incumbent(self):
        self.incumbent_solution = copy.copy(self.last_feasible_solution)

    def set_branching_var_values_for_inside_vnd(self):
        if self.last_feasible_solution is None:
            raise TypeError("last_feasible_solution should not be None at this point.")
        last_feasible_solution = self.last_feasible_solution
        self.assign_students_vars_values = last_feasible_solution.assign_students_vars_values
        self.establish_groups_vars_values = last_feasible_solution.establish_groups_vars_values

    def set_branching_var_values_for_shake(self):
        if self.incumbent_solution is None:
            raise TypeError("incumbent_solution should not be None at this point.")

        incumbent_solution = self.incumbent_solution
        self.assign_students_vars_values = incumbent_solution.assign_students_vars_values
        self.establish_groups_vars_values = incumbent_solution.establish_groups_vars_values

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

    def drop_latest_branching_constraint(self):
        branching_constraint = self.branching_constraints.pop()
        self.model.remove(branching_constraint)

    def add_excluding_branching_constraint(self, rhs: int):
        branching_constr_name = f"branching{next(self.counter)}"
        branching_constr = self.model.addConstr(
            self.branching_lin_expression() >= rhs + 1,
            name=branching_constr_name,
        )
        self.branching_constraints.append(branching_constr)

    def prohibit_last_solution(self):
        branching_constr_name = f"branching{next(self.counter)}"
        branching_constr = self.model.addConstr(
            self.branching_lin_expression() >= 1,
            name=branching_constr_name,
        )
        self.branching_constraints.append(branching_constr)

    def drop_all_branching_constraints(self):
        self.model.remove(self.branching_constraints)

    def add_shaking_constraints(self, k_cur: int, k_step: int):

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

    def recover_to_best_solution_at_end(self):
        if self.incumbent_solution is None:
            raise TypeError("incumbent_solution should not be None at this point.")

        for var, var_value_incumbent in zip(
            self.model.getVars(), self.incumbent_solution.variable_values
        ):
            var.Start = var_value_incumbent

        self.drop_all_branching_constraints()
        self.model.Params.Cutoff = round(self.incumbent_solution.objective_value) - 0.5
        self.model.Params.SolutionLimit = 1
        self.model.Params.TimeLimit = float("inf")
        self.model.optimize()
