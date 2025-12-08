"""A class that contains a model further constrained for local branching."""

import itertools as it

import gurobipy
from gurobipy import GRB

from model_components import ModelComponents
from model_wrapper import ModelWrapper
from solution_reminder import SolutionReminderBranching
from thin_wrappers import ConstrainedModelInitializer
from utilities import var_values


class ConstrainedModel(ModelWrapper):
    """Contains a model further constrained for local branching."""

    def __init__(
        self,
        model_components: ModelComponents,
        model: gurobipy.Model,
        start_time: float,
        solution_summaries: list[dict[str, int | float | str]],
        sol_reminder: SolutionReminderBranching,
    ):
        super().__init__(
            model_components=model_components,
            model=model,
            start_time=start_time,
            solution_summaries=solution_summaries,
            sol_reminder=sol_reminder,
        )
        variables = self.model_components.variables
        self.assign_students_vars = list(variables.assign_students.values())
        self.assign_students_vars_values: tuple[int | float, ...] = (
            sol_reminder.assign_students_var_values
        )
        self.establish_groups_vars = list(variables.establish_groups.values())
        self.establish_groups_vars_values: tuple[int | float, ...] = (
            sol_reminder.assign_students_var_values
        )
        self.branching_constraints: list[gurobipy.Constr] = []
        self.counter = it.count()
        self.shake_constraints: tuple[gurobipy.Constr, gurobipy.Constr] | None = None
        self.current_solution: SolutionReminderBranching
        self.best_found_solution: SolutionReminderBranching

    def store_solution(self):
        self.current_solution = SolutionReminderBranching(
            variable_values=var_values(self.model.getVars()),
            objective_value=self.objective_value,
            assign_students_var_values=var_values(self.assign_students_vars),
            establish_groups_var_values=var_values(self.establish_groups_vars),
        )

    def new_best_found(self) -> bool:
        return self.current_solution.objective_value > self.best_found_solution.objective_value

    def make_current_solution_best_solution(self):
        self.best_found_solution = self.current_solution

    def set_branching_var_values_for_vnd(self):
        self.assign_students_vars_values = self.current_solution.assign_students_var_values
        self.establish_groups_vars_values = self.current_solution.establish_groups_var_values

    def set_branching_var_values_for_shake(self):
        self.assign_students_vars_values = self.best_found_solution.assign_students_var_values
        self.establish_groups_vars_values = self.best_found_solution.establish_groups_var_values

    def branching_lin_expression(self):
        return gurobipy.quicksum(
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

    def pop_branching_constraints_stack(self):
        branching_constraint = self.branching_constraints.pop()
        self.model.remove(branching_constraint)

    def add_excluding_branching_constraint(self, rhs: int):
        branching_constr_name = f"branching{next(self.counter)}"
        branching_constr = self.model.addConstr(
            self.branching_lin_expression() >= rhs + 1,
            name=branching_constr_name,
        )
        self.branching_constraints.append(branching_constr)

    def drop_all_branching_constraints(self):
        self.model.remove(self.branching_constraints)
        self.branching_constraints.clear()

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

    def recover_to_best_found(self):
        for var, var_value_incumbent in zip(
            self.model.getVars(), self.best_found_solution.variable_values
        ):
            var.Start = var_value_incumbent

        self.drop_all_branching_constraints()
        self.model.Params.Cutoff = round(self.best_found_solution.objective_value) - 0.5
        self.model.Params.SolutionLimit = 1
        self.model.Params.TimeLimit = float("inf")
        self.model.optimize()
        self.model.Params.SolutionLimit = GRB.MAXINT

    @classmethod
    def get(
        cls,
        initializer: ConstrainedModelInitializer,
    ):
        return cls(
            model_components=initializer.model_components,
            model=initializer.model,
            sol_reminder=initializer.current_solution,
            start_time=initializer.start_time,
            solution_summaries=initializer.solution_summaries,
        )
