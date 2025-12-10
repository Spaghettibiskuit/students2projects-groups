"""A class that contains a model further constrained for local branching."""

import itertools

import gurobipy
from gurobipy import GRB

from model_components import ModelComponents
from model_wrapper import ModelWrapper
from solution_reminders import SolutionReminderBranching
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
        self.branching_constraints: list[gurobipy.Constr] = []
        self.counter = itertools.count()
        self.shake_constraints: tuple[gurobipy.Constr, gurobipy.Constr] | None = None
        self.current_solution: SolutionReminderBranching
        self.best_found_solution: SolutionReminderBranching

    def store_solution(self):
        self.current_solution = SolutionReminderBranching(
            variable_values=var_values(self.model.getVars()),
            objective_value=self.objective_value,
            assign_students_var_values=var_values(self.assign_students_vars),
        )

    def make_current_solution_best_solution(self):
        self.best_found_solution = self.current_solution

    def branching_lin_expression(self, shake: bool = False):
        relevant_solution = self.best_found_solution if shake else self.current_solution
        return gurobipy.quicksum(
            1 - var if var_value > 0.5 else var
            for var, var_value in zip(
                self.assign_students_vars,
                relevant_solution.assign_students_var_values,
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
        lin_expr = self.branching_lin_expression(shake=True)
        smaller_radius = self.model.addConstr(
            lin_expr >= k_cur,
        )
        bigger_radius = self.model.addConstr(
            lin_expr <= k_cur + k_step,
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
        self.model.Params.Cutoff = round(self.best_found_solution.objective_value) - 1e-6
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
