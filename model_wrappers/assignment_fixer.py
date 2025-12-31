"""A class that contains a model which it reduces according to VNS rules."""

import abc
import functools
import random

import gurobipy

from model_wrappers.model_wrapper import ModelWrapper
from modeling.configuration import Configuration
from modeling.derived_modeling_data import DerivedModelingData
from modeling.model_components import ModelComponents
from solving_utilities.assignment_fixing_data import AssignmentFixingData
from solving_utilities.solution_reminders import SolutionReminderAssignmentFixing
from solving_utilities.variable_access import GurobiVariableAccess
from utilities import var_values


class AssignmentFixer(ModelWrapper):
    """Contains a model further constrained for local branching."""

    def __init__(
        self,
        model_components: ModelComponents,
        model: gurobipy.Model,
        start_time: float,
        solution_summaries: list[dict[str, int | float | str]],
        config: Configuration,
        derived: DerivedModelingData,
        sol_reminder: SolutionReminderAssignmentFixing,
        fixing_data: AssignmentFixingData,
    ):
        super().__init__(model_components, model, start_time, solution_summaries, sol_reminder)
        self.config = config
        self.derived = derived
        self.current_sol_fixing_data = fixing_data
        self.best_sol_fixing_data = fixing_data
        self.variable_access = GurobiVariableAccess.get(self.model_components.variables)
        self.current_solution: SolutionReminderAssignmentFixing
        self.best_found_solution: SolutionReminderAssignmentFixing

    def store_solution(self):
        self.current_solution = SolutionReminderAssignmentFixing(
            variable_values=var_values(self.model.getVars()),
            objective_value=self.objective_value,
            assign_students_var_values=var_values(self.variable_access.assign_students),
            group_size_surplus_var_values=var_values(self.variable_access.group_size_surplus),
            group_size_deficit_var_values=var_values(self.variable_access.group_size_deficit),
        )
        self.current_sol_fixing_data = AssignmentFixingData.get(
            config=self.config,
            derived=self.derived,
            variables=self.model_components.variables,
            lin_expressions=self.model_components.lin_expressions,
            model=self.model,
        )

    def make_current_solution_best_solution(self):
        self.best_found_solution = self.current_solution
        self.best_sol_fixing_data = self.current_sol_fixing_data

    def make_best_solution_current_solution(self):
        self.current_solution = self.best_found_solution
        self.current_sol_fixing_data = self.best_sol_fixing_data

    @functools.lru_cache(maxsize=128)
    def zones(self, num_zones: int) -> list[tuple[int, int]]:
        num_students = self.config.number_of_students
        floor_size = num_students // num_zones
        ceil_size = floor_size + 1
        num_ceil = num_students - (floor_size * num_zones)
        num_floor = num_zones - num_ceil
        sizes = random.sample([floor_size, ceil_size], counts=[num_floor, num_ceil], k=num_zones)
        boundaries: list[tuple[int, int]] = []
        current_idx = 0
        for size in sizes:
            boundaries.append((current_idx, current_idx + size))
            current_idx += size
        if current_idx != num_students:
            raise ValueError("Sizes do not add up to number of students.")
        return boundaries

    def _separate_assignments(
        self, zone_a: int, zone_b: int, num_zones: int, assignments: list[tuple[int, int, int]]
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:

        zones = self.zones(num_zones)
        start_a, end_a = zones[zone_a]
        start_b, end_b = zones[zone_b]
        return (
            assignments[start_a:end_a] + assignments[start_b:end_b],
            assignments[:start_a] + assignments[end_a:start_b] + assignments[end_b:],
        )

    @abc.abstractmethod
    def fix_rest(self, zone_a: int, zone_b: int, num_zones: int):
        pass

    def increment_random_seed(self):
        self.model.Params.Seed += 1

    def delete_zoning_rules(self):
        self.zones.cache_clear()

    def force_k_worst_to_change(self, k: int):

        self.model.setAttr(
            "UB",
            self.variable_access.assign_students,
            [1] * len(self.variable_access.assign_students),
        )

        worst_k = self.current_sol_fixing_data.line_up_assignments[:k]
        variables = self.model_components.variables

        for project_id, group_id, student_id in worst_k:
            if project_id == -1:  # It is a pseudo_assignment
                var = variables.unassigned_students[student_id]
            else:
                var = variables.assign_students[project_id, group_id, student_id]
            var.UB = 0

        undefined_val = gurobipy.GRB.UNDEFINED
        worst_k_student_ids = set(student_id for _, _, student_id in worst_k)
        start_values = [
            undefined_val if student_id in worst_k_student_ids else value
            for (_, _, student_id), value in zip(
                self.derived.project_group_student_triples,
                self.current_solution.assign_students_var_values,
            )
        ]

        self.model.setAttr("Start", self.variable_access.assign_students, start_values)

    def free_all_unassigned_vars(self):
        variables = list(self.model_components.variables.unassigned_students.values())
        self.model.setAttr("UB", variables, [1] * len(variables))
