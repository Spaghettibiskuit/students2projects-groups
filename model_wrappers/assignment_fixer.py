"""A class that contains a model which it reduces according to VNS rules."""

import functools
import itertools
import random

import gurobipy

from model_wrappers.model_wrapper import ModelWrapper
from model_wrappers.thin_wrappers import ReducedModelInitializer
from modeling.configuration import Configuration
from modeling.derived_modeling_data import DerivedModelingData
from modeling.model_components import ModelComponents
from solving_utilities.assignment_fixing_data import AssignmentFixingData
from solving_utilities.solution_reminders import SolutionReminderDiving
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
        sol_reminder: SolutionReminderDiving,
        fixing_data: AssignmentFixingData,
    ):
        super().__init__(model_components, model, start_time, solution_summaries, sol_reminder)
        self.config = config
        self.derived = derived
        self.current_sol_fixing_data = fixing_data
        self.best_sol_fixing_data = fixing_data
        self.current_solution: SolutionReminderDiving
        self.best_found_solution: SolutionReminderDiving
        self.assign_students_vars = list(self.model_components.variables.assign_students.values())

    def store_solution(self):
        variables = self.model_components.variables
        self.current_solution = SolutionReminderDiving(
            variable_values=var_values(self.model.getVars()),
            objective_value=self.objective_value,
            assign_students_var_values=var_values(variables.assign_students.values()),
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

    def _shifted_groups(
        self, groups_only_free: set[tuple[int, int]], groups_of_fixed: set[tuple[int, int]]
    ) -> dict[tuple[int, int], tuple[int, int]]:
        all_groups = groups_only_free.union(groups_of_fixed)
        if len(all_groups) != len(groups_only_free) + len(groups_of_fixed):
            raise ValueError()

        all_groups_ordered = sorted(all_groups)
        groups: dict[int, list[int]] = {}
        for project_id, group_id in all_groups_ordered:
            groups.setdefault(project_id, []).append(group_id)

        groups_only_free_ordered = sorted(groups_only_free)
        affected_projects = [project_id for project_id, _ in groups_only_free_ordered]
        affected_groups = {project_id: groups[project_id] for project_id in affected_projects}

        only_free_affected_groups: dict[int, list[int]] = {}
        for project_id, group_id in groups_only_free_ordered:
            only_free_affected_groups.setdefault(project_id, []).append(group_id)

        mixed_affected_groups = {
            project_id: [
                group_id
                for group_id in group_ids
                if (project_id, group_id) not in groups_only_free
            ]
            for project_id, group_ids in affected_groups.items()
        }

        return {
            (project_id, group_id): (project_id, new_group_id)
            for (project_id, mixed_affected_groups), only_free_affected_groups in zip(
                mixed_affected_groups.items(), only_free_affected_groups.values()
            )
            for group_id, new_group_id in zip(
                mixed_affected_groups + only_free_affected_groups,
                itertools.count(),
            )
        }

    def _separate_groups(
        self,
        free_assignments: list[tuple[int, int, int]],
        fixed_assignments: list[tuple[int, int, int]],
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        groups_of_free = set(
            (project_id, group_id)
            for project_id, group_id, _ in free_assignments
            if project_id != -1  # Must be real group
        )
        groups_of_fixed = set(
            (project_id, group_id)
            for project_id, group_id, _ in fixed_assignments
            if project_id != -1
        )
        return groups_of_free.difference(groups_of_fixed), groups_of_fixed

    def _adjusted_line_up_assignments(
        self, shifted_groups: dict[tuple[int, int], tuple[int, int]]
    ) -> list[tuple[int, int, int]]:
        return [
            (
                (*shifted_group, student_id)
                if (shifted_group := shifted_groups.get((proj_id, group_id))) is not None
                else (proj_id, group_id, student_id)
            )
            for proj_id, group_id, student_id in self.current_sol_fixing_data.line_up_assignments
        ]

    def _adjusted_start_values(
        self, shifted_groups: dict[tuple[int, int], tuple[int, int]]
    ) -> list[int | float]:
        start_values = dict(
            zip(
                self.derived.project_group_student_triples,
                self.current_solution.assign_students_var_values,
            )
        )
        for project_id, group_id, student_id in self.current_sol_fixing_data.line_up_assignments:
            if (new_group := shifted_groups.get((project_id, group_id))) is not None:
                old = (project_id, group_id, student_id)
                new = (*new_group, student_id)
                start_values[old], start_values[new] = start_values[new], start_values[old]

        return list(start_values.values())

    def fix_rest(self, zone_a: int, zone_b: int, num_zones: int):
        line_up_assignments = self.current_sol_fixing_data.line_up_assignments
        free_assignments, fixed_assignments = self._separate_assignments(
            zone_a, zone_b, num_zones, line_up_assignments
        )
        groups_only_free, groups_mixed = self._separate_groups(free_assignments, fixed_assignments)
        if groups_only_free:
            shifted_groups = self._shifted_groups(groups_only_free, groups_mixed)
            line_up_assignments = self._adjusted_line_up_assignments(shifted_groups)
            free_assignments, fixed_assignments = self._separate_assignments(
                zone_a, zone_b, num_zones, line_up_assignments
            )
            start_values = self._adjusted_start_values(shifted_groups)
            assignments = set(free_assignments + fixed_assignments)
        else:
            start_values = self.current_solution.assign_students_var_values
            assignments = self.current_sol_fixing_data.assignments

        self.model.setAttr("Start", self.assign_students_vars, start_values)

        free_student_ids = set(student_id for _, _, student_id in free_assignments)
        upper_bounds = [
            (
                1
                if student_id in free_student_ids
                or (project_id, group_id, student_id) in assignments
                else 0
            )
            for project_id, group_id, student_id in self.derived.project_group_student_triples
        ]
        self.model.setAttr("UB", self.assign_students_vars, upper_bounds)

    def increment_random_seed(self):
        self.model.Params.Seed += 1

    def delete_zoning_rules(self):
        self.zones.cache_clear()

    def force_k_worst_to_change(self, k: int):
        self.model.setAttr("UB", self.assign_students_vars, [1] * len(self.assign_students_vars))

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

        self.model.setAttr("Start", self.assign_students_vars, start_values)

    def free_all_unassigned_vars(self):
        variables = list(self.model_components.variables.unassigned_students.values())
        self.model.setAttr("UB", variables, [1] * len(variables))

    @classmethod
    def get(cls, initializer: ReducedModelInitializer):
        return cls(
            config=initializer.config,
            derived=initializer.derived,
            model_components=initializer.model_components,
            model=initializer.model,
            sol_reminder=initializer.current_solution,
            fixing_data=initializer.fixing_data,
            start_time=initializer.start_time,
            solution_summaries=initializer.solution_summaries,
        )
