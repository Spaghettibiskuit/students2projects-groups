"""A class that contains a model which it reduces according to VNS rules."""

import collections
import functools
import random

import gurobipy

from configuration import Configuration
from derived_modeling_data import DerivedModelingData
from fixing_data import FixingByRankingData
from model_components import ModelComponents
from model_wrapper import ModelWrapper
from solution_reminders import SolutionReminderDiving
from thin_wrappers import ReducedModelInitializer
from utilities import var_values


class ReducedModel(ModelWrapper):
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
        fixing_data: FixingByRankingData,
    ):
        super().__init__(model_components, model, start_time, solution_summaries, sol_reminder)
        self.config = config
        self.derived = derived
        self.current_sol_fixing_data = fixing_data
        self.best_sol_fixing_data = fixing_data
        self.current_solution: SolutionReminderDiving
        self.best_found_solution: SolutionReminderDiving

    def store_solution(self):
        variables = self.model_components.variables
        self.current_solution = SolutionReminderDiving(
            variable_values=var_values(self.model.getVars()),
            objective_value=self.objective_value,
            assign_students_var_values=var_values(variables.assign_students.values()),
            mutual_unrealized_var_values=var_values(variables.mutual_unrealized.values()),
            unassigned_students_var_values=var_values(variables.unassigned_students.values()),
        )

        self.current_sol_fixing_data = FixingByRankingData.get(
            self.config, self.derived, variables
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

    def _ids_allowed_to_move(self, zone_a: int, zone_b: int, num_zones: int) -> set[int]:
        lin_up_ids = self.current_sol_fixing_data.line_up_ids
        zones = self.zones(num_zones)
        start_a, end_a = zones[zone_a]
        start_b, end_b = zones[zone_b]

        return set(lin_up_ids[start_a:end_a] + lin_up_ids[start_b:end_b])

    def _fixation_info(
        self, allowed_to_move: set[int]
    ) -> tuple[set[int], list[tuple[int, int, int]], dict[int, dict[int, int]]]:
        unassigned_to_be_fixed: set[int] = set()
        assignments_to_be_fixed: list[tuple[int, int, int]] = []
        group_ids: collections.defaultdict[int, set[int]] = collections.defaultdict(set)

        for project_id, group_id, student_id in self.current_sol_fixing_data.line_up_assignments:
            if student_id in allowed_to_move:
                continue
            if project_id == -1:
                unassigned_to_be_fixed.add(student_id)
            else:
                group_ids[project_id].add(group_id)
                assignments_to_be_fixed.append((project_id, group_id, student_id))

        shifted_group_ids: dict[int, dict[int, int]] = {}
        for project_id in sorted(group_ids.keys()):
            ids = group_ids[project_id]
            shifted_group_ids[project_id] = dict(zip(sorted(ids), range(len(ids))))

        return unassigned_to_be_fixed, assignments_to_be_fixed, shifted_group_ids

    def fix_rest(self, zone_a: int, zone_b: int, num_zones: int):
        allowed_to_move = self._ids_allowed_to_move(zone_a, zone_b, num_zones)

        unassigned_to_be_fixed, assignments_to_be_fixed, shifted_group_ids = self._fixation_info(
            allowed_to_move
        )
        groups_open_by_force = self._fix_rest_assign_students(
            allowed_to_move, assignments_to_be_fixed, shifted_group_ids
        )
        self._fix_rest_establish_groups(groups_open_by_force)
        self._fix_rest_unassigned_students(allowed_to_move, unassigned_to_be_fixed)
        self._fix_rest_mutual_unrealized(allowed_to_move)

    def _fix_rest_assign_students(
        self,
        allowed_to_move: set[int],
        assignments_to_be_fixed: list[tuple[int, int, int]],
        shifted_group_ids: dict[int, dict[int, int]],
    ) -> set[tuple[int, int]]:
        groups_open_by_force: set[tuple[int, int]] = set()
        assign_students = self.model_components.variables.assign_students
        for (project_id, group_id, student_id), var in assign_students.items():
            if student_id in allowed_to_move:
                var.LB = 0
                var.UB = 1
            else:
                var.LB = 0
                var.UB = 0

        for project_id, group_id, student_id in assignments_to_be_fixed:
            new_group_id = shifted_group_ids[project_id][group_id]
            groups_open_by_force.add((project_id, new_group_id))
            var = assign_students[project_id, new_group_id, student_id]
            var.UB = 1
            var.LB = 1

        return groups_open_by_force

    def _fix_rest_establish_groups(self, groups_open_by_force: set[tuple[int, int]]):
        for key, var in self.model_components.variables.establish_groups.items():
            if key in groups_open_by_force:
                var.LB = 1
                var.UB = 1
            else:
                var.LB = 0
                var.UB = 1

    def _fix_rest_unassigned_students(
        self, allowed_to_move: set[int], unassigned_to_be_fixed: set[int]
    ):
        for student_id, var in self.model_components.variables.unassigned_students.items():
            if student_id in allowed_to_move:
                var.LB = 0
                var.UB = 1

            elif student_id in unassigned_to_be_fixed:
                var.LB = 1
                var.UB = 1

            else:
                var.LB = 0
                var.UB = 0

    def _fix_rest_mutual_unrealized(self, allowed_to_move: set[int]):
        for ((first_id, second_id), var), val in zip(
            self.model_components.variables.mutual_unrealized.items(),
            self.current_solution.mutual_unrealized_var_values,
        ):
            if first_id in allowed_to_move or second_id in allowed_to_move:
                var.LB = 0
                var.UB = 1

            else:
                var.LB = val
                var.UB = val

    def new_best_found(self):
        return self.current_solution.objective_value > self.best_found_solution.objective_value

    def increment_random_seed(self):
        self.model.Params.Seed += 1

    def delete_zoning_rules(self):
        self.zones.cache_clear()

    def force_k_worst_to_change(self, k: int):
        self._free_all_possibly_fixed()
        worst_k = self.current_sol_fixing_data.line_up_assignments[:k]
        variables = self.model_components.variables

        for project_id, group_id, student_id in worst_k:
            if project_id == -1:  # It is a pseudo_assignment
                var = variables.unassigned_students[student_id]
            else:
                var = variables.assign_students[project_id, group_id, student_id]

            var.LB = 0
            var.UB = 0

    def _free_all_possibly_fixed(self):
        for var_cat in (
            self.model_components.variables.assign_students,
            self.model_components.variables.establish_groups,
            self.model_components.variables.mutual_unrealized,
            self.model_components.variables.unassigned_students,
        ):
            variables = list(var_cat.values())
            self.model.setAttr("LB", variables, [0] * len(var_cat))
            self.model.setAttr("UB", variables, [1] * len(var_cat))

    def recover_to_best_found(self):
        variables = self.model.getVars()
        variable_values = self.best_found_solution.variable_values
        self.model.setAttr("LB", variables, variable_values)
        self.model.setAttr("UB", variables, variable_values)
        self.eliminate_time_limit()
        self.eliminate_cutoff()
        self.model.optimize()

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
