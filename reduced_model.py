"""A class that contains a model which it reduces according to VNS rules."""

from __future__ import annotations

import functools
import random
from typing import TYPE_CHECKING

from base_model import BaseModelBuilder
from fixing_data import FixingData
from patience_callback import PatienceInsideLocalSearch, PatienceOutsideLocalSearch
from solution_reminder import SolutionReminderDiving
from utilities import var_values

if TYPE_CHECKING:
    from configuration import Configuration
    from derived_modeling_data import DerivedModelingData


class ReducedModel:
    """Contains a model further constrained for local branching."""

    def __init__(self, config: Configuration, derived: DerivedModelingData):
        self.config = config
        self.derived = derived
        self.variables, self.lin_expressions, self.initial_constraints, self.model = (
            BaseModelBuilder(config=self.config, derived=self.derived).get_base_model()
        )
        self.current_solution: SolutionReminderDiving | None = None
        self.best_found_solution: SolutionReminderDiving | None = None
        self.current_sol_fixing_data: FixingData | None = None
        self.best_sol_fixing_data: FixingData | None = None

    @property
    def status(self):
        return self.model.Status

    @property
    def solution_count(self):
        return self.model.SolCount

    def set_time_limit(self, time_limit: int | float):
        self.model.Params.TimeLimit = time_limit

    def eliminate_time_limit(self):
        self.model.Params.TimeLimit = float("inf")

    def set_cutoff(self):
        if self.current_solution is None:
            raise ValueError()
        self.model.Params.Cutoff = round(self.current_solution.objective_value) + 1 - 1e-6

    def eliminate_cutoff(self):
        self.model.Params.Cutoff = float("-inf")

    def optimize_inside_vnd(self, patience: float | int):
        self.model.optimize(PatienceInsideLocalSearch(patience))

    def optimize_outside_vnd(self, patience: float | int):
        self.model.optimize(PatienceOutsideLocalSearch(patience))

    def store_solution(self):
        self.current_solution = SolutionReminderDiving(
            variable_values=var_values(self.model.getVars()),
            objective_value=self.model.ObjVal,
            assign_students_var_values=var_values(self.variables.assign_students.values()),
            mutual_unrealized_var_values=var_values(self.variables.mutual_unrealized.values()),
            unassigned_students_var_values=var_values(self.variables.unassigned_students.values()),
        )

        self.current_sol_fixing_data = FixingData.get(self.config, self.derived, self.variables)

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

    def ids_allowed_to_move(self, zone_a: int, zone_b: int, num_zones: int) -> set[int]:
        if self.current_sol_fixing_data is None:
            raise ValueError()
        lin_up_ids = self.current_sol_fixing_data.line_up_ids
        zones = self.zones(num_zones)
        start_a, end_a = zones[zone_a]
        start_b, end_b = zones[zone_b]

        return set(lin_up_ids[start_a:end_a] + lin_up_ids[start_b:end_b])

    def fix_rest(self, zone_a: int, zone_b: int, num_zones: int):
        allowed_to_move = self.ids_allowed_to_move(zone_a, zone_b, num_zones)

        groups_open_by_force = self._fix_rest_assign_students(allowed_to_move)
        self._fix_rest_establish_groups(groups_open_by_force)
        # self._fix_rest_unassigned_students(allowed_to_move)
        self._fix_rest_mutual_unrealized(allowed_to_move)

    def _fix_rest_assign_students(self, allowed_to_move: set[int]) -> set[tuple[int, int]]:
        if self.current_sol_fixing_data is None:
            raise ValueError()
        assignments = self.current_sol_fixing_data.assignments
        groups_open_by_force: set[tuple[int, int]] = set()

        for (project_id, group_id, student_id), var in self.variables.assign_students.items():
            if student_id in allowed_to_move:
                var.LB = 0
                var.UB = 1

            elif (project_id, group_id, student_id) in assignments:
                var.LB = 1
                var.UB = 1
                groups_open_by_force.add((project_id, group_id))

            else:
                var.LB = 0
                var.UB = 0

        return groups_open_by_force

    def _fix_rest_establish_groups(self, groups_open_by_force: set[tuple[int, int]]):
        for key, var in self.variables.establish_groups.items():
            if key in groups_open_by_force:
                var.LB = 1
                var.UB = 1
            else:
                var.LB = 0
                var.UB = 1

    def _fix_rest_unassigned_students(self, allowed_to_move: set[int]):
        if self.current_sol_fixing_data is None:
            raise ValueError()

        if self.current_sol_fixing_data.unassigned_ids.difference(allowed_to_move):
            self.__fix_rest_unassigned_students_some_unassigned(allowed_to_move)
            return

        for student_id, var in self.variables.unassigned_students.items():
            if student_id in allowed_to_move:
                var.LB = 0
                var.UB = 1

            else:
                var.LB = 0
                var.UB = 0

    def __fix_rest_unassigned_students_some_unassigned(self, alllowed_to_move: set[int]):
        if self.current_sol_fixing_data is None:
            raise ValueError()

        unassigned_ids = self.current_sol_fixing_data.unassigned_ids

        for student_id, var in self.variables.unassigned_students.items():
            if student_id in alllowed_to_move:
                var.LB = 0
                var.UB = 1

            elif student_id in unassigned_ids:
                var.LB = 1
                var.UB = 1

            else:
                var.LB = 0
                var.UB = 0

    def _fix_rest_mutual_unrealized(self, allowed_to_move: set[int]):
        if self.current_solution is None:
            raise ValueError("Should not be called before first solution found.")
        for ((first_id, second_id), var), val in zip(
            self.variables.mutual_unrealized.items(),
            self.current_solution.mutual_unrealized_var_values,
        ):
            if first_id in allowed_to_move or second_id in allowed_to_move:
                var.LB = 0
                var.UB = 1

            else:
                var.LB = val
                var.UB = val

    def new_best_found(self):
        if self.current_solution is None or self.best_found_solution is None:
            raise ValueError()
        return self.current_solution.objective_value > self.best_found_solution.objective_value

    def increment_random_seed(self):
        self.model.Params.Seed += 1

    def delete_zoning_rules(self):
        self.zones.cache_clear()

    def force_k_worst_to_change(self, k: int):
        if self.current_sol_fixing_data is None:
            raise ValueError()

        self._free_all_possibly_fixed()
        worst_k = self.current_sol_fixing_data.line_up_assignments[:k]

        for project_id, group_id, student_id in worst_k:
            if project_id == -1:  # It is a pseudo_assignment
                var = self.variables.unassigned_students[student_id]
            else:
                var = self.variables.assign_students[project_id, group_id, student_id]

            var.LB = 0
            var.UB = 0

    def _free_all_possibly_fixed(self):

        for var_cat in (
            self.variables.assign_students,
            self.variables.establish_groups,
            self.variables.mutual_unrealized,
            self.variables.unassigned_students,
        ):
            self.model.setAttr("LB", list(var_cat.values()), [0] * len(var_cat))
            self.model.setAttr("UB", list(var_cat.values()), [1] * len(var_cat))

    def recover_to_best_found(self):
        if self.best_found_solution is None:
            raise ValueError()
        variables = self.model.getVars()
        variable_values = self.best_found_solution.variable_values
        self.model.setAttr("LB", variables, variable_values)
        self.model.setAttr("UB", variables, variable_values)
        self.eliminate_time_limit()
        self.eliminate_cutoff()
        self.model.optimize()
