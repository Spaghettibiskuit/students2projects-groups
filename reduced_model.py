"""A class that contains a model which it reduces according to VNS rules."""

from __future__ import annotations

import functools
import random
from typing import TYPE_CHECKING

from base_model import BaseModelBuilder
from individual_assignment_scorer import IndividualAssignmentScorer
from patience_callback import PatienceInsideLocalSearch, PatienceOutsideLocalSearch
from solution_reminder import SolutionReminderDiving

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
        self.ranked_assignments: list[tuple[int, int, int]] | None = None
        self.ranked_assigned_ids: list[int] | None = None
        self.fixing_line_up: list[int] | None = None
        self.assignments: set[tuple[int, int, int]] | None = None

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
        self.model.Params.Cutoff = round(self.current_solution.objective_value) + 0.5

    def eliminate_cutoff(self):
        self.model.Params.Cutoff = float("-inf")

    def optimize_inside_vnd(self, patience: float | int):
        self.model.optimize(PatienceInsideLocalSearch(patience))

    def optimize_outside_vnd(self, patience: float | int):
        self.model.optimize(PatienceOutsideLocalSearch(patience))

    def store_solution(self):
        self.current_solution = SolutionReminderDiving(
            variable_values=tuple(var.X for var in self.model.getVars()),
            objective_value=self.model.ObjVal,
            assign_students_vars_values=tuple(
                var.X for var in self.variables.assign_students.values()
            ),
            mutual_unrealized_vars_values=tuple(
                var.X for var in self.variables.mutual_unrealized.values()
            ),
            unassigned_students_vars_values=tuple(
                var.X for var in self.variables.unassigned_students.values()
            ),
        )

    def store_current_solution_as_best(self):
        self.best_found_solution = self.current_solution

    def make_best_solution_current_solution(self):
        self.current_solution = self.best_found_solution

    def update_basis_for_fixing_decisions(self):
        scorer = IndividualAssignmentScorer(self.config, self.derived, self.variables)
        scores = scorer.assignment_scores
        self.ranked_assignments = sorted(scores.keys(), key=lambda k: scores[k])
        self.assignments = set(self.ranked_assignments)
        self.ranked_assigned_ids = [student_id for _, _, student_id in self.ranked_assignments]
        self._update_fixing_line_up()

    def _update_fixing_line_up(self):
        if self.ranked_assigned_ids is None:
            raise ValueError("Should have been calculated immediately before.")
        if (num_unassigned := self.config.number_of_students - len(self.ranked_assigned_ids)) > 0:
            if self.current_solution is None:
                raise ValueError("Should not be called before first solution found.")
            unassigned_ids = (
                student_id
                for student_id, val in enumerate(
                    self.current_solution.unassigned_students_vars_values
                )
                if val > 0.5
            )
            positions = random.sample(self.derived.student_ids, k=num_unassigned)
            ranked_student_ids = iter(self.ranked_assigned_ids)
            mixed_ids: list[int] = []
            for student_id in self.derived.student_ids:
                if student_id in positions:
                    mixed_ids.append(next(unassigned_ids))
                else:
                    mixed_ids.append(next(ranked_student_ids))

            if len(mixed_ids) != self.config.number_of_students:
                raise ValueError("Length does not match.")
            self.fixing_line_up = mixed_ids

        self.fixing_line_up = self.ranked_assigned_ids

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
        if self.fixing_line_up is None:
            raise ValueError("No ranking available.")
        zones = self.zones(num_zones)
        start_a, end_a = zones[zone_a]
        start_b, end_b = zones[zone_b]

        return set(self.fixing_line_up[start_a:end_a] + self.fixing_line_up[start_b:end_b])

    def fix_rest(self, zone_a: int, zone_b: int, num_zones: int):
        if self.current_solution is None or self.assignments is None:
            raise ValueError("Should not be called before first solution found.")
        allowed_to_move = self.ids_allowed_to_move(zone_a, zone_b, num_zones)

        groups_open_by_force = self._fix_rest_assign_students(allowed_to_move)
        self._fix_rest_establish_groups(groups_open_by_force)
        self._fix_rest_mutual_unrealized(allowed_to_move)

    def _fix_rest_assign_students(self, allowed_to_move: set[int]) -> set[tuple[int, int]]:
        if self.assignments is None:
            raise ValueError("Should not be called before first solution found.")

        groups_open_by_force: set[tuple[int, int]] = set()

        for (project_id, group_id, student_id), var in self.variables.assign_students.items():
            if student_id in allowed_to_move:
                var.LB = 0
                var.UB = 1

            elif (project_id, group_id, student_id) in self.assignments:
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

    def _fix_rest_mutual_unrealized(self, allowed_to_move: set[int]):
        if self.current_solution is None:
            raise ValueError("Should not be called before first solution found.")
        for ((first_id, second_id), var), val in zip(
            self.variables.mutual_unrealized.items(),
            self.current_solution.mutual_unrealized_vars_values,
        ):
            if first_id in allowed_to_move or second_id in allowed_to_move:
                var.LB = 0
                var.UB = 1

            else:
                var.LB = val
                var.UB = val

    def new_best_found(self):
        if self.current_solution is None or self.best_found_solution is None:
            raise ValueError("Should not be called before first solution found.")
        return self.current_solution.objective_value > self.best_found_solution.objective_value

    def increment_random_seed(self):
        self.model.Params.Seed += 1

    def delete_zoning_rules(self):
        self.zones.cache_clear()

    def force_k_worst_to_change(self, k: int):
        if self.ranked_assignments is None or self.fixing_line_up is None:
            raise ValueError("Both should exist by now.")
        self._free_all_possibly_fixed()
        assignments = self.ranked_assignments[:k]

        keys = [
            next((a for a in assignments if a[2] == student_id), student_id)
            for student_id in self.fixing_line_up[:k]
        ]

        for key in keys:
            if isinstance(key, int):
                var = self.variables.unassigned_students[key]
            else:
                var = self.variables.assign_students[key]

            var.LB = 0
            var.UB = 0

    def _free_all_possibly_fixed(self):

        for var_cat in (
            self.variables.assign_students,
            self.variables.establish_groups,
            self.variables.mutual_unrealized,
        ):
            self.model.setAttr("LB", list(var_cat.values()), [0] * len(var_cat))
            self.model.setAttr("UB", list(var_cat.values()), [1] * len(var_cat))

    def recover_to_best_found(self):
        if self.best_found_solution is None:
            raise ValueError()
        variables = self.model.getVars()
        var_values = self.best_found_solution.variable_values
        self.model.setAttr("LB", variables, var_values)
        self.model.setAttr("UB", variables, var_values)
        self.eliminate_time_limit()
        self.eliminate_cutoff()
        self.model.optimize()
