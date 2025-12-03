"""A class that retrieves information regarding the solution within a Gurobi model."""

import itertools as it
from functools import cached_property, lru_cache

from configuration import Configuration
from constrained_model import ConstrainedModel
from derived_modeling_data import DerivedModelingData
from reduced_model import ReducedModel
from thin_wrappers import GurobiDuck


class SolutionInformationRetriever:
    """Retrieves information regarding the solution within a Gurobi model."""

    def __init__(
        self,
        config: Configuration,
        derived: DerivedModelingData,
        wrapped_model: ConstrainedModel | ReducedModel | GurobiDuck,
    ):
        self.config = config
        self.derived = derived
        self.model = wrapped_model.model
        self.variables = wrapped_model.variables
        self.lin_expressions = wrapped_model.lin_expressions

    @cached_property
    def objective_value(self):
        return self.model.ObjVal

    @cached_property
    def num_unassigned(self) -> int:
        return round(
            self.lin_expressions.sum_penalties_unassigned.getValue()
            / self.config.penalty_unassigned
        )

    @cached_property
    def unassigned_students(self) -> list[int]:
        return [
            student_id
            for student_id, is_unassigned in self.variables.unassigned_students.items()
            if round(is_unassigned.X)
        ]

    @cached_property
    def unassigned_students_names(self) -> list[str]:
        return [
            self.config.students_info["name"][student_id]
            for student_id in self.unassigned_students
        ]

    @lru_cache(maxsize=1_280)
    def num_students_in_group(self, project_id: int, group_id: int) -> int:
        return round(self.variables.assign_students.sum(project_id, group_id, "*").getValue())

    @lru_cache(maxsize=1_280)
    def students_in_group(self, project_id: int, group_id: int) -> list[int]:
        assign_students = self.variables.assign_students
        return [
            student_id
            for student_id in self.derived.student_ids
            if (project_id, group_id, student_id) in assign_students
            and round(assign_students[project_id, group_id, student_id].X)
        ]

    @lru_cache(maxsize=1_280)
    def students_in_group_names(self, project_id: int, group_id: int) -> list[str]:
        students_info = self.config.students_info
        return [
            students_info["name"][student_id]
            for student_id in self.students_in_group(project_id, group_id)
        ]

    @lru_cache(maxsize=1_280)
    def pref_vals_students_in_group(self, project_id: int, group_id: int) -> dict[int, int]:
        project_preferences = self.derived.project_preferences
        return {
            student_id: project_preferences[student_id, project_id]
            for student_id in self.students_in_group(project_id, group_id)
        }

    @lru_cache(maxsize=1_280)
    def mutual_pairs_in_group(self, project_id: int, group_id: int) -> list[tuple[int, int]]:
        mutual_pairs = self.derived.mutual_pairs_items
        return [
            pair
            for pair in it.combinations(sorted(self.students_in_group(project_id, group_id)), 2)
            if pair in mutual_pairs
        ]

    @lru_cache(maxsize=1_280)
    def num_mutual_pairs_in_group(self, project_id: int, group_id: int) -> int:
        return len(self.mutual_pairs_in_group(project_id, group_id))

    @lru_cache(maxsize=1_280)
    def mutual_pairs_in_group_names(self, project_id: int, group_id: int) -> set[tuple[str, str]]:
        students_info = self.config.students_info
        return {
            (students_info["name"][first_id], students_info["name"][second_id])
            for first_id, second_id in self.mutual_pairs_in_group(project_id, group_id)
        }

    @cached_property
    def mutual_pairs(self) -> list[tuple[int, int]]:
        return sorted(
            pair
            for project_id in self.derived.project_ids
            for group_id in self.derived.group_ids[project_id]
            for pair in self.mutual_pairs_in_group(project_id, group_id)
        )
