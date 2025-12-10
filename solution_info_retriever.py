"""A class that retrieves information regarding the solution within a Gurobi model."""

import itertools
from functools import cached_property, lru_cache

from configuration import Configuration
from derived_modeling_data import DerivedModelingData
from model_components import Variables


class SolutionInformationRetriever:
    """Retrieves information regarding the solution within a Gurobi model."""

    def __init__(
        self,
        config: Configuration,
        derived: DerivedModelingData,
        variables: Variables,
    ):
        self.config = config
        self.derived = derived
        self.variables = variables

    @cached_property
    def assignments(self) -> list[tuple[int, int, int]]:
        return sorted(k for k, v in self.variables.assign_students.items() if v.X > 0.5)

    @cached_property
    def established_groups(self) -> list[tuple[int, int]]:
        return sorted(k for k, v in self.variables.establish_groups.items() if v.X > 0.5)

    @cached_property
    def unassigned_students(self) -> list[int]:
        return sorted(k for k, v in self.variables.unassigned_students.items() if v.X > 0.5)

    @lru_cache(maxsize=1_280)
    def students_in_group(self, project_id: int, group_id: int) -> list[int]:
        return sorted(
            student_id
            for proj_id, gr_id, student_id in self.assignments
            if proj_id == project_id and gr_id == group_id
        )

    @lru_cache(maxsize=128)
    def groups_in_project(self, project_id: int) -> list[int]:
        return sorted(
            group_id for proj_id, group_id in self.established_groups if proj_id == project_id
        )

    @lru_cache(maxsize=128)
    def students_in_project(self, project_id: int) -> list[int]:
        return sorted(
            student_id for proj_id, _, student_id in self.assignments if proj_id == project_id
        )

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
        return sorted(
            pair
            for pair in itertools.combinations(
                sorted(self.students_in_group(project_id, group_id)), 2
            )
            if pair in mutual_pairs
        )

    @cached_property
    def mutual_pairs(self) -> list[tuple[int, int]]:
        return sorted(
            pair
            for project_id in self.derived.project_ids
            for group_id in self.derived.group_ids[project_id]
            for pair in self.mutual_pairs_in_group(project_id, group_id)
        )
