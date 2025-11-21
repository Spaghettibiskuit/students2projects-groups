"""Class that handles assigning a score to the assignments of students."""

from __future__ import annotations

import functools
import itertools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from configuration import Configuration
    from derived_modeling_data import DerivedModelingData
    from model_components import Variables


class IndividualAssignmentScorer:

    def __init__(
        self,
        config: Configuration,
        derived: DerivedModelingData,
        variables: Variables,
    ):
        self.config = config
        self.derived = derived
        self.variables = variables

    @functools.cached_property
    def _assignments(self) -> list[tuple[int, int, int]]:
        assign_students = self.variables.assign_students
        return [k for k, v in assign_students.items() if v.X > 0.5]

    @functools.cached_property
    def assignment_scores(self):
        return {
            assignment: self._individual_score(*assignment) for assignment in self._assignments
        }

    def _individual_score(self, project_id: int, group_id: int, student_id: int):

        return (
            self.derived.project_preferences[student_id, project_id]
            + self._individual_reward_mutual(project_id, group_id, student_id)
            + self._individual_penalty_surplus_groups(project_id)
            + self._individual_penalty_group_size(project_id, group_id)
        )

    @functools.lru_cache(maxsize=128)
    def _num_students_in_project(self, project_id: int) -> int:
        return sum(proj_id == project_id for proj_id, _, _ in self._assignments)

    @functools.lru_cache(1_280)
    def _students_in_group(self, project_id: int, group_id: int) -> list[int]:
        return sorted(
            student_id
            for proj_id, gr_id, student_id in self._assignments
            if proj_id == project_id and gr_id == group_id
        )

    @functools.lru_cache(maxsize=128)
    def _groups_in_project(self, project_id: int) -> list[int]:
        return [group_id for proj_id, group_id, _ in self._assignments if proj_id == project_id]

    @functools.lru_cache(maxsize=1_280)
    def _mutual_pairs_in_group(self, project_id: int, group_id: int) -> list[tuple[int, int]]:
        pairs = set(itertools.combinations(self._students_in_group(project_id, group_id), 2))
        return [pair for pair in self.derived.mutual_pairs if pair in pairs]

    @functools.lru_cache(maxsize=128)
    def _individual_penalty_surplus_groups(self, project_id: int) -> float:
        projects_info = self.config.projects_info
        penalty = projects_info["pen_groups"][project_id]
        num_desired_groups = projects_info["desired#groups"][project_id]
        num_groups = len(self._groups_in_project(project_id))
        total_num_students = self._num_students_in_project(project_id)
        return penalty * max(0, num_groups - num_desired_groups) / total_num_students

    @functools.lru_cache(maxsize=1_280)
    def _individual_penalty_group_size(self, project_id: int, group_id: int) -> float:
        projects_info = self.config.projects_info
        ideal_group_size = projects_info["ideal_group_size"][project_id]
        penalty = projects_info["pen_groups"][project_id]
        num_students = len(self._students_in_group(project_id, group_id))
        return penalty * abs(ideal_group_size - num_students) / num_students

    def _individual_reward_mutual(self, project_id: int, group_id: int, student_id: int) -> float:
        reward_mutual = self.config.reward_mutual_pair
        mutual_pairs_in_group = self._mutual_pairs_in_group(project_id, group_id)
        num_included = sum(student_id in pair for pair in mutual_pairs_in_group)
        return num_included * reward_mutual / 2
