"""A class that checks whether a solution that was found is valid and correct."""

import functools

from configuration import Configuration
from derived_modeling_data import DerivedModelingData
from model_components import Variables
from solution_info_retriever import SolutionInformationRetriever


class SolutionChecker:
    """Checks whether a solution that was found is valid and correct."""

    def __init__(
        self,
        config: Configuration,
        derived: DerivedModelingData,
        variables: Variables,
        retriever: SolutionInformationRetriever,
    ):
        self.config = config
        self.derived = derived
        self.variables = variables
        self.retriever = retriever

    @functools.cached_property
    def assigned_students_triples(self) -> list[tuple[int, int, int]]:
        assign_students = self.variables.assign_students
        return [key for key, var in assign_students.items() if var.X > 0.5]

    @functools.cached_property
    def established_groups_pairs(self) -> list[tuple[int, int]]:
        establish_groups = self.variables.establish_groups
        return [key for key, var in establish_groups.items() if var.X > 0.5]

    @functools.lru_cache(maxsize=128)
    def groups_in_project(self, project_id: int) -> list[int]:
        return [
            group_id
            for proj_id, group_id in self.established_groups_pairs
            if proj_id == project_id
        ]

    def all_students_either_assigned_once_or_unassigned(self) -> bool:
        unassigned_students = self.retriever.unassigned_students
        assigned_students = [student_id for _, _, student_id in self.assigned_students_triples]
        all_student_ids = list(range(self.config.number_of_students))
        return sorted(unassigned_students + assigned_students) == all_student_ids

    def groups_opened_if_and_only_if_students_inside(self) -> bool:
        derived_pairs = set(
            (project_id, group_id) for project_id, group_id, _ in self.assigned_students_triples
        )
        return sorted(derived_pairs) == self.established_groups_pairs

    def all_group_sizes_within_bounds(self) -> bool:
        projects_info = self.config.projects_info
        return all(
            projects_info["min_group_size"][project_id]
            <= len(self.retriever.students_in_group(project_id, group_id))
            <= projects_info["max_group_size"][project_id]
            for project_id, group_id in self.established_groups_pairs
        )

    def no_project_too_many_established_groups(self) -> bool:
        projects_info = self.config.projects_info
        return all(
            len(self.groups_in_project(project_id)) <= projects_info["max#groups"][project_id]
            for project_id in self.derived.project_ids
        )

    def is_valid(self) -> bool:
        return (
            self.all_students_either_assigned_once_or_unassigned()
            and self.groups_opened_if_and_only_if_students_inside()
            and self.all_group_sizes_within_bounds()
            and self.no_project_too_many_established_groups()
            and self.no_project_too_many_established_groups()
        )
