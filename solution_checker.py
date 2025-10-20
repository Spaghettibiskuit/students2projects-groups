"""A class that checks whether a solution that was found is valid and correct."""

import functools

from configuration import Configuration
from derived_modeling_data import DerivedModelingData
from model_components import LinExpressions, Variables
from solution_info_retriever import SolutionInformationRetriever


class SolutionChecker:
    """Checks whether a solution that was found is valid and correct."""

    def __init__(
        self,
        config: Configuration,
        derived: DerivedModelingData,
        variables: Variables,
        lin_expressions: LinExpressions,
        retriever: SolutionInformationRetriever,
    ):
        self.config = config
        self.derived = derived
        self.variables = variables
        self.lin_expressions = lin_expressions
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

    @functools.cached_property
    def all_students_either_assigned_once_or_unassigned(self) -> bool:
        unassigned_students = self.retriever.unassigned_students
        assigned_students = [student_id for _, _, student_id in self.assigned_students_triples]
        all_student_ids = list(range(self.config.number_of_students))
        return sorted(unassigned_students + assigned_students) == all_student_ids

    @functools.cached_property
    def groups_opened_if_and_only_if_students_inside(self) -> bool:
        derived_pairs = set(
            (project_id, group_id) for project_id, group_id, _ in self.assigned_students_triples
        )
        return sorted(derived_pairs) == self.established_groups_pairs

    @functools.cached_property
    def all_group_sizes_within_bounds(self) -> bool:
        projects_info = self.config.projects_info
        return all(
            projects_info["min_group_size"][project_id]
            <= len(self.retriever.students_in_group(project_id, group_id))
            <= projects_info["max_group_size"][project_id]
            for project_id, group_id in self.established_groups_pairs
        )

    @functools.cached_property
    def no_project_too_many_established_groups(self) -> bool:
        projects_info = self.config.projects_info
        return all(
            len(self.groups_in_project(project_id)) <= projects_info["max#groups"][project_id]
            for project_id in self.derived.project_ids
        )

    @functools.cached_property
    def is_valid(self) -> bool:
        return (
            self.all_students_either_assigned_once_or_unassigned
            and self.groups_opened_if_and_only_if_students_inside
            and self.all_group_sizes_within_bounds
            and self.no_project_too_many_established_groups
        )

    @functools.cached_property
    def sum_realized_project_preferences(self) -> int | float:
        project_preferences = self.derived.project_preferences
        return sum(
            project_preferences[student_id, project_id]
            for project_id, _, student_id in self.assigned_students_triples
        )

    @functools.cached_property
    def sum_reward_mutual(self) -> int | float:
        return len(self.retriever.mutual_pairs) * self.config.reward_mutual_pair

    @functools.cached_property
    def sum_penalties_unassigned(self) -> int | float:
        return len(self.retriever.unassigned_students) * self.config.penalty_unassigned

    @functools.cached_property
    def sum_penalties_surplus_groups(self) -> int | float:
        desired_nums_of_groups = self.config.projects_info["desired#groups"]
        penalties_per_excess_group = self.config.projects_info["pen_groups"]
        return sum(
            max(0, len(self.groups_in_project(project_id)) - desired_nums_of_groups[project_id])
            * penalties_per_excess_group[project_id]
            for project_id in self.derived.project_ids
        )

    @functools.cached_property
    def sum_penalties_group_size(self) -> int | float:
        ideal_group_sizes = self.config.projects_info["ideal_group_size"]
        penalties_deviation = self.config.projects_info["pen_size"]
        return sum(
            abs(
                len(self.retriever.students_in_group(project_id, group_id))
                - ideal_group_sizes[project_id]
            )
            * penalties_deviation[project_id]
            for project_id, group_id in self.established_groups_pairs
        )

    @functools.cached_property
    def objective_value_calculated_correctly(self) -> bool:
        lin_exprs = self.lin_expressions
        return (
            self.sum_realized_project_preferences
            == lin_exprs.sum_realized_project_preferences.getValue()
            and self.sum_reward_mutual == lin_exprs.sum_reward_mutual.getValue()
            and self.sum_penalties_unassigned == lin_exprs.sum_penalties_unassigned.getValue()
            and self.sum_penalties_surplus_groups
            == lin_exprs.sum_penalties_surplus_groups.getValue()
            and self.sum_penalties_group_size == lin_exprs.sum_penalties_group_size.getValue()
        )

    @functools.cached_property
    def is_correct(self) -> bool:
        return self.is_valid and self.objective_value_calculated_correctly
