"""A class that checks whether a solution that was found is valid and correct."""

import functools

from modeling.configuration import Configuration
from modeling.derived_modeling_data import DerivedModelingData
from modeling.model_components import LinExpressions
from solution_processing.solution_info_retriever import SolutionInformationRetriever


class SolutionChecker:
    """Checks whether a solution that was found is valid and correct."""

    def __init__(
        self,
        config: Configuration,
        derived: DerivedModelingData,
        lin_expressions: LinExpressions,
        retriever: SolutionInformationRetriever,
    ):
        self.config = config
        self.derived = derived
        self.lin_expressions = lin_expressions
        self.retriever = retriever

    @functools.cached_property
    def all_students_either_assigned_once_or_unassigned(self) -> bool:
        unassigned_students = self.retriever.unassigned_students
        assigned_students = [student_id for _, _, student_id in self.retriever.assignments]
        all_student_ids = list(self.derived.student_ids)
        return sorted(unassigned_students + assigned_students) == all_student_ids

    @functools.cached_property
    def groups_opened_if_and_only_if_students_inside(self) -> bool:
        derived_open_groups = set(
            (project_id, group_id) for project_id, group_id, _ in self.retriever.assignments
        )
        return sorted(derived_open_groups) == self.retriever.established_groups

    @functools.cached_property
    def all_group_sizes_within_bounds(self) -> bool:
        projects_info = self.config.projects_info
        return all(
            projects_info["min_group_size"][project_id]
            <= len(self.retriever.students_in_group(project_id, group_id))
            <= projects_info["max_group_size"][project_id]
            for project_id, group_id in self.retriever.established_groups
        )

    @functools.cached_property
    def no_project_too_many_established_groups(self) -> bool:
        projects_info = self.config.projects_info
        return all(
            len(self.retriever.groups_in_project(project_id))
            <= projects_info["max#groups"][project_id]
            for project_id in self.derived.project_ids
        )

    @functools.cached_property
    def all_projects_only_consecutive_group_ids(self) -> bool:
        return all(
            sorted(groups := self.retriever.groups_in_project(project_id))
            == list(range(len(groups)))
            for project_id in self.derived.project_ids
        )

    @functools.cached_property
    def is_valid(self) -> bool:
        return (
            self.all_students_either_assigned_once_or_unassigned
            and self.groups_opened_if_and_only_if_students_inside
            and self.all_group_sizes_within_bounds
            and self.no_project_too_many_established_groups
            and self.all_projects_only_consecutive_group_ids
        )

    @functools.cached_property
    def sum_realized_project_preferences(self) -> int | float:
        project_preferences = self.derived.project_preferences
        return sum(
            project_preferences[student_id, project_id]
            for project_id, _, student_id in self.retriever.assignments
        )

    @functools.cached_property
    def sum_reward_mutual(self) -> int | float:
        return len(self.retriever.mutual_pairs) * self.config.reward_mutual_pair

    @functools.cached_property
    def sum_penalties_unassigned(self) -> int | float:
        return len(self.retriever.unassigned_students) * self.config.penalty_unassigned

    @functools.cached_property
    def sum_penalties_surplus_groups(self) -> int | float:
        desired_num_of_groups = self.config.projects_info["desired#groups"]
        penalty_per_excess_group = self.config.projects_info["pen_groups"]
        return sum(
            max(
                0,
                len(self.retriever.groups_in_project(project_id))
                - desired_num_of_groups[project_id],
            )
            * penalty_per_excess_group[project_id]
            for project_id in self.derived.project_ids
        )

    @functools.cached_property
    def sum_penalties_group_size(self) -> int | float:
        ideal_group_size = self.config.projects_info["ideal_group_size"]
        penalty_deviation = self.config.projects_info["pen_size"]
        return sum(
            abs(
                len(self.retriever.students_in_group(project_id, group_id))
                - ideal_group_size[project_id]
            )
            * penalty_deviation[project_id]
            for project_id, group_id in self.retriever.established_groups
        )

    @functools.cached_property
    def sum_realized_project_preferences_correct(self) -> bool:
        return (
            self.sum_realized_project_preferences
            == self.lin_expressions.sum_realized_project_preferences.getValue()
        )

    @functools.cached_property
    def sum_reward_mutual_correct(self) -> bool:
        return self.sum_reward_mutual == self.lin_expressions.sum_reward_mutual.getValue()

    @functools.cached_property
    def sum_penalties_unassigned_correct(self) -> bool:
        return (
            self.sum_penalties_unassigned
            == self.lin_expressions.sum_penalties_unassigned.getValue()
        )

    @functools.cached_property
    def sum_penalties_surplus_groups_correct(self) -> bool:
        return (
            self.sum_penalties_surplus_groups
            == self.lin_expressions.sum_penalties_surplus_groups.getValue()
        )

    @functools.cached_property
    def sum_penalties_group_size_correct(self) -> bool:
        return (
            self.sum_penalties_group_size
            == self.lin_expressions.sum_penalties_group_size.getValue()
        )

    @functools.cached_property
    def objective_value_calculated_correctly(self) -> bool:
        return (
            self.sum_realized_project_preferences_correct
            and self.sum_reward_mutual_correct
            and self.sum_penalties_unassigned_correct
            and self.sum_penalties_surplus_groups_correct
            and self.sum_penalties_group_size_correct
        )

    @functools.cached_property
    def is_correct(self) -> bool:
        return self.is_valid and self.objective_value_calculated_correctly
