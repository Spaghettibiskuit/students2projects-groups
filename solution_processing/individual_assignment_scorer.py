"""Class that handles assigning a score to the assignments of students."""

import functools

from modeling.configuration import Configuration
from modeling.derived_modeling_data import DerivedModelingData
from modeling.model_components import LinExpressions, Variables
from solution_processing.solution_info_retriever import SolutionInformationRetriever


class IndividualAssignmentScorer:

    def __init__(
        self,
        config: Configuration,
        derived: DerivedModelingData,
        variables: Variables,
        lin_expressions: LinExpressions,
    ):
        self.config = config
        self.derived = derived
        self.lin_expressions = lin_expressions
        self.variables = variables
        self.retriever = SolutionInformationRetriever(config, derived, variables)

    @functools.cached_property
    def assignment_scores(self) -> dict[tuple[int, int, int], float]:
        scores = {
            assignment: self._individual_score(*assignment)
            for assignment in self.retriever.assignments
        }
        self._sum_checks()
        return scores

    def _individual_score(self, project_id: int, group_id: int, student_id: int):
        return (
            self.derived.project_preferences[student_id, project_id]
            + self._individual_reward_mutual(project_id, group_id, student_id)
            - self._individual_penalty_surplus_groups(project_id)
            - self._individual_penalty_group_size(project_id, group_id)
        )

    def _sum_checks(self):
        if not self._sum_individual_penalty_surplus_groups_matches():
            raise ValueError()

        if not self._sum_individual_penalty_group_size_matches():
            raise ValueError()

        if not self._sum_individual_reward_mutual_matches():
            raise ValueError()

    def _sum_individual_penalty_surplus_groups_matches(self) -> bool:
        derived_sum = sum(
            self._individual_penalty_surplus_groups(project_id)
            for project_id, _, _ in self.retriever.assignments
        )
        actual = self.lin_expressions.sum_penalties_surplus_groups.getValue()
        return abs(derived_sum - actual) < 1e-4

    def _sum_individual_penalty_group_size_matches(self) -> bool:
        derived_sum = sum(
            self._individual_penalty_group_size(project_id, group_id)
            for project_id, group_id, _ in self.retriever.assignments
        )
        actual = self.lin_expressions.sum_penalties_group_size.getValue()
        return abs(derived_sum - actual) < 1e-4

    def _sum_individual_reward_mutual_matches(self) -> bool:
        derived_sum = sum(
            self._individual_reward_mutual(project_id, group_id, student_id)
            for project_id, group_id, student_id in self.retriever.assignments
        )
        actual = self.lin_expressions.sum_reward_mutual.getValue()
        return abs(derived_sum - actual) < 1e-4

    @functools.lru_cache(maxsize=128)
    def _individual_penalty_surplus_groups(self, project_id: int) -> float:
        projects_info = self.config.projects_info
        penalty = projects_info["pen_groups"][project_id]
        num_desired_groups = projects_info["desired#groups"][project_id]
        num_groups = len(self.retriever.groups_in_project(project_id))
        total_num_students = len(self.retriever.students_in_project(project_id))
        return penalty * max(0, num_groups - num_desired_groups) / total_num_students

    @functools.lru_cache(maxsize=1_280)
    def _individual_penalty_group_size(self, project_id: int, group_id: int) -> float:
        projects_info = self.config.projects_info
        ideal_group_size = projects_info["ideal_group_size"][project_id]
        penalty = projects_info["pen_size"][project_id]
        num_students = len(self.retriever.students_in_group(project_id, group_id))
        return penalty * abs(ideal_group_size - num_students) / num_students

    @functools.lru_cache(maxsize=12_800)
    def _individual_reward_mutual(self, project_id: int, group_id: int, student_id: int) -> float:
        reward_mutual = self.config.reward_mutual_pair
        mutual_pairs_in_group = self.retriever.mutual_pairs_in_group(project_id, group_id)
        num_included = sum(student_id in pair for pair in mutual_pairs_in_group)
        return num_included * reward_mutual / 2
