"""Create model of SPAwGBP instance before any local branching constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import gurobipy as gp
from gurobipy import GRB

from model_components import InitialConstraints, LinExpressions, Variables
from names import InitialConstraintNames, VariableNames

if TYPE_CHECKING:
    from configuration import Configuration
    from derived_modeling_data import DerivedModelingData


class BaseModelBuilder:

    def __init__(self, config: Configuration, derived: DerivedModelingData):
        self.penalty_unassigned = config.penalty_unassigned
        self.reward_mutual_pair = config.reward_mutual_pair
        self.projects_info = config.projects_info
        self.students_info = config.students_info
        self.project_ids = derived.project_ids
        self.student_ids = derived.student_ids
        self.group_ids = derived.group_ids
        self.project_group_pairs = derived.project_group_pairs
        self.project_group_student_triples = derived.project_group_student_triples
        self.mutual_pairs = derived.mutual_pairs
        self.project_preferences = derived.project_preferences
        self.model = gp.Model()

    def get_base_model(self) -> tuple[Variables, LinExpressions, InitialConstraints, gp.Model]:
        variables = self._add_variables()
        lin_expressions = self._construct_linear_expressions(variables=variables)
        self._set_objective(lin_expressions=lin_expressions)
        initial_constraints = self._add_constraints(variables=variables)
        return variables, lin_expressions, initial_constraints, self.model

    def _add_variables(self) -> Variables:
        assign_students = self.model.addVars(
            self.project_group_student_triples,
            vtype=GRB.BINARY,
            name=VariableNames.ASSIGN_STUDENTS.value,
        )
        establish_groups = self.model.addVars(
            self.project_group_pairs, vtype=GRB.BINARY, name=VariableNames.ESTABLISH_GROUPS.value
        )
        mutual_unrealized = self.model.addVars(
            self.mutual_pairs, vtype=GRB.BINARY, name=VariableNames.MUTUAL_UNREALIZED.value
        )
        unassigned_students = self.model.addVars(
            self.student_ids, name=VariableNames.UNASSIGNED_STUDENTS.value
        )
        group_size_surplus = self.model.addVars(
            self.project_group_pairs, name=VariableNames.GROUP_SIZE_SURPLUS.value
        )

        group_size_deficit = self.model.addVars(
            self.project_group_pairs, name=VariableNames.GROUP_SIZE_DEFICIT.value
        )
        return Variables(
            assign_students=cast(gp.tupledict[tuple[int, int, int], gp.Var], assign_students),
            establish_groups=cast(gp.tupledict[tuple[int, int], gp.Var], establish_groups),
            mutual_unrealized=cast(gp.tupledict[tuple[int, int], gp.Var], mutual_unrealized),
            unassigned_students=cast(gp.tupledict[int, gp.Var], unassigned_students),
            group_size_surplus=cast(gp.tupledict[tuple[int, int], gp.Var], group_size_surplus),
            group_size_deficit=cast(gp.tupledict[tuple[int, int], gp.Var], group_size_deficit),
        )

    def _construct_linear_expressions(self, variables: Variables) -> LinExpressions:

        assign_students = variables.assign_students
        sum_realized_project_preferences = gp.quicksum(
            self.project_preferences[student_id, project_id]
            * assign_students[project_id, group_id, student_id]
            for project_id, group_id, student_id in self.project_group_student_triples
        )

        mutual_unrealized = variables.mutual_unrealized
        sum_reward_mutual = self.reward_mutual_pair * gp.quicksum(
            1 - mutual_unrealized[*mutual_pair] for mutual_pair in self.mutual_pairs
        )

        sum_penalties_unassigned = self.penalty_unassigned * gp.quicksum(
            variables.unassigned_students.values()
        )

        establish_groups = variables.establish_groups
        sum_penalties_surplus_groups = gp.quicksum(
            penalty_surplus_group * establish_groups[project_id, group_id]
            for project_id, penalty_surplus_group, num_groups_desired in zip(
                self.project_ids,
                self.projects_info["pen_groups"],
                self.projects_info["desired#groups"],
            )
            for group_id in self.group_ids[project_id]
            if group_id >= num_groups_desired
        )

        group_size_surplus = variables.group_size_surplus
        group_size_deficit = variables.group_size_deficit

        sum_penalties_group_size = gp.quicksum(
            penalty_size_deviation
            * (group_size_surplus[project_id, group_id] + group_size_deficit[project_id, group_id])
            for project_id, penalty_size_deviation in enumerate(self.projects_info["pen_size"])
            for group_id in self.group_ids[project_id]
        )
        return LinExpressions(
            sum_realized_project_preferences=sum_realized_project_preferences,
            sum_reward_mutual=sum_reward_mutual,
            sum_penalties_unassigned=sum_penalties_unassigned,
            sum_penalties_surplus_groups=sum_penalties_surplus_groups,
            sum_penalties_group_size=sum_penalties_group_size,
        )

    def _set_objective(self, lin_expressions: LinExpressions):
        self.model.setObjective(
            lin_expressions.sum_realized_project_preferences
            + lin_expressions.sum_reward_mutual
            - lin_expressions.sum_penalties_unassigned
            - lin_expressions.sum_penalties_surplus_groups
            - lin_expressions.sum_penalties_group_size,
            sense=GRB.MAXIMIZE,
        )

    def _add_constraints(self, variables: Variables) -> InitialConstraints:
        assign_students = variables.assign_students
        unassigned_students = variables.unassigned_students
        establish_groups = variables.establish_groups
        group_size_surplus = variables.group_size_surplus
        group_size_deficit = variables.group_size_deficit
        mutual_unrealized = variables.mutual_unrealized

        one_assignment_or_unassigned = self.model.addConstrs(
            (
                assign_students.sum("*", "*", student_id) + unassigned_students[student_id] == 1
                for student_id in self.student_ids
            ),
            name=InitialConstraintNames.ONE_ASSIGNMENT_OR_UNASSIGNED.value,
        )

        open_groups_consecutively = self.model.addConstrs(
            (
                establish_groups[project_id, group_id]
                <= establish_groups[project_id, group_id - 1]
                for project_id, group_id in self.project_group_pairs
                if group_id > 0
            ),
            name=InitialConstraintNames.OPEN_GROUPS_CONSECUTIVELY.value,
        )

        min_group_size_if_open = self.model.addConstrs(
            (
                assign_students.sum(project_id, group_id, "*")
                >= min_group_size * establish_groups[project_id, group_id]
                for project_id, min_group_size in enumerate(self.projects_info["min_group_size"])
                for group_id in self.group_ids[project_id]
            ),
            name=InitialConstraintNames.MIN_GROUP_SIZE_IF_OPEN.value,
        )

        max_group_size_if_open = self.model.addConstrs(
            (
                assign_students.sum(project_id, group_id, "*")
                <= max_group_size * establish_groups[project_id, group_id]
                for project_id, max_group_size in enumerate(self.projects_info["max_group_size"])
                for group_id in self.group_ids[project_id]
            ),
            name=InitialConstraintNames.MAX_GROUP_SIZE_IF_OPEN.value,
        )

        lower_bound_group_size_surplus = self.model.addConstrs(
            (
                group_size_surplus[project_id, group_id]
                >= assign_students.sum(project_id, group_id, "*") - ideal_group_size
                for project_id, ideal_group_size in enumerate(
                    self.projects_info["ideal_group_size"]
                )
                for group_id in self.group_ids[project_id]
            ),
            name=InitialConstraintNames.LOWER_BOUND_GROUP_SIZE_SURPLUS.value,
        )

        lower_bound_group_size_deficit = self.model.addConstrs(
            (
                group_size_deficit[project_id, group_id]
                >= ideal_group_size
                - assign_students.sum(project_id, group_id, "*")
                - max_group_size * (1 - establish_groups[project_id, group_id])
                for project_id, ideal_group_size, max_group_size in zip(
                    self.project_ids,
                    self.projects_info["ideal_group_size"],
                    self.projects_info["max_group_size"],
                )
                for group_id in self.group_ids[project_id]
            ),
            name=InitialConstraintNames.LOWER_BOUND_GROUP_SIZE_DEFICIT.value,
        )

        max_num_groups = max(self.projects_info["max#groups"])

        unique_group_identifiers = {
            (project_id, group_id): project_id + group_id / max_num_groups
            for project_id, group_id in self.project_group_pairs
        }

        num_projects = len(self.projects_info)

        only_reward_materialized_pairs_1 = self.model.addConstrs(
            (
                (mutual_unrealized[first, second] - unassigned_students[first]) * num_projects
                >= sum(
                    unique_group_identifiers[project_id, group_id]
                    * (
                        assign_students[project_id, group_id, first]
                        - assign_students[project_id, group_id, second]
                    )
                    for project_id, group_id in self.project_group_pairs
                )
                for first, second in self.mutual_pairs
            ),
            name=InitialConstraintNames.ONLY_REWARD_MATERIALIZED_PAIRS_1.value,
        )

        only_reward_materialized_pairs_2 = self.model.addConstrs(
            (
                (mutual_unrealized[first, second] - unassigned_students[second]) * num_projects
                >= sum(
                    unique_group_identifiers[project_id, group_id]
                    * (
                        assign_students[project_id, group_id, second]
                        - assign_students[project_id, group_id, first]
                    )
                    for project_id, group_id in self.project_group_pairs
                )
                for first, second in self.mutual_pairs
            ),
            name=InitialConstraintNames.ONLY_REWARD_MATERIALIZED_PAIRS_2.value,
        )

        return InitialConstraints(
            one_assignment_or_unassigned=cast(
                gp.tupledict[int, gp.Constr], one_assignment_or_unassigned
            ),
            open_groups_consecutively=cast(
                gp.tupledict[int, gp.Constr], open_groups_consecutively
            ),
            min_group_size_if_open=cast(
                gp.tupledict[tuple[int, int, int], gp.Constr], min_group_size_if_open
            ),
            max_group_size_if_open=cast(
                gp.tupledict[tuple[int, int, int], gp.Constr], max_group_size_if_open
            ),
            lower_bound_group_size_surplus=cast(
                gp.tupledict[tuple[int, int, int], gp.Constr], lower_bound_group_size_surplus
            ),
            lower_bound_group_size_deficit=cast(
                gp.tupledict[tuple[int, int, int, int], gp.Constr], lower_bound_group_size_deficit
            ),
            only_reward_materialized_pairs_1=cast(
                gp.tupledict[tuple[int, int], gp.Constr], only_reward_materialized_pairs_1
            ),
            only_reward_materialized_pairs_2=cast(
                gp.tupledict[tuple[int, int], gp.Constr], only_reward_materialized_pairs_2
            ),
        )
