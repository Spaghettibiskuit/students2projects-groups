"""Create model of SPAwGBP instance before any local branching constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gurobipy as gp
from gurobipy import GRB

from model_components import LinExpressions, Variables
from names import InitialConstraintNames, VariableNames

if TYPE_CHECKING:
    from configuration import Configuration
    from derived_modeling_data import DerivedModelingData


def get_base_model(
    config: Configuration,
    derived: DerivedModelingData,
) -> gp.Model:
    """Return the basic model for the instance"""
    model = gp.Model()

    projects = config.projects_info

    project_ids = derived.project_ids
    student_ids = derived.student_ids
    group_ids = derived.group_ids
    project_group_pairs = derived.project_group_pairs
    project_group_student_triples = derived.project_group_student_triples
    mutual_pairs = derived.mutual_pairs

    assign_students = model.addVars(
        project_group_student_triples,
        vtype=GRB.BINARY,
        name=VariableNames.ASSIGN_STUDENTS.value,
    )

    establish_groups = model.addVars(
        project_group_pairs, vtype=GRB.BINARY, name=VariableNames.ESTABLISH_GROUPS.value
    )

    mutual_unrealized = model.addVars(
        mutual_pairs, vtype=GRB.BINARY, name=VariableNames.MUTUAL_UNREALIZED.value
    )

    unassigned_students = model.addVars(student_ids, name=VariableNames.UNASSIGNED_STUDENTS.value)

    group_size_surplus = model.addVars(
        project_group_pairs, name=VariableNames.GROUP_SIZE_SURPLUS.value
    )

    group_size_deficit = model.addVars(
        project_group_pairs, name=VariableNames.GROUP_SIZE_DEFICIT.value
    )

    model_vars = Variables(
        assign_students=assign_students,
        establish_groups=establish_groups,
        mutual_unrealized=mutual_unrealized,
        unassigned_students=unassigned_students,
        group_size_surplus=group_size_surplus,
        group_size_deficit=group_size_deficit,
    )

    model_lin_expressions = LinExpressions.get(
        config=config, derived=derived, variables=model_vars
    )

    model.setObjective(
        model_lin_expressions.sum_realized_project_preferences
        + model_lin_expressions.sum_reward_mutual
        - model_lin_expressions.sum_penalties_unassigned
        - model_lin_expressions.sum_penalties_surplus_groups
        - model_lin_expressions.sum_penalties_group_size,
        sense=GRB.MAXIMIZE,
    )

    model.addConstrs(
        (
            assign_students.sum("*", "*", student_id) + unassigned_students[student_id] == 1
            for student_id in student_ids
        ),
        name=InitialConstraintNames.ONE_ASSIGNMENT_OR_UNASSIGNED.value,
    )

    model.addConstrs(
        (
            establish_groups[project_id, group_id] <= establish_groups[project_id, group_id - 1]
            for project_id, group_id in project_group_pairs
            if group_id > 0
        ),
        name=InitialConstraintNames.OPEN_GROUPS_CONSECUTIVELY.value,
    )

    model.addConstrs(
        (
            assign_students.sum(project_id, group_id, "*")
            >= min_group_size * establish_groups[project_id, group_id]
            for project_id, min_group_size in enumerate(projects["min_group_size"])
            for group_id in group_ids[project_id]
        ),
        name=InitialConstraintNames.MIN_GROUP_SIZE_IF_OPEN.value,
    )

    model.addConstrs(
        (
            assign_students.sum(project_id, group_id, "*")
            <= max_group_size * establish_groups[project_id, group_id]
            for project_id, max_group_size in enumerate(projects["max_group_size"])
            for group_id in group_ids[project_id]
        ),
        name=InitialConstraintNames.MAX_GROUP_SIZE_IF_OPEN.value,
    )

    model.addConstrs(
        (
            group_size_surplus[project_id, group_id]
            >= assign_students.sum(project_id, group_id, "*") - ideal_group_size
            for project_id, ideal_group_size in enumerate(projects["ideal_group_size"])
            for group_id in group_ids[project_id]
        ),
        name=InitialConstraintNames.LOWER_BOUND_GROUP_SIZE_SURPLUS.value,
    )

    model.addConstrs(
        (
            group_size_deficit[project_id, group_id]
            >= ideal_group_size
            - assign_students.sum(project_id, group_id, "*")
            - max_group_size * (1 - establish_groups[project_id, group_id])
            for project_id, ideal_group_size, max_group_size in zip(
                project_ids, projects["ideal_group_size"], projects["max_group_size"]
            )
            for group_id in group_ids[project_id]
        ),
        name=InitialConstraintNames.LOWER_BOUND_GROUP_SIZE_DEFICIT.value,
    )

    max_num_groups = max(projects["max#groups"])

    unique_group_identifiers = {
        (project_id, group_id): project_id + group_id / max_num_groups
        for project_id, group_id in project_group_pairs
    }

    num_projects = len(projects)

    model.addConstrs(
        (
            (mutual_unrealized[first, second] - unassigned_students[first]) * num_projects
            >= sum(
                unique_group_identifiers[project_id, group_id]
                * (
                    assign_students[project_id, group_id, first]
                    - assign_students[project_id, group_id, second]
                )
                for project_id, group_id in project_group_pairs
            )
            for first, second in mutual_pairs
        ),
        name=InitialConstraintNames.ONLY_REWARD_MATERIALIZED_PAIRS_1.value,
    )

    model.addConstrs(
        (
            (mutual_unrealized[first, second] - unassigned_students[second]) * num_projects
            >= sum(
                unique_group_identifiers[project_id, group_id]
                * (
                    assign_students[project_id, group_id, second]
                    - assign_students[project_id, group_id, first]
                )
                for project_id, group_id in project_group_pairs
            )
            for first, second in mutual_pairs
        ),
        name=InitialConstraintNames.ONLY_REWARD_MATERIALIZED_PAIRS_2.value,
    )

    model.update()

    return model
