"""Create model of SPAwGBP instance before any local branching constraints."""

import typing

import gurobipy as gp
import pandas as pd
from gurobipy import GRB

type ProjectsInfo = pd.DataFrame
type StudentsInfo = pd.DataFrame
type VarKey = tuple[int, int, int] | tuple[int, int] | int
type ModelVariables = gp.tupledict[tuple[typing.Any, ...], gp.Var]


def base_model(
    projects: ProjectsInfo,
    reward_bilateral: int,
    penalty_unassigned: int,
    project_ids: range,
    student_ids: range,
    group_ids: dict[int, range],
    project_group_pairs: list[tuple[int, int]],
    project_group_student_triples: list[tuple[int, int, int]],
    mutual_pairs: set[tuple[int, int]],
    project_preferences: dict[tuple[int, int], int],
) -> tuple[gp.Model, dict[str, ModelVariables], dict[str, gp.LinExpr]]:
    """Return the basic model for the instance"""
    model = gp.Model()

    variables: dict[str, ModelVariables] = {}

    assign_students = model.addVars(
        project_group_student_triples,
        vtype=GRB.BINARY,
        name="assign_students",
    )
    variables["assign_students"] = assign_students

    establish_groups = model.addVars(
        project_group_pairs,
        vtype=GRB.BINARY,
        name="establish_groups",
    )
    variables["establish_groups"] = establish_groups

    mutual_unrealized = model.addVars(mutual_pairs, vtype=GRB.BINARY, name="mutual_unrealized")
    variables["mutual_unrealized"] = mutual_unrealized

    unassigned_students = model.addVars(student_ids, name="unassigned_students")
    variables["unassigned_students"] = unassigned_students

    group_size_surplus = model.addVars(project_group_pairs, name="group_size_surplus")
    variables["group_size_surplus"] = group_size_surplus

    group_size_deficit = model.addVars(project_group_pairs, name="group_size_deficit")
    variables["group_size_deficit"] = group_size_deficit

    linear_expressions: dict[str, gp.LinExpr] = {}

    sum_realized_project_preferences = gp.quicksum(
        project_preferences[student_id, project_id]
        * assign_students[project_id, group_id, student_id]
        for project_id, group_id, student_id in project_group_student_triples
    )
    linear_expressions["sum_realized_project_preferences"] = sum_realized_project_preferences

    sum_reward_bilateral = reward_bilateral * gp.quicksum(
        1 - mutual_unrealized[*mutual_pair] for mutual_pair in mutual_pairs
    )
    linear_expressions["sum_reward_bilateral"] = sum_reward_bilateral

    sum_penalties_unassigned = penalty_unassigned * gp.quicksum(unassigned_students.values())
    linear_expressions["sum_penalties_unassigned"] = sum_penalties_unassigned

    sum_penalties_surplus_groups = gp.quicksum(
        penalty_surplus_group * establish_groups[project_id, group_id]
        for project_id, penalty_surplus_group, num_groups_desired in zip(
            project_ids, projects["pen_groups"], projects["desired#groups"]
        )
        for group_id in group_ids[project_id]
        if group_id >= num_groups_desired
    )
    linear_expressions["sum_penalties_surplus_groups"] = sum_penalties_surplus_groups

    sum_penalties_group_size = gp.quicksum(
        penalty_size_deviation
        * (group_size_surplus[project_id, group_id] + group_size_deficit[project_id, group_id])
        for project_id, penalty_size_deviation in enumerate(projects["pen_size"])
        for group_id in group_ids[project_id]
    )
    linear_expressions["sum_penalties_group_size"] = sum_penalties_group_size

    model.setObjective(
        sum_realized_project_preferences
        + sum_reward_bilateral
        - sum_penalties_unassigned
        - sum_penalties_surplus_groups
        - sum_penalties_group_size,
        sense=GRB.MAXIMIZE,
    )

    model.addConstrs(
        (
            assign_students.sum("*", "*", student_id) + unassigned_students[student_id] == 1
            for student_id in student_ids
        ),
        name="one_assignment_or_unassigned",
    )

    model.addConstrs(
        (
            establish_groups[project_id, group_id] <= establish_groups[project_id, group_id - 1]
            for project_id, group_id in project_group_pairs
            if group_id > 0
        ),
        name="open_groups_consecutively",
    )

    model.addConstrs(
        (
            assign_students.sum(project_id, group_id, "*")
            >= min_group_size * establish_groups[project_id, group_id]
            for project_id, min_group_size in enumerate(projects["min_group_size"])
            for group_id in group_ids[project_id]
        ),
        name="min_group_size_if_open",
    )

    model.addConstrs(
        (
            assign_students.sum(project_id, group_id, "*")
            <= max_group_size * establish_groups[project_id, group_id]
            for project_id, max_group_size in enumerate(projects["max_group_size"])
            for group_id in group_ids[project_id]
        ),
        name="max_group_size_if_open",
    )

    model.addConstrs(
        (
            group_size_surplus[project_id, group_id]
            >= assign_students.sum(project_id, group_id, "*") - ideal_group_size
            for project_id, ideal_group_size in enumerate(projects["ideal_group_size"])
            for group_id in group_ids[project_id]
        ),
        name="lower_bound_group_size_surplus",
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
        name="lower_bound_group_size_deficit",
    )

    max_num_groups = max(projects["max#groups"])

    unique_group_identifiers = {
        (project_id, group_id): project_id + group_id / max_num_groups
        for project_id, group_id in project_group_pairs
    }

    num_projects = len(projects)

    model.addConstrs(
        (
            mutual_unrealized[first, second] * num_projects
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
        name="only_reward_materialized_pairs_1",
    )

    model.addConstrs(
        (
            mutual_unrealized[first, second] * num_projects
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
        name="only_reward_materialized_pairs_2",
    )

    model.update()

    return model, variables, linear_expressions
