"""Central place where linear expressions for the SPAwGBP are created."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import gurobipy as gp

from names import VariableNames

if TYPE_CHECKING:
    from configuration import Configuration
    from constrained_model import ConstrainedModel
    from derived_modeling_data import DerivedModelingData


@dataclass(frozen=True)
class Variables:
    assign_students: gp.tupledict[tuple[Any, ...], gp.Var]
    establish_groups: gp.tupledict[tuple[Any, ...], gp.Var]
    mutual_unrealized: gp.tupledict[tuple[Any, ...], gp.Var]
    unassigned_students: gp.tupledict[tuple[Any, ...], gp.Var] | gp.tupledict[int, gp.Var]
    group_size_surplus: gp.tupledict[tuple[Any, ...], gp.Var]
    group_size_deficit: gp.tupledict[tuple[Any, ...], gp.Var]

    @classmethod
    def get(
        cls,
        derived: DerivedModelingData,
        constrained_model: ConstrainedModel,
    ):
        gurobi_model = constrained_model.model

        assign_students_name = VariableNames.ASSIGN_STUDENTS.value
        assign_students_vars = tuple(
            cast(
                gp.Var,
                gurobi_model.getVarByName(
                    f"{assign_students_name}[{project_id},{group_id},{student_id}]"
                ),
            )
            for project_id, group_id, student_id in derived.project_group_student_triples
        )
        assign_students = gp.tupledict(
            zip(derived.project_group_student_triples, assign_students_vars)
        )

        establish_groups_name = VariableNames.ESTABLISH_GROUPS.value
        establish_groups_vars = tuple(
            cast(
                gp.Var,
                gurobi_model.getVarByName(f"{establish_groups_name}[{project_id},{group_id}]"),
            )
            for project_id, group_id in derived.project_group_pairs
        )

        establish_groups = gp.tupledict(zip(derived.project_group_pairs, establish_groups_vars))

        mutual_unrealized_name = VariableNames.MUTUAL_UNREALIZED.value
        mutual_unrealized_vars = tuple(
            cast(
                gp.Var,
                gurobi_model.getVarByName(f"{mutual_unrealized_name}[{first_id},{second_id}]"),
            )
            for first_id, second_id in derived.mutual_pairs
        )
        mutual_unrealized = gp.tupledict(zip(derived.mutual_pairs, mutual_unrealized_vars))

        unassigned_students_name = VariableNames.UNASSIGNED_STUDENTS.value
        unassigned_students_vars = tuple(
            cast(
                gp.Var,
                gurobi_model.getVarByName(f"{unassigned_students_name}[{student_id}]"),
            )
            for student_id in derived.student_ids
        )
        unassigned_students = gp.tupledict(zip(derived.student_ids, unassigned_students_vars))

        group_size_surplus_name = VariableNames.GROUP_SIZE_SURPLUS.value
        group_size_surplus_vars = tuple(
            cast(
                gp.Var,
                gurobi_model.getVarByName(f"{group_size_surplus_name}[{project_id},{group_id}]"),
            )
            for project_id, group_id in derived.project_group_pairs
        )
        group_size_surplus = gp.tupledict(
            zip(derived.project_group_pairs, group_size_surplus_vars)
        )

        group_size_deficit_name = VariableNames.GROUP_SIZE_DEFICIT.value
        group_size_deficit_vars = tuple(
            cast(
                gp.Var,
                gurobi_model.getVarByName(f"{group_size_deficit_name}[{project_id},{group_id}]"),
            )
            for project_id, group_id in derived.project_group_pairs
        )
        group_size_deficit = gp.tupledict(
            zip(derived.project_group_pairs, group_size_deficit_vars)
        )

        return cls(
            assign_students=assign_students,
            establish_groups=establish_groups,
            mutual_unrealized=mutual_unrealized,
            unassigned_students=unassigned_students,
            group_size_surplus=group_size_surplus,
            group_size_deficit=group_size_deficit,
        )


@dataclass(frozen=True)
class LinExpressions:
    sum_realized_project_preferences: gp.LinExpr
    sum_reward_mutual: gp.LinExpr
    sum_penalties_unassigned: gp.LinExpr
    sum_penalties_surplus_groups: gp.LinExpr
    sum_penalties_group_size: gp.LinExpr

    @classmethod
    def get(cls, config: Configuration, derived: DerivedModelingData, variables: Variables):

        project_preferences = derived.project_preferences
        assign_students = variables.assign_students

        sum_realized_project_preferences = gp.quicksum(
            project_preferences[student_id, project_id]
            * assign_students[project_id, group_id, student_id]
            for project_id, group_id, student_id in derived.project_group_student_triples
        )

        mutual_unrealized = variables.mutual_unrealized
        sum_reward_mutual = config.reward_mutual_pair * gp.quicksum(
            1 - mutual_unrealized[*mutual_pair] for mutual_pair in derived.mutual_pairs
        )

        sum_penalties_unassigned = config.penalty_unassigned * gp.quicksum(
            variables.unassigned_students.values()
        )

        establish_groups = variables.establish_groups
        group_ids = derived.group_ids
        sum_penalties_surplus_groups = gp.quicksum(
            penalty_surplus_group * establish_groups[project_id, group_id]
            for project_id, penalty_surplus_group, num_groups_desired in zip(
                derived.project_ids,
                config.projects_info["pen_groups"],
                config.projects_info["desired#groups"],
            )
            for group_id in group_ids[project_id]
            if group_id >= num_groups_desired
        )

        group_size_surplus = variables.group_size_surplus
        group_size_deficit = variables.group_size_deficit

        sum_penalties_group_size = gp.quicksum(
            penalty_size_deviation
            * (group_size_surplus[project_id, group_id] + group_size_deficit[project_id, group_id])
            for project_id, penalty_size_deviation in enumerate(config.projects_info["pen_size"])
            for group_id in group_ids[project_id]
        )

        return cls(
            sum_realized_project_preferences=sum_realized_project_preferences,
            sum_reward_mutual=sum_reward_mutual,
            sum_penalties_unassigned=sum_penalties_unassigned,
            sum_penalties_surplus_groups=sum_penalties_surplus_groups,
            sum_penalties_group_size=sum_penalties_group_size,
        )
