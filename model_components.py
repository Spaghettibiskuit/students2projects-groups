"""Central place where linear expressions for the SPAwGBP are created."""

from dataclasses import dataclass

import gurobipy as gp


@dataclass(frozen=True)
class Variables:
    assign_students: gp.tupledict[tuple[int, int, int], gp.Var]
    establish_groups: gp.tupledict[tuple[int, int], gp.Var]
    mutual_unrealized: gp.tupledict[tuple[int, int], gp.Var]
    unassigned_students: gp.tupledict[int, gp.Var]
    group_size_surplus: gp.tupledict[tuple[int, int], gp.Var]
    group_size_deficit: gp.tupledict[tuple[int, int], gp.Var]


@dataclass(frozen=True)
class LinExpressions:
    sum_realized_project_preferences: gp.LinExpr
    sum_reward_mutual: gp.LinExpr
    sum_penalties_unassigned: gp.LinExpr
    sum_penalties_surplus_groups: gp.LinExpr
    sum_penalties_group_size: gp.LinExpr


@dataclass(frozen=True)
class InitialConstraints:
    one_assignment_or_unassigned: gp.tupledict[int, gp.Constr]
    open_groups_consecutively: gp.tupledict[int, gp.Constr]
    min_group_size_if_open: gp.tupledict[tuple[int, int, int], gp.Constr]
    max_group_size_if_open: gp.tupledict[tuple[int, int, int], gp.Constr]
    lower_bound_group_size_surplus: gp.tupledict[tuple[int, int, int], gp.Constr]
    lower_bound_group_size_deficit: gp.tupledict[tuple[int, int, int, int], gp.Constr]
    only_reward_materialized_pairs_1: gp.tupledict[tuple[int, int], gp.Constr]
    only_reward_materialized_pairs_2: gp.tupledict[tuple[int, int], gp.Constr]
