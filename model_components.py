"""Central place where linear expressions for the SPAwGBP are created."""

from dataclasses import dataclass

import gurobipy


@dataclass(frozen=True)
class Variables:
    assign_students: gurobipy.tupledict[tuple[int, int, int], gurobipy.Var]
    establish_groups: gurobipy.tupledict[tuple[int, int], gurobipy.Var]
    mutual_unrealized: gurobipy.tupledict[tuple[int, int], gurobipy.Var]
    unassigned_students: gurobipy.tupledict[int, gurobipy.Var]
    group_size_surplus: gurobipy.tupledict[tuple[int, int], gurobipy.Var]
    group_size_deficit: gurobipy.tupledict[tuple[int, int], gurobipy.Var]


@dataclass(frozen=True)
class LinExpressions:
    sum_realized_project_preferences: gurobipy.LinExpr
    sum_reward_mutual: gurobipy.LinExpr
    sum_penalties_unassigned: gurobipy.LinExpr
    sum_penalties_surplus_groups: gurobipy.LinExpr
    sum_penalties_group_size: gurobipy.LinExpr


@dataclass(frozen=True)
class InitialConstraints:
    one_assignment_or_unassigned: gurobipy.tupledict[int, gurobipy.Constr]
    open_groups_consecutively: gurobipy.tupledict[int, gurobipy.Constr]
    min_group_size_if_open: gurobipy.tupledict[tuple[int, int, int], gurobipy.Constr]
    max_group_size_if_open: gurobipy.tupledict[tuple[int, int, int], gurobipy.Constr]
    lower_bound_group_size_surplus: gurobipy.tupledict[tuple[int, int, int], gurobipy.Constr]
    lower_bound_group_size_deficit: gurobipy.tupledict[tuple[int, int, int, int], gurobipy.Constr]
    only_reward_materialized_pairs_1: gurobipy.tupledict[tuple[int, int], gurobipy.Constr]
    only_reward_materialized_pairs_2: gurobipy.tupledict[tuple[int, int], gurobipy.Constr]
