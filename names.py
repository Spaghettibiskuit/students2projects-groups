"""Contains dataclasses for accessing a model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableNames:
    assign_students: str = "assign_students"
    establish_groups: str = "establish_groups"
    mutual_unrealized: str = "mutual_unrealized"
    unassigned_students: str = "unassigned_students"
    group_size_surplus: str = "group_size_surplus"
    group_size_deficit: str = "group_size_deficit"


@dataclass(frozen=True)
class InitialConstraintNames:
    one_assignment_or_unassigned: str = "one_assignment_or_unassigned"
    open_groups_consecutively: str = "open_groups_consecutively"
    min_group_size_if_open: str = "min_group_size_if_open"
    max_group_size_if_open: str = "max_group_size_if_open"
    lower_bound_group_size_surplus: str = "lower_bound_group_size_surplus"
    lower_bound_group_size_deficit: str = "lower_bound_group_size_deficit"
    only_reward_materialized_pairs_1: str = "only_reward_materialized_pairs_1"
    only_reward_materialized_pairs_2: str = "only_reward_materialized_pairs_2"
