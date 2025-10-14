"""Contains dataclasses for accessing a model."""

from enum import Enum, unique


@unique
class VariableNames(Enum):
    ASSIGN_STUDENTS = "assign_students"
    ESTABLISH_GROUPS = "establish_groups"
    MUTUAL_UNREALIZED = "mutual_unrealized"
    UNASSIGNED_STUDENTS = "unassigned_students"
    GROUP_SIZE_SURPLUS = "group_size_surplus"
    GROUP_SIZE_DEFICIT = "group_size_deficit"


@unique
class InitialConstraintNames(Enum):
    ONE_ASSIGNMENT_OR_UNASSIGNED = "one_assignment_or_unassigned"
    OPEN_GROUPS_CONSECUTIVELY = "open_groups_consecutively"
    MIN_GROUP_SIZE_IF_OPEN = "min_group_size_if_open"
    MAX_GROUP_SIZE_IF_OPEN = "max_group_size_if_open"
    LOWER_BOUND_GROUP_SIZE_SURPLUS = "lower_bound_group_size_surplus"
    LOWER_BOUND_GROUP_SIZE_DEFICIT = "lower_bound_group_size_deficit"
    ONLY_REWARD_MATERIALIZED_PAIRS_1 = "only_reward_materialized_pairs_1"
    ONLY_REWARD_MATERIALIZED_PAIRS_2 = "only_reward_materialized_pairs_2"
