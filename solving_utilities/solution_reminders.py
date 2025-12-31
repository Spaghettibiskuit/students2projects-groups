"""A dataclass which stores the variable values and the objective value of a solution."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class SolutionReminderBase:
    variable_values: tuple[int | float, ...]
    objective_value: int


@dataclasses.dataclass(frozen=True)
class SolutionReminderBranching(SolutionReminderBase):
    assign_students_var_values: tuple[int | float, ...]


@dataclasses.dataclass(frozen=True)
class SolutionReminderAssignmentFixing(SolutionReminderBase):
    assign_students_var_values: tuple[int | float, ...]
    group_size_surplus_var_values: tuple[int | float, ...]
    group_size_deficit_var_values: tuple[int | float, ...]
