"""A dataclass which is stores the variable values and the objective value of a solution."""

from dataclasses import dataclass


@dataclass
class SolutionReminderBranching:
    """Stores the variable values and the objective value of a solution."""

    variable_values: tuple[int | float, ...]
    objective_value: int | float
    assign_students_var_values: tuple[int | float, ...]
    establish_groups_var_values: tuple[int | float, ...]


@dataclass
class SolutionReminderDiving:
    """Stores the variable values and the objective of a solution."""

    variable_values: tuple[int | float, ...]
    objective_value: int | float
    assign_students_var_values: tuple[int | float, ...]
    mutual_unrealized_var_values: tuple[int | float, ...]
    unassigned_students_var_values: tuple[int | float, ...]
