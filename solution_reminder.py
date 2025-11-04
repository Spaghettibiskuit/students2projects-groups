"""A dataclass which is stores the variable values and the objective value of a solution."""

from dataclasses import dataclass


@dataclass
class SolutionReminder:
    """Stores the variable values and the objective value of a solution."""

    variable_values: tuple[int | float, ...]
    objective_value: int | float
    assign_students_vars_values: tuple[int | float, ...]
    establish_groups_vars_values: tuple[int | float, ...]
