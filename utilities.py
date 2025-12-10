import enum
from typing import Iterable

import gurobipy


class Subfolders(enum.StrEnum):
    GUROBI = "gurobi"
    LOCAL_BRANCHING = "local_branching"
    VARIABLE_FIXING = "variable_fixing"


class Stations(enum.StrEnum):
    INITIAL_OPTIMIZATION = "initial_optimization"
    VND = "vnd"
    SHAKE = "shake"


def var_values(variables: Iterable[gurobipy.Var]) -> tuple[float, ...]:
    return tuple(var.X for var in variables)


def gurobi_round(value: float) -> int:
    rounded = round(value)
    if abs(value - rounded) > 1e-6:
        raise ValueError("Unexpectedly large deviation from closest integer.")
    return rounded
