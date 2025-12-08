import enum
from typing import Iterable

import gurobipy


class BenchmarkingSubfolders(enum.StrEnum):
    GUROBI = "gurobi"
    LOCAL_BRANCHING = "local_branching"
    VARIABLE_FIXING = "variable_fixing"


def var_values(variables: Iterable[gurobipy.Var]) -> tuple[float, ...]:
    return tuple(var.X for var in variables)
