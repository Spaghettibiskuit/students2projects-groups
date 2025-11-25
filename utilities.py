from typing import Iterable

import gurobipy as gp


def var_values(variables: Iterable[gp.Var]) -> tuple[float, ...]:
    return tuple(var.X for var in variables)
