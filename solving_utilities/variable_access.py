import dataclasses

import gurobipy

from modeling.model_components import Variables


@dataclasses.dataclass
class GurobiVariableAccess:
    assign_students: tuple[gurobipy.Var, ...]
    group_size_surplus: tuple[gurobipy.Var, ...]
    group_size_deficit: tuple[gurobipy.Var, ...]

    @classmethod
    def get(cls, variables: Variables):
        return cls(
            assign_students=tuple(variables.assign_students.values()),
            group_size_surplus=tuple(variables.group_size_surplus.values()),
            group_size_deficit=tuple(variables.group_size_deficit.values()),
        )
