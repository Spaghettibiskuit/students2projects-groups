"""A class that controls all components which enable VNS with linear branching."""

from configurator import Configuration
from constrained_model import ConstrainedModel
from derived_modeling_data import get_derived_modeling_data
from names import InitialConstraintNames, VariableNames


class VariableNeighborhoodSearch:
    """Controls all components which enable VNS with linear branching."""

    def __init__(
        self,
        number_of_students: int,
        number_of_projects: int,
        instance_index: int,
        reward_mutual_pair: int,
        penalty_unassigned: int,
    ):
        self.config = Configuration(
            number_of_students=number_of_students,
            number_of_projects=number_of_projects,
            instance_index=instance_index,
            reward_mutual_pair=reward_mutual_pair,
            penalty_unassigned=penalty_unassigned,
        )
        self.derived = get_derived_modeling_data(self.config)
        self.var_names = VariableNames()
        self.initial_constr_names = InitialConstraintNames()
        self.initial_model = ConstrainedModel.initial_model(
            config=self.config,
            derived=self.derived,
            var_names=self.var_names,
            constr_names=self.initial_constr_names,
        )
        self.best_model = None

    def solve_exactly(self) -> ConstrainedModel:
        self.initial_model.model.optimize()
        self.best_model = self.initial_model
        return self.best_model
