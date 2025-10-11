"""A class that controls all components which enable VNS with linear branching."""

from configurator import Configurator


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
        self.config = Configurator(
            number_of_students=number_of_students,
            number_of_projects=number_of_projects,
            instance_index=instance_index,
            reward_mutual_pair=reward_mutual_pair,
            penalty_unassigned=penalty_unassigned,
        )
