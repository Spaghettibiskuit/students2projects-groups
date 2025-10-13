"""A dataclass that contains all information on the instance all parameters on how to run VNS."""

from dataclasses import dataclass, field

import pandas as pd

from load_instance import load_instance


@dataclass(frozen=True)
class Configuration:
    """Contains all information on the instance and all parameters on how to run VNS."""

    number_of_students: int
    number_of_projects: int
    instance_index: int
    reward_mutual_pair: int
    penalty_unassigned: int

    projects_info: pd.DataFrame = field(init=False)
    students_info: pd.DataFrame = field(init=False)

    def __post_init__(self):

        projects_info, students_info = load_instance(
            self.number_of_projects, self.number_of_students, self.instance_index
        )

        object.__setattr__(self, "projects_info", projects_info)
        object.__setattr__(self, "students_info", students_info)
