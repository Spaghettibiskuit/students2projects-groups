"""A dataclass that contains all information on the instance all parameters on how to run VNS."""

from dataclasses import dataclass

import pandas as pd

from load_instance import load_instance


@dataclass(frozen=True)
class Configuration:
    """Contains all information on the instance and all parameters on how to run VNS."""

    number_of_projects: int
    number_of_students: int
    instance_index: int
    reward_mutual_pair: int
    penalty_unassigned: int

    projects_info: pd.DataFrame
    students_info: pd.DataFrame

    @classmethod
    def get(
        cls,
        number_of_projects: int,
        number_of_students: int,
        instance_index: int,
        reward_mutual_pair: int,
        penalty_unassigned: int,
    ):
        projects_info, students_info = load_instance(
            number_of_projects, number_of_students, instance_index
        )
        return cls(
            number_of_projects,
            number_of_students,
            instance_index,
            reward_mutual_pair,
            penalty_unassigned,
            projects_info,
            students_info,
        )
