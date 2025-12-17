import dataclasses

import pandas

from instance_creation.projects_info import random_projects_df
from instance_creation.students_info import random_students_df


@dataclasses.dataclass
class ProjectsParams:
    min_desired_num_groups: int = 2
    max_desired_num_groups: int = 4
    min_manageable_surplus_groups: int = 1
    max_manageable_surplus_groups: int = 3
    min_ideal_group_size: int = 2
    max_ideal_group_size: int = 4
    min_tolerable_group_size_deficit: int = 1
    max_tolerable_group_size_deficit: int = 3
    min_tolerable_group_size_surplus: int = 1
    max_tolerable_group_size_surplus: int = 3
    min_pen_num_groups: int = 1
    max_pen_num_groups: int = 3
    min_pen_group_size: int = 1
    max_pen_group_size: int = 3


@dataclasses.dataclass
class StudentsParams:
    min_num_partner_preferences: int = 1
    max_num_partner_preferences: int = 5
    percentage_reciprocity: float = 0.7
    percentage_peer_influenced_project_preferences: float = 0.7
    min_project_preference: int = 0
    max_project_preference: int = 3


def create_instance(
    num_projects: int, num_students: int
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    projects_info = random_projects_df(num_projects, **dataclasses.asdict(ProjectsParams()))
    students_info = random_students_df(
        num_projects, num_students, **dataclasses.asdict(StudentsParams())
    )
    return projects_info, students_info
