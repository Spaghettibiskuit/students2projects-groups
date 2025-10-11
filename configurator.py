"""A dataclass that contains all information on the instance all parameters on how to run VNS."""

from dataclasses import dataclass, field

import pandas as pd

from load_instance import load_instance


@dataclass(frozen=True)
class Configurator:
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

        # self.project_ids = range(len(self.projects_info))
        # self.student_ids = range(len(self.students_info))
        # self.group_ids = {
        #     project_id: range(max_num_groups)
        #     for project_id, max_num_groups in enumerate(self.projects_info["max#groups"])
        # }
        # self.project_group_pairs = [
        #     (project_id, group_id)
        #     for project_id, project_group_ids in self.group_ids.items()
        #     for group_id in project_group_ids
        # ]
        # self.project_group_student_triples = [
        #     (project_id, group_id, student_id)
        #     for project_id, group_id in self.project_group_pairs
        #     for student_id in self.student_ids
        # ]
        # self.mutual_pairs = {
        #     (student_id, partner_id)
        #     for student_id, partner_ids in enumerate(self.students_info["fav_partners"])
        #     for partner_id in partner_ids
        #     if partner_id > student_id
        #     and student_id in self.students_info["fav_partners"][partner_id]
        # }
        # self.project_preferences = {
        #     (student_id, project_id): preference_value
        #     for student_id, preference_values in enumerate(self.students_info["project_prefs"])
        #     for project_id, preference_value in enumerate(preference_values)
        # }
