import json
import random

import utilities
from create_instance import create_instance

if __name__ == "__main__":
    random.seed = 0
    for num_projects, num_students in [(10, 100), (20, 200), (30, 300), (40, 400), (50, 500)]:
        for i in range(10):
            projects_path, students_path = utilities.build_paths(
                num_projects, num_students, instance_index=i
            )
            if projects_path.exists() or students_path.exists():
                raise ValueError()
            projects_info, students_info = create_instance(num_projects, num_students)
            projects_info.to_csv(projects_path, index=False)
            students_info["project_prefs"] = students_info["project_prefs"].apply(json.dumps)  # type: ignore
            students_info.to_csv(students_path, index=False)
