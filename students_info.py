"""Contains functions to generate a dataframe on random students."""

import random

import pandas


def random_partner_preferences(
    num_students: int,
    percentage_reciprocity: float,
    num_partner_preferences: int,
) -> list[list[int]]:
    """Returns partner preferences for all students.

    Args:
        num_students: The number of students in the problem.
        percentage_reciprocity: Roughly the probability that a student
            specifies another student as a partner preference if that
            student specified him/her as a partner preference before.
        num_partner_preferences: The number of partner preferences
            each student specifies with the ID of the students he/she
            wants to  work together with the most.
    """
    students_partner_preferences: list[list[int]] = []
    student_ids = list(range(num_students))
    chosen_by: dict[int, list[int]] = {student_id: [] for student_id in student_ids}

    for student_id in range(num_students):
        all_other_student_ids = student_ids[:student_id] + student_ids[student_id + 1 :]

        if ids_that_chose_student := chosen_by[student_id]:
            applicable_for_reciprocity = random.sample(
                ids_that_chose_student,
                min(len(ids_that_chose_student), num_partner_preferences),
            )
            reciprocal_preferences = [
                other_students_id
                for other_students_id in applicable_for_reciprocity
                if random.random() <= percentage_reciprocity
            ]
            num_missing_preferences = num_partner_preferences - len(reciprocal_preferences)

            if num_missing_preferences > 0:
                left_options = [
                    student_id
                    for student_id in all_other_student_ids
                    if student_id not in reciprocal_preferences
                ]
                student_partner_preferences = reciprocal_preferences + random.sample(
                    left_options, num_missing_preferences
                )
            else:
                student_partner_preferences = reciprocal_preferences

        else:
            student_partner_preferences = random.sample(
                all_other_student_ids, num_partner_preferences
            )

        students_partner_preferences.append(student_partner_preferences)

        for partner_preference in student_partner_preferences:
            chosen_by[partner_preference].append(student_id)

    return students_partner_preferences


def average_project_preferences(
    desired_partners: list[int], project_preferences_so_far: list[tuple[int, ...]]
) -> tuple[float, ...] | None:
    """Returns the average available project preferences among a student's partner preferences.

    Available means that the partner preference i.e., one of the students the student in
    question wants to work with the most has specified his/her project preferences before.

    Args:
        desired_partners: The students the student in question wants to work with the most.
        project_preferences_so_far: The project preferences made so far. The index position is
            the ID of the student who specified the project preferences.
    """

    desired_partners_with_prefs = [
        desired_partner
        for desired_partner in desired_partners
        if desired_partner < len(project_preferences_so_far)
    ]
    if not desired_partners_with_prefs:
        return None

    relevant_prefs = [
        project_preferences_so_far[desired_partner]
        for desired_partner in desired_partners_with_prefs
    ]
    return tuple(
        sum(prefs_for_project) / len(prefs_for_project)
        for prefs_for_project in zip(*relevant_prefs)
    )


def random_project_preferences(
    num_projects: int,
    everyones_partner_preferences: list[list[int]],
    overlap: float,
    min_pref: int,
    max_pref: int,
) -> list[tuple[int, ...]]:
    """Returns project preference values for all students in the problem.

    Args:
        num_projects: The number of projects in the problem.
        everyones_partner_preferences: The IDs of the students a student wants
            to work with the most for every student in the problem
        overlap: To what degree the
            student's preference value for a specific project is the
            average preference for that project among those that are
            partner preferences and already have specified their
            project preferences.
        min_pref: The lowest possible project preference.
        max_pref: The highest possible project preference.
    """
    project_preferences_so_far: list[tuple[int, ...]] = []
    for partner_preferences in everyones_partner_preferences:
        average_preferences = average_project_preferences(
            partner_preferences, project_preferences_so_far
        )
        if average_preferences is None:
            student_project_preferences = tuple(
                round(random.uniform(min_pref - 0.5, max_pref + 0.5)) for _ in range(num_projects)
            )
        else:
            student_project_preferences = tuple(
                round(
                    overlap * average_preference
                    + ((1 - overlap) * random.uniform(min_pref - 0.5, max_pref + 0.5))
                )
                for average_preference in average_preferences
            )

        project_preferences_so_far.append(student_project_preferences)

    return project_preferences_so_far


def random_students_df(
    num_projects: int,
    num_students: int,
    num_partner_preferences: int,
    percentage_reciprocity: float,
    percentage_project_preference_overlap: float,
    min_project_preference: int,
    max_project_preference: int,
) -> pandas.DataFrame:
    """Returns random students with partner and project preferences.

    Args:
        num_projects: The number of projects in the problem instance.
        num_students: The number of students in the problem instance.
        num_partner_preferences: The number of partner preferences
            each student specifies with the ID of the students he/she
            wants to  work together with the most.
        percentage_reciprocity: Roughly the probability that a student
            specifies another student as a partner preference if that
            student specified him/her as a partner preference before.
        percentage_project_preference_overlap: To what degree the
            student's preference value for a specific project is the
            average preference for that project among those that are
            partner preferences and already have specified their
            project preferences.
        min_project_preference: The lowest possible project preference.
        max_project_preference: The highest possible project preference.

    Returns:
        The project preferences for all projects and the partner preferences
        i.e., the students a student wants to work with the most, for all
        students in the problem instance. Project preferences and partner preferences
        are influenced by each other to a specifiable degree. Otherwise values
        are random within bounds set by the arguments. THE INDEX POSITION IN THE
        DATAFRAME LATER BECOMES THE STUDENT'S ID.
    """
    partner_preferences = random_partner_preferences(
        num_students, percentage_reciprocity, num_partner_preferences
    )
    project_preferences = random_project_preferences(
        num_projects,
        partner_preferences,
        percentage_project_preference_overlap,
        min_project_preference,
        max_project_preference,
    )

    return pandas.DataFrame(
        {
            "fav_partners": partner_preferences,
            "project_prefs": project_preferences,
        }
    )
