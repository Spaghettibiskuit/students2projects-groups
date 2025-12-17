"""Contains functions to generate a dataframe on random students."""

import random

import pandas


def _random_partner_preferences(
    num_students: int,
    percentage_reciprocity: float,
    min_num_partner_prefs: int,
    max_num_partner_prefs: int,
) -> list[list[int]]:
    students_partner_prefs: list[list[int]] = []
    student_ids = list(range(num_students))
    chosen_by: dict[int, list[int]] = {student_id: [] for student_id in student_ids}

    for student_id in student_ids:
        all_other_ids = student_ids[:student_id] + student_ids[student_id + 1 :]
        num_partner_prefs = random.randint(min_num_partner_prefs, max_num_partner_prefs)

        if not (ids_that_chose_student := chosen_by[student_id]):
            student_partner_prefs = random.sample(all_other_ids, num_partner_prefs)
        else:
            applicable_for_reciprocity = random.sample(
                ids_that_chose_student,
                min(len(ids_that_chose_student), num_partner_prefs),
            )
            reciprocal_prefs = [
                other_student_id
                for other_student_id in applicable_for_reciprocity
                if random.random() <= percentage_reciprocity
            ]
            num_missing_prefs = num_partner_prefs - len(reciprocal_prefs)

            if not num_missing_prefs:
                student_partner_prefs = reciprocal_prefs
            else:
                left_options = [
                    student_id
                    for student_id in all_other_ids
                    if student_id not in reciprocal_prefs
                ]
                random_prefs = random.sample(left_options, num_missing_prefs)
                student_partner_prefs = reciprocal_prefs + random_prefs

        students_partner_prefs.append(student_partner_prefs)

        for partner_preference in student_partner_prefs:
            chosen_by[partner_preference].append(student_id)

    return students_partner_prefs


def _peer_project_preferences(
    desired_partners: list[int], project_preferences_so_far: list[tuple[int, ...]]
) -> tuple[list[int], ...] | None:
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
    return tuple(list(preferences_for_project) for preferences_for_project in zip(*relevant_prefs))


def peer_influenced_project_preference(preferences_for_project: list[int]) -> int:
    return max(preferences_for_project) if random.random() <= 0.5 else min(preferences_for_project)


def _random_project_preferences(
    num_projects: int,
    everyones_partner_preferences: list[list[int]],
    percentage_peer_influenced: float,
    min_pref: int,
    max_pref: int,
) -> list[tuple[int, ...]]:
    project_preferences_per_student: list[tuple[int, ...]] = []
    for partner_preferences in everyones_partner_preferences:
        peer_preferences_per_project = _peer_project_preferences(
            partner_preferences, project_preferences_per_student
        )
        if peer_preferences_per_project is None:
            student_project_preferences = tuple(
                random.randint(min_pref, max_pref) for _ in range(num_projects)
            )
        else:
            student_project_preferences = tuple(
                (
                    peer_influenced_project_preference(preferences_for_project)
                    if random.random() <= percentage_peer_influenced
                    else random.randint(min_pref, max_pref)
                )
                for preferences_for_project in peer_preferences_per_project
            )

        project_preferences_per_student.append(student_project_preferences)

    return project_preferences_per_student


def random_students_df(
    num_projects: int,
    num_students: int,
    min_num_partner_preferences: int,
    max_num_partner_preferences: int,
    percentage_reciprocity: float,
    percentage_peer_influenced_project_preferences: float,
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
    partner_preferences = _random_partner_preferences(
        num_students,
        percentage_reciprocity,
        min_num_partner_preferences,
        max_num_partner_preferences,
    )
    project_preferences = _random_project_preferences(
        num_projects,
        partner_preferences,
        percentage_peer_influenced_project_preferences,
        min_project_preference,
        max_project_preference,
    )

    return pandas.DataFrame(
        {
            "fav_partners": partner_preferences,
            "project_prefs": project_preferences,
        }
    )
