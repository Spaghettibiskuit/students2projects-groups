"""A class that bundles information on how rewards can be achieved."""

from dataclasses import dataclass

from configurator import Configuration


@dataclass(frozen=True)
class RewardPotentials:
    mutual_pairs: frozenset[tuple[int, int]]
    project_preferences: dict[tuple[int, int], int]


def get_reward_potentials(config: Configuration) -> RewardPotentials:
    mutual_pairs = frozenset(
        (student_id, partner_id)
        for student_id, partner_ids in enumerate(config.students_info["fav_partners"])
        for partner_id in partner_ids
        if partner_id > student_id
        and student_id in config.students_info["fav_partners"][partner_id]
    )
    project_preferences = {
        (student_id, project_id): preference_value
        for student_id, preference_values in enumerate(config.students_info["project_prefs"])
        for project_id, preference_value in enumerate(preference_values)
    }
    return RewardPotentials(mutual_pairs, project_preferences)
