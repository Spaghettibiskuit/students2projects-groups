"""Contains function to generate a dataframe on random projects."""

import random as rd

import pandas as pd


def random_projects_df(
    num_projects: int,
    min_desired_num_groups: int,
    max_desired_num_groups: int,
    min_manageable_surplus_groups: int,
    max_manageable_surplus_groups: int,
    min_ideal_group_size: int,
    max_ideal_group_size: int,
    min_tolerable_group_size_deficit: int,
    max_tolerable_group_size_deficit: int,
    min_tolerable_group_size_surplus: int,
    max_tolerable_group_size_surplus: int,
    min_pen_num_groups: int,
    max_pen_num_groups: int,
    min_pen_group_size: int,
    max_pen_group_size: int,
) -> pd.DataFrame:
    """Returns random projects with names and information on them.

    Args:
        num_projects: The number of projects in the problem.
        min_desired_num_groups: The minimum number of groups a project
            can want to supervise.
        max_desired_num_groups: The maximum number of groups a project
            can want to supervise.
        min_manageable_surplus_groups: The minimum number of groups on
            top of those it wants to supervise any group is willing to
            supervise.
        max_manageable_surplus_groups: The maximum number of groups on
            top of those it wants to supervise any group is willing to
            supervise.
        min_ideal_group_size: The minimum group size any project deems
            ideal.
        max_ideal_group_size: The maximum group size any project deems
            ideal.
        min_tolerable_group_size_deficit: The minimum negative deviation
            from the ideal group size any project can accept.
        max_tolerable_group_size_deficit: The maximum negative deviation
            from the ideal group size any project can accept.
        min_tolerable_group_size_surplus: The minimum positive deviation
            from the ideal group size any project can accept.
        max_tolerable_group_size_surplus: The maximum positive deviation
            from the ideal group size any project can accept.
        min_pen_num_groups: The minimum penalty any project attributes
            to each group exceeding the number of groups it wants to
            supervise.
        max_pen_num_groups: The maximum penalty any project attributes
            to each group exceeding the number of groups it wants to
            supervise.
        min_pen_group_size: The minimum coefficient with which deviation
            from ideal_group_size is penalized by any project.
        max_pen_group_size: The maximum coefficient with which deviation
            from ideal_group_size is penalized by any project.

        Returns:
            Project names with each project's guidelines, whishes and penalties
            regarding the number of groups and group sizes. All is generated
            randomly within the bounds set by the arguments. THE INDEX
            POSITION IN THE DATAFRAME LATER BECOMES THE PROJECT'S ID.
    """

    desired_nums_groups = [
        rd.randint(min_desired_num_groups, max_desired_num_groups) for _ in range(num_projects)
    ]
    max_nums_groups = [
        desired_num_groups
        + rd.randint(min_manageable_surplus_groups, max_manageable_surplus_groups)
        for desired_num_groups in desired_nums_groups
    ]
    ideal_group_sizes = [
        rd.randint(min_ideal_group_size, max_ideal_group_size) for _ in range(num_projects)
    ]
    min_group_sizes = [
        max(
            1,
            ideal_group_size
            - rd.randint(
                min_tolerable_group_size_deficit,
                max_tolerable_group_size_deficit,
            ),
        )
        for ideal_group_size in ideal_group_sizes
    ]
    max_group_sizes = [
        ideal_group_size
        + rd.randint(min_tolerable_group_size_surplus, max_tolerable_group_size_surplus)
        for ideal_group_size in ideal_group_sizes
    ]
    pen_num_groups = [
        rd.randint(min_pen_num_groups, max_pen_num_groups) for _ in range(num_projects)
    ]
    pen_group_size = [
        rd.randint(min_pen_group_size, max_pen_group_size) for _ in range(num_projects)
    ]

    data_projects = {
        "desired#groups": desired_nums_groups,
        "max#groups": max_nums_groups,
        "ideal_group_size": ideal_group_sizes,
        "min_group_size": min_group_sizes,
        "max_group_size": max_group_sizes,
        "pen_groups": pen_num_groups,
        "pen_size": pen_group_size,
    }

    return pd.DataFrame(data_projects)
