def get_specific_rewards(general_reward_functions, depth_in_route, reward_mapping):
    """Convert general {-1, 0, 1} rewards to scaled specific rewards.

    Each route is a list of steps; each step is a list of per-frontier-node general
    rewards (1=building block, 0=intermediate, -1=dead end). Depth is used to scale
    rewards when scale_with_depth is truthy.

    >>> mapping = {
    ...     "scale_score":      {-1: -2, 0: -2, 1: 0},
    ...     "scale_with_depth": {-1:  2, 0:  1, 1: 2},
    ... }
    >>> # Single route, two steps: bb at depth 0, dead-end at depth 1
    >>> get_specific_rewards([[[1], [-1]]], [[0, 1]], mapping)
    [[0, -8]]
    >>> # Two routes (single step each)
    >>> get_specific_rewards([[[0]], [[1]]], [[2], [3]], mapping)
    [[-6], [0]]
    """

    specific_rewards = []
    for route, depths in zip(general_reward_functions, depth_in_route):
        route_rewards = []
        for states, depth in zip(route, depths):
            states_return = 0
            for state in states:
                states_return += (
                    reward_mapping["scale_score"][state]
                    * reward_mapping["scale_with_depth"][state]
                    * (depth + 1)
                    if reward_mapping["scale_with_depth"][state]
                    else reward_mapping["scale_score"][state]
                )
            route_rewards.append(states_return)
        specific_rewards.append(route_rewards)
    return specific_rewards


def get_reward_mapping(config):
    """Build the reward-mapping dict from the config ``reward`` section.

    >>> cfg = {"reward": {
    ...     "dead_end_reward_factor": -2, "intermediate_reward_factor": -2,
    ...     "building_block_reward_factor": 0,
    ...     "dead_end_scale_with_depth": 2, "intermediate_scale_with_depth": 1,
    ...     "building_block_scale_with_depth": 2,
    ... }}
    >>> m = get_reward_mapping(cfg)
    >>> m["scale_score"][-1]
    -2
    >>> m["scale_with_depth"][1]
    2
    """
    reward_mapping = {
        "scale_score": {
            -1: config["reward"]["dead_end_reward_factor"],
            0: config["reward"]["intermediate_reward_factor"],
            1: config["reward"]["building_block_reward_factor"],
        },
        "scale_with_depth": {
            -1: config["reward"]["dead_end_scale_with_depth"],
            0: config["reward"]["intermediate_scale_with_depth"],
            1: config["reward"]["building_block_scale_with_depth"],
        },
    }
    return reward_mapping


# ---- Example of reward_mapping -----
building_block_reward_factor = 2
dead_end_reward_factor = -2
intermediate_reward_factor = -2

building_block_scale_with_depth = 10
dead_end_scale_with_depth = None
intermediate_scale_with_depth = None

reward_mapping = {
    "scale_score": {
        -1: dead_end_reward_factor,
        0: intermediate_reward_factor,
        1: building_block_reward_factor,
    },
    "scale_with_depth": {
        -1: dead_end_scale_with_depth,
        0: intermediate_scale_with_depth,
        1: building_block_scale_with_depth,
    },
}
