def get_specific_rewards(general_reward_functions, depth_in_route, reward_mapping):
    """Convert a nested list to 1D tensor. Convert the general reward to specific according to some mapping.
    In input 0 represent intermediate, 1 represents buildingblock and -1 represents a dead end.
    """

    specific_rewards = []
    for route, depths in zip(general_reward_functions, depth_in_route):
        route_rewards = []
        # print('General reward route', route)
        # print(depths)
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
