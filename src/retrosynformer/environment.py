import copy

from rdkit import Chem

from .utils import utils


class RetroGymEnvironment:
    def __init__(
        self,
        building_blocks,
        action2templates,
        reward_mapping,
        max_depth,
        process_routes=False,
    ):
        self.building_blocks = building_blocks
        self.action2templates = action2templates
        self.reward_mapping = reward_mapping
        self._branch_solved_state = "<BRANCH_SOLVED>"
        self._branch_not_solved_state = "<DEADEND_BRANCH>"
        self.max_depth = max_depth
        self.available_templates = action2templates[
            "smiles"
        ].values  # SMARTS NOT SMILES!
        self.process_routes = process_routes

        self.state = None
        self.all_rewards = None
        self.rewards = None
        self.branching_depths = None  # The depth of the latest branching point, when the branch is done, this depth is the new depth
        self.current_depth = None  # Depth for the current state
        self.route_depth = None
        self.total_reward = None
        self.route_done = False
        self.route_solved = False

    def copy(self):
        # Create a dictionary of all attributes

        # Iterate over all attributes and decide if we need a deep copy
        copied_environment = RetroGymEnvironment(
            self.building_blocks,
            self.action2templates,
            self.reward_mapping,
            self.max_depth,
            self.process_routes,
        )
        for key, value in vars(self).items():
            # Perform deep copy only on mutable objects (for example, lists, dicts)
            if key in [
                "state",
                "rewards",
                "branching_depths",
                "current_depth",
                "route_depth",
                "total_reward",
                "route_done",
                "route_solved",
                "visited_intermediates",
                "visited_intermediates_branch",
                "leafs",
            ]:
                setattr(copied_environment, key, copy.deepcopy(value))
            else:
                setattr(copied_environment, key, value)

        # Create a new instance with the copied attributes
        return copied_environment

    def set_target_compound(self, target_compound, reward_function=None):
        self.state = [[target_compound]]
        self.visited_intermediates, self.visited_intermediates_branch = [], []
        self.reward_function = reward_function
        if self.process_routes:
            self.all_reward_functions = ["reward_general"]
        else:
            self.all_reward_functions = [
                "reward_general",
                "reward_specific",
            ]
        self.all_rewards = {r: [] for r in self.all_reward_functions}
        self.rewards = []
        self.branching_depths = [0]
        self.current_depth = 0
        self.route_depth = 0
        self.total_reward = None
        self.route_solved = self.check_if_building_block(target_compound)
        self.route_done = self.route_solved
        self.number_of_branchings = 0
        self.dead_ends = []
        self.leafs = []

    def _get_reward(self, n):
        """
        Reward function. Return the reward for each state. The reward here is calculated for each reaction step.
        The reward is calculated as the mean of the reward for each reactant, after evaluating for branch-ending criterias such as: building block and maximum depth.
        """
        states = self.state[-n:]
        reward_functions = (
            [self.reward_function]
            if self.reward_function
            else self.all_reward_functions
        )

        for reward_fn in reward_functions:
            if reward_fn == "reward_specific":
                reward = get_specific_rewards(
                    states, self.current_depth, self.reward_mapping
                )
            elif reward_fn == "reward_general":
                reward = get_general_rewards(states, self.current_depth)

            self.all_rewards[reward_fn].append(reward)
        self.rewards = self.all_rewards[reward_functions[0]]

        return self.rewards[-1]

    def get_total_reward(self):
        self.total_reward = sum(utils.flatten_list(self.rewards))
        return self.total_reward

    def _check_if_route_done(self):
        if self.route_done:
            print("Route already done.")
            route_done = True
        elif len(self.state) < 1:
            print("State is empty. Route already done.")
            route_done = True
        else:
            route_done = False
        return route_done

    def step(self, action):
        branch_done = False
        route_done = self._check_if_route_done()
        reactants = []

        if action == 0 or action[0] == "<eos>":
            ordered_reactants = []
            reward = 0
            branch_done = True
            self.rewards.append(0)
            return ordered_reactants
        template = None
        if type(action) == int:
            templates = self.action2template[action]
        elif type(action) == list:
            templates = action
        elif type(action) == str:
            templates = [action]
        product_smiles = self.state[-1][0]
        for template in templates:
            try:
                reactants = utils.apply_template(template, product_smiles)
            except:
                print(
                    "reactants = apply_template(template, product_smiles) command failed"
                )
                print("product smiles: ", product_smiles, "template: ", template)

            if len(reactants) > 0:
                reactants = reactants[0]
                ordered_reactants = self._decide_reactant_order(reactants)
                self._step_impl(ordered_reactants)
                if (
                    len(
                        set(reactants).intersection(
                            set(self.visited_intermediates_branch)
                        )
                    )
                    == 0
                ):
                    return ordered_reactants

        return None

    def step_from_reactants(self, reactants):
        ordered_reactants = self._decide_reactant_order(reactants)
        reward, branch_done = self._step_impl(ordered_reactants)
        return ordered_reactants, reward, branch_done

    def _step_impl(self, reactants):
        """
        Enroll step in environment. Apply reaction template to obtain new state and reward for action.
        """
        # Check if route is done or if we continue
        if self.route_done:
            print("Route already done.")
            return None, True
        if len(self.state) < 1:
            print("State is empty. Route already done.")
            return self.rewards, True

        branch_done = False
        self.visited_intermediates_branch = self.state.pop()
        assert type(self.visited_intermediates_branch) == list
        # adjust the depth and branching parameter
        self.current_depth += 1
        if self.current_depth > self.route_depth:
            self.route_depth = self.current_depth
        n_open_reactants = len(reactants)
        for _ in range(n_open_reactants - 1):
            self.branching_depths.append(self.current_depth)
            self.visited_intermediates.append(
                copy.deepcopy(self.visited_intermediates_branch)
            )

        for reactant in reactants:
            done = self._check_if_branch_done(reactant)
            if done:
                n_open_reactants -= 1

        is_branching = True if n_open_reactants > 1 else False
        if is_branching:
            self.number_of_branchings += 1

        self._get_reward(n=len(reactants))

    def _check_if_branch_done(self, reactant):
        is_building_block = self.check_if_building_block(reactant)
        if is_building_block:
            self.state.append(self._branch_solved_state)
            branch_done = True
            self.leafs.append(reactant)
        elif self.current_depth > self.max_depth:
            self.state.append(self._branch_not_solved_state)
            self.dead_ends.append(reactant)
            branch_done = True
        else:
            self.state.append([reactant])
            branch_done = False

        if branch_done:
            if len(self.visited_intermediates) > 0:
                new_branch_visited_intermediates = self.visited_intermediates.pop()
                self.visited_intermediates_branch = new_branch_visited_intermediates

        return branch_done

    def _check_if_done(self):
        """
        Check if the route is done.
        """
        if len(self.state) < 1:
            self.route_done = True
            self.route_solved = True if len(self.dead_ends) == 0 else False
        elif self.state[-1] == self._branch_not_solved_state:
            self.route_done = True
            self.route_solved = False
        else:
            while len(self.state) > 0 and self.state[-1] == self._branch_solved_state:
                self._proceed_with_next_branch()
                self.route_done = True if len(self.state) < 1 else False
                self.route_solved = (
                    True
                    if (len(self.state) < 1 and len(self.dead_ends) == 0)
                    else False
                )
        if self.route_depth >= self.max_depth:
            self.route_done = True

        self.total_reward = self.get_total_reward()
        return self.route_done, self.route_solved, self.total_reward

    def _proceed_with_next_branch(self):
        """
        Adjusting the current latest branching depth for the next branch.
        """
        while len(self.state) > 0 and (self.state[-1] == self._branch_solved_state):
            self.state.pop()
            self.current_depth = self.branching_depths.pop()

    def get_reactants_from_templates(self, product, template):
        return utils.apply_template(template, product)

    def check_if_building_block(self, smiles):
        inchi_key = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(smiles))
        if inchi_key in self.building_blocks:
            building_block = True
        else:
            building_block = False

        return building_block

    def _decide_reactant_order(self, reactants):
        """Simple sorting function based on alphabetical order."""
        reactants.sort(key=len, reverse=True)

        return reactants


def get_specific_rewards(states, current_depth, reward_mapping):
    """Convert a nested list to 1D tensor. Convert the general reward to specific according to some mapping.
    In input 0 represent intermediate, 1 represents buildingblock and -1 represents a dead end.
    """

    reward = 0

    for state in states:
        if state == "<BRANCH_SOLVED>":  # if branch is solved
            if reward_mapping["scale_with_depth"][1]:
                reward += (
                    reward_mapping["scale_score"][1]
                    * reward_mapping["scale_with_depth"][1]
                    * current_depth
                )
            else:
                reward += reward_mapping["scale_score"][-1]
        elif state == "<DEADEND_BRANCH>":  # if branch not solvable
            if reward_mapping["scale_with_depth"][-1]:
                reward += (
                    reward_mapping["scale_score"][-1]
                    * reward_mapping["scale_with_depth"][-1]
                    * current_depth
                )
            else:
                reward += reward_mapping["scale_score"][-1]
        else:  # if an additional reaction step
            if reward_mapping["scale_with_depth"][0]:
                reward += (
                    reward_mapping["scale_score"][0]
                    * reward_mapping["scale_with_depth"][0]
                    * current_depth
                )
            else:
                reward += reward_mapping["scale_score"][0]

    return reward


def get_general_rewards(states, current_depth):

    reward = []

    for state in states:
        if state == "<BRANCH_SOLVED>":  # if branch is solved
            reward.append(1)
        elif state == "<DEADEND_BRANCH>":  # if branch not solvable
            reward.append(-1)
        else:  # if an additional reaction step
            reward.append(0)

    return reward
