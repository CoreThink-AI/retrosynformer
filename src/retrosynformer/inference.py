import copy
import time
from collections import namedtuple
from operator import attrgetter

import pandas as pd
import torch
import tqdm
from rdkit import Chem
from rxnutils.routes import base, comparison

from .data import convert_smiles_states_to_fp
from .environment import RetroGymEnvironment
from .utils import reward_functions, utils


class RoutePredictor:
    def __init__(self, model, config, beam_width=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.config = config
        self.building_block_path = self.config["context"]["building_blocks"]
        self.building_blocks = pd.read_csv(self.building_block_path)[
            "inchi_key"
        ].tolist()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.templates_df = pd.read_pickle(self.config["context"]["templates_path"])
        self.reward_mapping = reward_functions.get_reward_mapping(config)
        self.calculator = comparison.route_distances_calculator("ted")
        self.softmax = torch.nn.Softmax(dim=-1)
        self.max_depth = self.config["evaluation"]["max_depth"]
        self.result_df = pd.DataFrame({})
        self.eval_df = pd.DataFrame({})
        self.Beam = namedtuple(
            "Beam",
            [
                "env",
                "states",
                "actions",
                "rtgs_tensor",
                "rewards",
                "total_reward",
                "timesteps",
                "attention_mask",
                "reaction_list",
                "predicted_actions",
                "route_solved",
                "route_done",
                "trajectory_prob",
            ],
        )

    def set_model(self, model):
        self.model = model
        print("New model set.")

    def predict_route(self, target, beam_width, target_reward=0.5):

        beam_width = (
            beam_width if beam_width else self.config["evaluation"]["beam_width"]
        )

        """Predict route with beam search.
        beam_width=1 is equivalent to greedy or no beam search"""
        self.model.eval()
        self.env = RetroGymEnvironment(
            self.building_blocks, self.templates_df, self.reward_mapping, self.max_depth
        )
        self.env.set_target_compound(
            target,
            reward_function="reward_specific",
        )

        states = [[target]]
        actions = torch.zeros((1, 1, len(self.env.available_templates)))
        rtgs_tensor = torch.tensor([target_reward]).reshape((1, 1, 1))
        rewards = torch.zeros((1, 1, 1))
        timesteps = torch.zeros((1, 1))
        attention_mask = torch.ones((1, 1)).to(self.device)

        logits_unmasked = []
        logits = []
        episode_length = 1
        batch_size = 1

        actions_tensor = actions.to(
            device=self.device, dtype=torch.float32
        )  # (batch_size, episode_length, state_dim)
        rtgs_tensor = rtgs_tensor.to(
            device=self.device, dtype=torch.float32
        )  # (batch_size, episode_length, 1)
        rewards = rewards.to(device=self.device, dtype=torch.float32)
        attention_mask = attention_mask.to(device=self.device, dtype=torch.float32)
        timesteps = torch.cat(
            [
                torch.arange(episode_length).reshape(1, episode_length)
                for _ in range(batch_size)
            ],
            dim=0,
        )
        timesteps = timesteps.to(
            device=self.device, dtype=torch.long
        )  # (batch_size, episode_length))

        beam = self.Beam(
            env=self.env,
            states=states,
            actions=actions,
            rtgs_tensor=rtgs_tensor,
            rewards=rewards,
            total_reward=float(torch.sum(rewards, dim=None)),
            timesteps=timesteps,
            attention_mask=attention_mask,
            reaction_list=[],
            predicted_actions=[],
            route_solved=self.env.route_solved,
            route_done=self.env.route_done,
            trajectory_prob=1,
        )
        current_beams = [beam]
        any_solved_beam = False
        all_beams_done = False
        with torch.no_grad():

            while not (any_solved_beam or all_beams_done):
                new_beams = []
                route_done_beams, route_solved_beams = [], []
                for i, beam_i in enumerate(current_beams):
                    assert not beam_i.route_done, (
                        beam_i,
                        any_solved_beam,
                    )
                    top_k_beams, _route_done, _route_solved = self.expand_beam(
                        beam_i, beam_width
                    )
                    new_beams.extend(top_k_beams)
                    route_done_beams.extend(_route_done)
                    route_solved_beams.extend(_route_solved)
                    if (
                        sum(_route_solved) > 0
                    ):  # Change if we want more than one solved beam
                        any_solved_beam = True
                    if sum(_route_done) == len(_route_done):
                        all_beams_done = True

                if len(new_beams) == 0:
                    all_beams_done = True
                filtered_new_beams = []
                for i in range(len(new_beams)):
                    if not route_done_beams[i]:
                        filtered_new_beams.append(new_beams[i])
                    else:
                        if route_solved_beams[i]:
                            filtered_new_beams.append(new_beams[i])

                if "sort_on" in self.config["evaluation"].keys():
                    sort_on = self.config["evaluation"]["sort_on"]
                else:
                    sort_on = "total_reward"  # "trajectory_prob"

                sorted_beams = sorted(
                    filtered_new_beams,
                    key=attrgetter("route_solved", sort_on),
                    reverse=True,
                )[:beam_width]
                if len(sorted_beams) > 0:
                    best_beam = sorted_beams[0]
                else:
                    best_beam = None
                current_beams = sorted_beams

        if best_beam:
            return best_beam
        else:
            return None

    def expand_beam(self, parent_beam, beam_width=3):
        """Takes one beam containing a sequence of actions and its corresponding environment and expands the state.
        Returns the k new beams."""
        new_beams = []
        state = parent_beam.env.state[-1][0]
        target_mol = Chem.MolFromSmiles(state)
        if not target_mol:
            return [], [], []

        states_tensor = convert_smiles_states_to_fp(
            parent_beam.states,
            n_bits=self.config["dataset"]["fp_dim"],
            include_n_fps=self.config["dataset"]["n_in_state"],
        )

        states_tensor = states_tensor.to(
            device=self.device, dtype=torch.float32
        ).unsqueeze(
            0
        )  # (batch_size, episode_length, state_dim)

        _, action_preds, _ = self.model(
            states=states_tensor.to(self.device),
            actions=parent_beam.actions.to(self.device),
            rewards=parent_beam.rewards.to(self.device),
            returns_to_go=parent_beam.rtgs_tensor.to(self.device),
            timesteps=parent_beam.timesteps.to(self.device),
            attention_mask=parent_beam.attention_mask.to(self.device),
            return_dict=False,
        )
        action_preds = self.softmax(action_preds)
        action_preds = action_preds[0][-1].flatten().cpu()
        k = int(self.config["dataset"]["action_dim"])  # 1573
        _, top50_action_idx = torch.topk(action_preds, k=k, dim=-1)

        action_preds_mask = torch.ones(action_preds.shape, dtype=bool)
        action_preds_mask[top50_action_idx] = False
        action_preds[action_preds_mask] = -2

        top50_actions = self.env.available_templates[
            top50_action_idx
        ]

        available_actions = torch.tensor(
            utils.check_available_actions(
                state,
                top50_actions,
                use_template=True,
            )[0]
        )

        available_actions_mask = torch.ones(action_preds.shape, dtype=bool)
        available_actions_mask[top50_action_idx[available_actions]] = False

        avail_actions = self.env.available_templates[
            top50_action_idx[available_actions]
        ]

        if type(avail_actions) == str:
            avail_actions = [avail_actions]
        else:
            avail_actions = avail_actions.tolist()

        action_preds[available_actions_mask] = -2
        action_preds[0] = -2

        if sum(available_actions) < 1:
            next_action_idx = [0]
        else:
            # Sorted on trajectory_prob
            next_action_pred, next_action_idx = torch.topk(
                action_preds, k=beam_width, dim=0
            )

        # Expand current beam
        route_done_beams, route_solved_beams = [], []
        for i, next_action in enumerate(next_action_idx):

            current_beam = copy.deepcopy(parent_beam)
            current_beam.predicted_actions.append(next_action)
            next_action = self.env.available_templates[next_action]
            next_reactants = current_beam.env.step([next_action])

            if next_reactants and len(next_reactants) > 0:
                route_done, route_solved, _ = current_beam.env._check_if_done()
                route_done_beams.append(route_done)
                route_solved_beams.append(route_solved)

                reaction = ".".join(next_reactants) + ">>" + state
                current_beam.reaction_list.append(reaction)

                current_beam.states.append(next_reactants)

                new_actions = torch.cat(
                    [
                        current_beam.actions,
                        utils.one_hot_encoder(
                            next_action_idx, self.config["dataset"]["action_dim"]
                        )
                        .unsqueeze(0)
                        .unsqueeze(0),
                    ],
                    dim=1,
                )
                new_rewards = (
                    torch.tensor(current_beam.env.rewards).unsqueeze(0).unsqueeze(-1)
                )
                new_rtg = (
                    (current_beam.rtgs_tensor[0][-1] - current_beam.env.rewards[-1])
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )
                new_attention_mask = torch.cat(
                    (current_beam.attention_mask, torch.ones(1, 1).to(self.device)),
                    dim=1,
                )
                new_rtgs_tensor = torch.cat(
                    (current_beam.rtgs_tensor, new_rtg), dim=1
                )
                new_timesteps = torch.arange(
                    0, len(current_beam.predicted_actions) + 1
                ).unsqueeze(0)

                beam_new = self.Beam(
                    env=current_beam.env,
                    states=current_beam.states,
                    actions=new_actions,
                    rtgs_tensor=new_rtgs_tensor,
                    rewards=new_rewards,
                    total_reward=float(torch.sum(new_rewards, dim=None)),
                    timesteps=new_timesteps,
                    attention_mask=new_attention_mask,
                    reaction_list=current_beam.reaction_list,
                    predicted_actions=current_beam.predicted_actions,
                    route_solved=route_solved,
                    route_done=route_done,
                    trajectory_prob=float(
                        current_beam.trajectory_prob * next_action_pred[i]
                    ),
                )
                new_beams.append(beam_new)

        return new_beams, route_done_beams, route_solved_beams

    def eval_predicted_routes(self, dataloader):

        routes = []
        self.model.eval()
        with torch.no_grad():
            for batch_no, data in enumerate(
                dataloader
            ):
                if batch_no == self.config["evaluation"]["eval_n_batches"]:
                    break

                (
                    (
                        states,
                        actions,
                        rewards,
                        timesteps,
                        attention_mask,
                    ),
                    action_labels,
                    target_routes,
                ) = data

                for j in range(len(states)):

                    target_compound = target_routes[j][0]["smiles"]
                    start_time = time.time()
                    best_beam = self.predict_route(
                        target_compound,
                        beam_width=self.config["evaluation"]["beam_width"],
                    )
                    total_time = time.time() - start_time
                    route = {}
                    route["target"] = target_compound
                    route["target_tree"] = target_routes[j]
                    route["time"] = total_time
                    route["pred_tree"] = None
                    if best_beam:
                        route["route_solved"] = best_beam.route_solved
                        route["n_reactions"] = len(best_beam.reaction_list)
                        route["leafs"] = best_beam.env.leafs
                        route["n_branchings"] = best_beam.env.number_of_branchings
                        route["n_dead_ends"] = best_beam.env.dead_ends
                        route["predicted_reaction_lists"] = best_beam.reaction_list
                        route["predicted_action_list"] = [
                            a.item() for a in best_beam.predicted_actions
                        ]
                        route["target_action_list"] = [
                            torch.argmax(a, dim=0).tolist() for a in actions[j]
                        ]
                        route["predicted_rewards"] = best_beam.env.rewards
                        route["target_rewards"] = [
                            r.flatten().tolist() for r in rewards[j]
                        ]
                        route["trajectory_prob"] = best_beam.trajectory_prob
                        try:
                            pred_tree = utils.list2route(best_beam.reaction_list
                            ).reaction_tree
                            pred_tree, route_solved = utils.add_in_stock_property_to_trees(
                                 pred_tree, self.building_blocks)
                            route["pred_tree"] = pred_tree
                            route["route_solved"] = route_solved
                            route["TED to target"], most_similar_target_route_idx = (
                                utils.calculate_ted(
                                    self.calculator,
                                    base.SynthesisRoute(route["pred_tree"]),
                                    [
                                        base.SynthesisRoute(r)
                                        for r in route["target_tree"]
                                    ],
                                )
                            )
                            route["target_tree"] = target_routes[j][
                                most_similar_target_route_idx
                            ]
                            route["valid_route"] = True
                        except:
                            route["valid_route"] = False
                    else:
                        route["route_solved"] = False
                        route["valid_route"] = False
                    
                    routes.append(route)
        return routes
