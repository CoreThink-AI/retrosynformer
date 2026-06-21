import copy
import logging
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from operator import attrgetter

import pandas as pd
import torch
from rdkit import Chem
from rxnutils.routes import base

from .data import convert_smiles_states_to_fp
from .environment import RetroGymEnvironment
from .utils import reward_functions, utils

logger = logging.getLogger(__name__)


class RoutePredictor:
    def __init__(self, model, config, beam_width=None):
        self.device = utils.get_device()
        self.model = model.to(self.device)
        self.config = config
        self.building_block_path = self.config["context"]["building_blocks"]
        self.building_blocks = pd.read_csv(self.building_block_path)[
            "inchi_key"
        ].tolist()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.templates_df = pd.read_pickle(self.config["context"]["templates_path"])
        self.reward_mapping = reward_functions.get_reward_mapping(config)
        exhaustive_limit = config["evaluation"].get("ted_exhaustive_limit", 20)
        self.calculator = utils.make_ted_calculator(exhaustive_limit=exhaustive_limit)
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
        logger.debug("New model set.")

    def predict_route(self, target, beam_width, target_reward=0.5):

        beam_width = (
            beam_width if beam_width else self.config["evaluation"]["beam_width"]
        )

        """Predict route with beam search.
        beam_width=1 is equivalent to greedy or no beam search"""
        self.model.eval()
        # default RetroGymEnvironment.process_routes is False
        self.env = RetroGymEnvironment(
            self.building_blocks, self.templates_df, self.reward_mapping, self.max_depth, process_routes=False,
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

        episode_length = 1
        batch_size = 1

        actions = actions.to(
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

        if isinstance(avail_actions, str):
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
                        .unsqueeze(0)
                        .to(self.device),
                    ],
                    dim=1,
                )
                new_rewards = (
                    torch.tensor(current_beam.env.rewards, device=self.device).unsqueeze(0).unsqueeze(-1)
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
                    0, len(current_beam.predicted_actions) + 1, device=self.device
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

    def _apply_templates_for_beam(
        self,
        parent_beam,
        action_preds_1d: torch.Tensor,
        beam_width: int,
    ) -> tuple:
        """Template-application half of beam expansion given pre-computed predictions.

        Receives a 1-D CPU tensor of softmaxed action probabilities (already
        computed by a batched GPU forward pass in predict_all_routes) and applies
        rdchiral templates to generate child beams.  Called in parallel across
        all current beams via ThreadPoolExecutor; safe because:
          - action_preds_1d is a per-beam tensor with no shared mutable state
          - self.env.available_templates is read-only
          - copy.deepcopy gives each child beam its own env copy
        """
        new_beams = []
        state = parent_beam.env.state[-1][0]
        if not Chem.MolFromSmiles(state):
            return [], [], []

        k = int(self.config["dataset"]["action_dim"])
        _, top_k_idx = torch.topk(action_preds_1d, k=k)

        preds = action_preds_1d.clone()
        not_top_k = torch.ones_like(preds, dtype=torch.bool)
        not_top_k[top_k_idx] = False
        preds[not_top_k] = -2.0

        top_k_actions = self.env.available_templates[top_k_idx]
        avail_mask = torch.tensor(
            utils.check_available_actions(state, top_k_actions, use_template=True)[0]
        )

        not_avail = torch.ones_like(preds, dtype=torch.bool)
        not_avail[top_k_idx[avail_mask]] = False
        preds[not_avail] = -2.0
        preds[0] = -2.0

        if avail_mask.sum() < 1:
            next_action_idx = torch.tensor([0])
            next_action_pred = preds[[0]]
        else:
            next_action_pred, next_action_idx = torch.topk(
                preds, k=min(beam_width, int(avail_mask.sum()))
            )

        route_done_beams, route_solved_beams = [], []
        for i, next_action in enumerate(next_action_idx):
            current_beam = copy.deepcopy(parent_beam)
            current_beam.predicted_actions.append(next_action)
            next_action_template = self.env.available_templates[next_action]
            next_reactants = current_beam.env.step([next_action_template])

            if not next_reactants:
                continue

            route_done, route_solved, _ = current_beam.env._check_if_done()
            route_done_beams.append(route_done)
            route_solved_beams.append(route_solved)

            reaction = ".".join(next_reactants) + ">>" + state
            current_beam.reaction_list.append(reaction)
            current_beam.states.append(next_reactants)

            new_actions = torch.cat([
                current_beam.actions,
                utils.one_hot_encoder(next_action_idx, self.config["dataset"]["action_dim"])
                .unsqueeze(0).unsqueeze(0).to(self.device),
            ], dim=1)
            new_rewards = (
                torch.tensor(current_beam.env.rewards, device=self.device)
                .unsqueeze(0).unsqueeze(-1)
            )
            new_rtg = (
                (current_beam.rtgs_tensor[0][-1] - current_beam.env.rewards[-1])
                .unsqueeze(0).unsqueeze(-1)
            )
            new_attention_mask = torch.cat(
                (current_beam.attention_mask, torch.ones(1, 1, device=self.device)), dim=1
            )
            new_rtgs_tensor = torch.cat((current_beam.rtgs_tensor, new_rtg), dim=1)
            new_timesteps = torch.arange(
                0, len(current_beam.predicted_actions) + 1, device=self.device
            ).unsqueeze(0)

            beam_new = self.Beam(
                env=current_beam.env,
                states=current_beam.states,
                actions=new_actions,
                rtgs_tensor=new_rtgs_tensor,
                rewards=new_rewards,
                total_reward=float(torch.sum(new_rewards)),
                timesteps=new_timesteps,
                attention_mask=new_attention_mask,
                reaction_list=current_beam.reaction_list,
                predicted_actions=current_beam.predicted_actions,
                route_solved=route_solved,
                route_done=route_done,
                trajectory_prob=float(current_beam.trajectory_prob * next_action_pred[i]),
            )
            new_beams.append(beam_new)

        return new_beams, route_done_beams, route_solved_beams

    def predict_all_routes(
        self,
        target: str,
        beam_width: int,
        target_reward: float = 0.5,
        max_depth: int | None = None,
    ) -> list:
        """Run beam search to full completion and return all terminal beams.

        Unlike ``predict_route``, which stops the moment any beam is solved,
        this method continues expanding non-terminal beams until none remain.
        Both solved beams (all leaves are building blocks) and dead-end beams
        (at least one branch hit a dead-end) are collected and returned sorted
        by ``trajectory_prob`` descending.

        Returns an empty list when the target SMILES cannot be parsed by
        ``expand_beam`` (e.g. all templates fail on the first step).

        ``max_depth`` overrides the value from config when provided, allowing
        per-request depth control without reloading the model.

        >>> # Instantiation requires data files; use predict_route tests for that.
        >>> # This docstring exists to document the return contract.
        >>> # Returns: list[Beam] — all terminal beams, sorted by trajectory_prob desc.
        True
        """
        effective_max_depth = max_depth if max_depth is not None else self.max_depth
        self.model.eval()
        self.env = RetroGymEnvironment(
            self.building_blocks,
            self.templates_df,
            self.reward_mapping,
            effective_max_depth,
            process_routes=False,
        )
        self.env.set_target_compound(target, reward_function="reward_specific")

        states = [[target]]
        n_templates = len(self.env.available_templates)
        actions = torch.zeros((1, 1, n_templates), device=self.device, dtype=torch.float32)
        rtgs_tensor = torch.tensor([[[target_reward]]], device=self.device, dtype=torch.float32)
        rewards = torch.zeros((1, 1, 1), device=self.device, dtype=torch.float32)
        attention_mask = torch.ones((1, 1), device=self.device, dtype=torch.float32)
        timesteps = torch.zeros((1, 1), device=self.device, dtype=torch.long)

        initial_beam = self.Beam(
            env=self.env,
            states=states,
            actions=actions,
            rtgs_tensor=rtgs_tensor,
            rewards=rewards,
            total_reward=0.0,
            timesteps=timesteps,
            attention_mask=attention_mask,
            reaction_list=[],
            predicted_actions=[],
            route_solved=self.env.route_solved,
            route_done=self.env.route_done,
            trajectory_prob=1.0,
        )

        # Target is itself a building block — nothing to expand.
        if initial_beam.route_done:
            return [initial_beam]

        current_beams = [initial_beam]
        terminal_beams: list = []

        # One ThreadPoolExecutor for the whole search; workers apply templates in
        # parallel while the GPU runs the batched forward pass each depth level.
        with torch.no_grad(), ThreadPoolExecutor(max_workers=beam_width) as executor:
            while current_beams:
                next_beams: list = []

                # ── Batched fingerprint computation (parallel, RDKit releases GIL) ──
                fp_futures = [
                    executor.submit(
                        convert_smiles_states_to_fp,
                        b.states,
                        self.config["dataset"]["fp_dim"],
                        self.config["dataset"]["n_in_state"],
                    )
                    for b in current_beams
                ]
                states_batch = torch.stack(
                    [f.result().to(dtype=torch.float32) for f in fp_futures], dim=0
                ).to(self.device)  # (N, T, state_dim)

                # ── Single GPU forward pass for all N beams ──
                _, action_preds_batch, _ = self.model(
                    states=states_batch,
                    actions=torch.cat([b.actions for b in current_beams], dim=0),
                    rewards=torch.cat([b.rewards for b in current_beams], dim=0),
                    returns_to_go=torch.cat([b.rtgs_tensor for b in current_beams], dim=0),
                    timesteps=torch.cat([b.timesteps for b in current_beams], dim=0),
                    attention_mask=torch.cat([b.attention_mask for b in current_beams], dim=0),
                    return_dict=False,
                )
                # (N, T, n_templates) → softmax over last timestep → CPU: (N, n_templates)
                preds_per_beam = self.softmax(action_preds_batch[:, -1, :]).cpu()

                # ── Parallel template application across all beams ──
                results = list(executor.map(
                    lambda args: self._apply_templates_for_beam(*args),
                    [(b, preds_per_beam[i], beam_width) for i, b in enumerate(current_beams)],
                ))

                for (new_beams, done_flags, _), beam_i in zip(results, current_beams):
                    if not new_beams:
                        terminal_beams.append(beam_i)
                    else:
                        for b, done in zip(new_beams, done_flags):
                            (terminal_beams if done else next_beams).append(b)

                next_beams.sort(key=lambda b: b.trajectory_prob, reverse=True)
                current_beams = next_beams[:beam_width]

        terminal_beams.sort(key=lambda b: b.trajectory_prob, reverse=True)
        return terminal_beams

    def eval_predicted_routes(self, dataloader):

        routes = []
        n_batches = self.config["evaluation"]["eval_n_batches"]
        batch_size = self.config["evaluation"]["batch_size"]
        n_total = (n_batches or 0) * batch_size
        eval_t0 = time.time()
        self.model.eval()
        with torch.no_grad():
            for batch_no, data in enumerate(
                dataloader
            ):
                if batch_no == n_batches:
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

                batch_t0 = time.time()
                print(f"  Route eval batch {batch_no + 1}/{n_batches or '?'}"
                      f" ({len(states)} compounds) …", flush=True)
                n_solved_batch = 0
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
                        except Exception as exc:
                            logger.error(exc)
                            route["valid_route"] = False
                    else:
                        route["route_solved"] = False
                        route["valid_route"] = False

                    if route.get("route_solved"):
                        n_solved_batch += 1
                    routes.append(route)

                elapsed = time.time() - batch_t0
                frac = n_solved_batch / len(states) if len(states) and states.any() else 0.0
                print(f"    solved {n_solved_batch}/{len(states)} ({frac:.1%})"
                      f"  {elapsed:.1f}s  ({elapsed / len(states):.2f}s/mol)",
                      flush=True)
        n_solved = sum(r.get("route_solved", False) for r in routes)
        print(f"  Route eval done: {n_solved}/{len(routes)} solved"
              f"  total {time.time() - eval_t0:.1f}s", flush=True)
        return routes
