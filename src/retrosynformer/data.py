import pickle
import random
from collections import namedtuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .environment import RetroGymEnvironment
from .utils import utils


class RouteDatasetTorch(Dataset):
    def __init__(
        self,
        data,
        reward_col,
        action_dim,
        state_fp_dim,
        shuffle=False,
        drop_duplicates=True,
    ):
        self.data_full = data
        print(self.data_full.columns)
        if drop_duplicates:
            self.data = self.data_full.drop_duplicates(subset=["target"])
        else:
            self.data = self.data_full

        if shuffle:
            random.shuffle(self.data)
        self.target_routes = self.data.dict.tolist()
        self.target_routes = [
            self.data_full[self.data_full["target"] == target]["dict"].tolist()
            for target in self.data["target"].tolist()
        ]
        self.action_labels = self.data.actions.tolist()
        self.actions = [
            torch.nn.functional.one_hot(torch.tensor(actions), action_dim)
            for actions in self.data.actions.tolist()
        ]
        self.states = [
            convert_smiles_states_to_fp(state, n_bits=state_fp_dim)
            for state in self.data.states
        ]

        self.rewards = [
            torch.tensor(reward).unsqueeze(-1)
            for reward in self.data[reward_col].tolist()
        ]
        self.timesteps = [
            torch.arange(state.shape[0]).unsqueeze(1) for state in self.states
        ]
        self.attention_mask = [
            torch.ones(state.shape[0]).unsqueeze(1) for state in self.states
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return (
            self.states[key],
            self.actions[key],
            self.rewards[key],
            self.timesteps[key],
            self.attention_mask[key],
            self.action_labels[key],  #
            self.target_routes[key],
        )

    def get_all_target_routes(self, target):
        return self.data_full[self.data_full["target"] == target]["dict"].tolist()


def collate_fn(data):
    # print(data)
    lengths = [d[0].shape[0] for d in data]
    max_len = max(lengths)
    n_features = len(data[0]) - 2
    features = []
    for i in range(n_features):
        feature = torch.zeros((len(data), max_len, data[0][i].size(1)))
        features.append(feature)
        # print('feature.shape', feature.shape)
    action_labels, labels = [], []
    for i in range(len(data)):
        for j in range(n_features):
            j1, j2 = data[i][j].shape
            features[j][i] = torch.cat([data[i][j], torch.zeros((max_len - j1, j2))])
        action_labels.append(data[i][-2])
        labels.append(data[i][-1])

    return tuple(features), action_labels, labels


class RouteDataset:
    def __init__(
        self, routes_path, building_block_path, templates_path, template_library
    ):
        self.building_blocks = set(
            pd.read_csv(building_block_path)["inchi_key"].tolist()
        )
        self.templates_df = pd.read_pickle(templates_path)
        self.templates = list(self.templates_df["smiles"])
        print("n templates: ", len(self.templates))
        print("n templates_df: ", len(self.templates_df))
        template_library = pd.read_csv(template_library, sep="\t")
        self.template_library = template_library[
            template_library["template_hash_corr"].isin(list(self.templates_df["hash"]))
        ]
        if type(routes_path) == str:
            with open(routes_path, "rb") as file:
                self.routes = pickle.load(file)
        else:
            for r_path in routes_path:
                with open(routes_path, "rb") as file:
                    self.routes = pickle.load(file)

        self.Route = namedtuple(
            "Route",
            "dict reconstructed_dict target states actions reward_general reward_specific depth",
        )

        self.reaction_hash2template_general_hash = (
            get_reaction_hash2template_general_hash(self.template_library)
        )
        self.template_general_hash2template_general = (
            get_template_general_hash2template_general(self.templates_df)
        )

        self.corr_hash2idx = {
            row["hash"]: idx for idx, row in self.templates_df.iterrows()
        }

        self.max_steps = 10

    def create_route_data(self):
        self.Route = namedtuple(
            "Route",
            "dict reconstructed_dict target states actions reward_general depth",
        )
        self.data, self.invalid_routes = [], []
        invalid_routes_count = 0

        if (
            "reaction_hash"
            in self.routes[0]["route"]["rt"]["children"][0]["metadata"].keys()
        ):

            key_in_route = "reaction_hash"
        elif (
            "template_hash_corr"
            in self.routes[0]["route"]["rt"]["children"][0]["metadata"].keys()
        ):
            key_in_route = "template_hash_corr"

        for i in tqdm(range(0, len(self.routes))):

            route = self.routes[i]["route"]["rt"]

            product2reactants = utils.get_product2reactants(route)
            product2reaction_hash = utils.get_product2reaction_hash(
                route, hash_col=key_in_route
            )

            rxn_in_route = utils.route_to_list(route)

            try:
                if key_in_route == "reaction_hash":
                    hash_in_route = [r["hash"] for r in rxn_in_route]
                    template_hash_in_route = [
                        self.reaction_hash2template_general_hash[hash]
                        for hash in hash_in_route
                    ]
                    template_in_route = [
                        self.template_general_hash2template_general[hash]
                        for hash in template_hash_in_route
                    ]

                    template_hash = self.reaction_hash2template_general_hash[
                        route["children"][0]["metadata"]["reaction_hash"]
                    ]
                elif key_in_route == "template_hash_corr":
                    template_hash = route["children"][0]["metadata"][
                        "template_hash_corr"
                    ]

                reaction_list = []
                target = route["smiles"]
                template = self.template_general_hash2template_general[template_hash]

                env = RetroGymEnvironment(
                    self.building_blocks,
                    self.templates_df,
                    None,
                    None,
                    process_routes=True,
                )
                env.set_target_compound(target)
                route_done, route_done, _ = env._check_if_done()

                states_smiles = [[target]]
                actions_route = []
                depths_in_route = [0]

                while not route_done:
                    if key_in_route == "reaction_hash":
                        template_from_route = self.reaction_hash2template_general_hash[
                            product2reaction_hash[target]
                        ]
                    elif key_in_route == "template_hash_corr":
                        try:
                            template_from_route = product2reaction_hash[target]
                        except:
                            template_from_route = None
                            route_done = True
                    if template_from_route:
                        reactants = product2reactants[target]
                        ordered_reactants, reward, _ = env.step_from_reactants(
                            reactants=reactants
                        )
                        route_done, _, _ = env._check_if_done()

                    if not route_done:
                        states_smiles.append(ordered_reactants[0:2])
                        depths_in_route.append(env.current_depth)
                    if template_from_route:
                        action_idx = self.corr_hash2idx[template_from_route]
                        actions_route.append(action_idx)

                        reaction_list.append(
                            get_reaction_string(ordered_reactants, target)
                        )

                    if not route_done:
                        target = env.state[-1][0]

                        reconstructed_route = utils.list2route(
                            reaction_list
                        ).reaction_tree
                    else:
                        route_data = self.Route(
                            dict=route,
                            reconstructed_dict=reconstructed_route,
                            target=states_smiles[0],
                            states=states_smiles,
                            actions=actions_route,
                            reward_general=env.all_rewards["reward_general"],
                            depth=depths_in_route,
                        )
                        self.data.append(route_data)
            except:
                invalid_routes_count += 1

        print("Faild to reconstruct ", invalid_routes_count, " routes.")

        return pd.DataFrame(data=self.data)


# Functions for extracting leaves from routes - to be used to create a set of building blocks
def extract_all_leaves(routes: list) -> list:
    leaves = []
    for route in routes:
        leaves.extend(route["route"]["leaves"])

    return list(set(leaves))


def create_building_block_df(leaves: list) -> pd.DataFrame:
    df = pd.DataFrame({"inchi_key": leaves})
    return df


def extract_template_products(available_templates: list, retro: bool = True) -> list:
    product_idx = 0 if retro else 1
    return [t.split(">>")[product_idx] for t in available_templates]


# Create Dictionaries
def get_reaction_string(reactants: list, product: str) -> str:
    return ".".join(reactants) + ">>" + product


def get_reaction2template_hash_general(template_df: pd.DataFrame) -> dict:
    return {
        row["reaction_hash"]: row["template_hash"] for _, row in template_df.iterrows()
    }


def get_reaction_hash2template_general(template_df: pd.DataFrame) -> dict:
    return {
        row["reaction_hash"]: row["retro_template_corr"]
        for _, row in template_df.iterrows()
    }


def get_reaction_hash2template_general_hash(template_df: pd.DataFrame) -> dict:
    return {
        row["reaction_hash"]: row["template_hash_corr"]
        for _, row in template_df.iterrows()
    }


def get_template_general_hash2template_general(template_df: pd.DataFrame) -> dict:
    return {row["hash"]: row["smiles"] for _, row in template_df.iterrows()}


def convert_smiles_states_to_fp(smiles_list, n_bits=1024, include_n_fps=2):
    fingerprint_list = []
    empty_fp = np.zeros(n_bits)
    for state in smiles_list:
        state_fp = []
        for smi in state[0:include_n_fps]:
            state_fp.append(utils.get_morgan_fingerprint(smi, radius=2, n_bits=n_bits))
        while len(state_fp) < include_n_fps:
            state_fp.append(empty_fp)
        state_fp = torch.tensor(np.concatenate(state_fp)[: int(n_bits * include_n_fps)])
        fingerprint_list.append(state_fp)
    return torch.stack(fingerprint_list)
