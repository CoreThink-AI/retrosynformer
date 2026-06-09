import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdchiral.main as rdc
import rdkit
import seaborn as sns
import torch
import yaml
from rdkit import Chem
from rdkit.Chem import AllChem
from rxnutils.routes import base, readers


def flatten(xss):
    return [x for xs in xss for x in xs]

def get_product2reactants(route):
    rxn = base.SynthesisRoute(route)
    product2reactants = {}
    for reaction in rxn.reaction_data():
        reactants, product = reaction["reaction_smiles"].split(">>")
        reactants = reactants.split(".")
        product2reactants[product] = reactants

    return product2reactants


def get_product2reaction_hash(route, hash_col="reaction_hash"):
    product2reaction_hash = {}
    rxn = base.SynthesisRoute(route)
    for rxn_dict in rxn.reaction_data():
        reaction_hash = rxn_dict[hash_col]
        product = rxn_dict["reaction_smiles"].split(">>")[1]
        assert product not in product2reaction_hash, (
            "product already in route",
            product,
            rxn.reaction_data(),
        )
        product2reaction_hash[product] = reaction_hash

    return product2reaction_hash


def apply_template(template_smarts, product_smiles):
    """Apply a reaction template on the products to get the reactants."""
    outcome = rdc.rdchiralRunText(template_smarts, product_smiles)
    reactants = list(str_.split(".") for str_ in outcome)
    return reactants


def display_route(route):
    rxn = base.SynthesisRoute(route)
    return rxn.image(show_atom_mapping=False)


def route_to_list(route):
    rxn = base.SynthesisRoute(route)
    rxn.reaction_data()
    reactions = []
    for r in rxn.reaction_data():
        rxn_dict = {"smiles": r["reaction_smiles"]}
        try:
            rxn_dict["hash"] = r["reaction_hash"]
        except:
            try:
                rxn_dict["hash_corr"] = r["template_hash_corr"]
            except:
                try:
                    rxn_dict["hash_corr"] = r["template_hash"]
                except:
                    pass

        reactions.append(rxn_dict)
    return reactions  # reaction_list, reaction_hash_list


def list2route(reaction_list):
    return readers.reactions2route(reaction_list)


# TODO - fix cache
# import functools
# @functools.cache
# move loop to outside of function
def check_available_template(target_compound, template_products, retro=True):
    """TODO: Write function to check if no applicable templates are available.
    Without this there will be issues in trying to apply templates.
    Also the reward might be off, as it need to run until reaching max depth.
    Currently dummy function.
    """

    target_mol = Chem.MolFromSmiles(target_compound)

    available_actions = []
    for template in template_products:
        if template == "<eos>":
            available_actions.append(True)
        else:
            available_actions.append(
                target_mol.HasSubstructMatch(Chem.MolFromSmarts(template))
            )
    return available_actions


def check_available_actions(
    target_compound, available_reactions, use_template=True, retro=True
):
    """TODO: Write function to check if no applicable templates are available.
    Without this there will be issues in trying to apply templates.
    Also the reward might be off, as it need to run until reaching max depth.
    Currently dummy function.
    """

    dir_idx = 0 if retro else 1
    if use_template:
        template_products = []
        for template_list in available_reactions:
            if type(template_list) == list:
                template_list = template_list[0]

            if template_list == "<eos>":
                template_products.append("<eos>")
            else:
                product = template_list.split(">>")[dir_idx]
                template_products.append(product)
        assert len(template_products) == len(available_reactions)
    else:
        template_products = available_reactions
    target_mol = Chem.MolFromSmiles(target_compound)
    available_actions, available_templates = [], []

    for template in template_products:
        is_available = False
        available_template = None
        if template == "<eos>":
            is_available = True
        else:
            is_available = target_mol.HasSubstructMatch(Chem.MolFromSmarts(template))
            if is_available:
                available_template = template
        available_actions.append(is_available)
        available_templates.append(available_template)
    return available_actions, available_templates


def one_hot_encoder(i, dim):
    one_hot_vector = torch.zeros(dim)
    one_hot_vector[i] = 1

    return one_hot_vector


def get_morgan_fingerprint(smiles, radius, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)
    fingerprint_list = [int(bit) for bit in fingerprint]
    return torch.tensor(fingerprint_list)


def read_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def write_config(config_path, content):
    with open(config_path, "w") as file:
        yaml.dump(content, file)


def convert_batches_to_action_ids(actions, action_preds):
    actions_id = []
    actions_id_pred = []
    action_preds_prob = []

    for i in range(len(actions)):
        for j in range(len(actions[i])):
            if torch.all(actions[i][j] == 0):
                pass
            else:
                actions_id.append(int(torch.argmax(actions[i][j], dim=-1)))
                actions_id_pred.append(int(torch.argmax(action_preds[i][j], dim=-1)))
                action_preds_prob.append(action_preds[i][j].tolist())

    return actions_id, actions_id_pred, action_preds_prob


def convert_batches_to_action_ids_batch(actions, action_preds):
    actions_id = []
    actions_id_pred = []

    for i in range(len(actions)):
        route_ids = []
        route_pred_ids = []
        route_preds = []
        for j in range(len(actions[i])):
            if not torch.all(actions[i][j] == 0):
                route_ids.append(int(torch.argmax(actions[i][j], dim=-1)))
                route_pred_ids.append(int(torch.argmax(action_preds[i][j], dim=-1)))
                route_preds.append(actions[i][j].tolist())

        actions_id.append(route_ids)
        actions_id_pred.append(route_pred_ids)

    return (
        actions_id,
        actions_id_pred,
    )  # action_preds


def calculate_ted(calculator, route, target_routes):
    """Calculate the Tree Edit Distance between 2 routes."""
    routes = [route]
    routes.extend(target_routes)
    c = calculator(routes)
    min_dist_idx = np.argmin(c[0][1:])

    return c[0][min_dist_idx + 1], min_dist_idx


# ----------- PLOT RESULTS -----------


def plot_train_progress_accuracy(train_results_path, save_as):
    sns.set()
    train_progress = pd.read_csv(train_results_path)
    # print(train_progress.keys())
    fig = plt.figure()
    plt.plot(
        train_progress["epoch"],
        train_progress["train_action_accuracy"],
        label="Train action accuracy",
    )
    plt.plot(
        train_progress["epoch"],
        train_progress["valid_action_accuracy"],
        label="Valid action accuracy",
    )
    plt.plot(
        train_progress["epoch"],
        train_progress["train_route_accuracy"],
        label="Train route accuracy",
    )
    plt.plot(
        train_progress["epoch"],
        train_progress["valid_route_accuracy"],
        label="Valid route accuracy",
    )
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()


def plot_train_progress(train_results_path, save_as):
    sns.set()
    train_progress = pd.read_csv(train_results_path)

    fig = plt.figure()
    plt.plot(train_progress["epoch"], train_progress["train_loss"], 
             label="Train loss")
    plt.plot(train_progress["epoch"], train_progress["valid_loss"], label="Valid loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()


def plot_evaluation_results(evaluation_results_path, save_as):
    eval_progress = pd.read_json(evaluation_results_path)
    epochs = eval_progress["epoch"].tolist()
    percent_solved = []
    for epoch in epochs:
        eval_progress_epoch = pd.DataFrame(
            eval_progress[eval_progress["epoch"] == epoch]["result"].tolist()[0]
        )
        solved = sum(eval_progress_epoch["route_solved"]) / len(
            eval_progress_epoch["route_solved"]
        )
        percent_solved.append(solved * 100)

    sns.set()
    fig = plt.figure()
    plt.plot(epochs, percent_solved, marker="o", label="Percent targets solved")
    plt.ylabel("% targets solved")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def flatten_and_crop(list_a, list_b):
    flat_and_cropped_list_a, flat_and_cropped_list_b = [], []

    for itema, itemb in zip(list_a, list_b):
        if isinstance(itema, list):
            itema = [i for i in itema if i != 0]
            itemb = [i for i in itemb if i != 0]
            len_shortest_list = min([len(itema), len(itemb)])

            itema = itema[:len_shortest_list]
            itemb = itemb[:len_shortest_list]

            flat_and_cropped_list_a.extend(itema)
            flat_and_cropped_list_b.extend(itemb)
    return flat_and_cropped_list_a, flat_and_cropped_list_b


def get_index_values(list_a, list_b, idx):
    flat_and_cropped_list_a, flat_and_cropped_list_b = [], []

    for itema, itemb in zip(list_a, list_b):
        if len(itema) > idx and len(itemb) > idx:
            flat_and_cropped_list_a.extend(itema[idx])
            flat_and_cropped_list_b.extend(itemb[idx])

    return flat_and_cropped_list_a, flat_and_cropped_list_b


def check_if_building_block(smiles, building_blocks):
    mol = Chem.MolFromSmiles(smiles)
    inchi_key = Chem.inchi.MolToInchiKey(mol)
    if inchi_key in building_blocks:
        building_block = True
    else:
        building_block = False

    return building_block

# In stock property
def add_in_stock_property_to_trees(tree, building_blocks, only_bb_leafs=True):
    # If the current dictionary is of type 'mol' and has no children
    if tree.get("type") == "mol" and "children" not in tree:
        if check_if_building_block(tree.get("smiles"), building_blocks):
            tree["in_stock"] = True
        else:
            tree["in_stock"] = False
            only_bb_leafs = False
    # If the current dictionary has children, recurse
    elif "children" in tree:
        for child in tree["children"]:
            modified_child, bb_leafs = add_in_stock_property_to_trees(
                child, building_blocks, only_bb_leafs
            )
            if not bb_leafs:
                only_bb_leafs = bb_leafs

    return tree, only_bb_leafs  # Return updated tree

def check_route_leafs(rxn, bb):
    route_leafs = []
    all_is_bb = True
    is_not_bb = []
    for i in str(rxn).split("'smiles': '"):
        if not "children" in i:
            if "}" in i:
                i = i.split("'}")[0]
                i = i.split("',")[0]
                route_leafs.append(i)
                is_bb = check_if_building_block(i, bb)
                if not is_bb:
                    all_is_bb = False
                    is_not_bb.append(i)
    return all_is_bb, is_not_bb