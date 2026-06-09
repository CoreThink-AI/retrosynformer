import argparse

import matplotlib.colors as clr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem

import utils

cmap1 = clr.LinearSegmentedColormap.from_list(
    "custom blue red", ["#6BAED6", "#DF9076"], N=256
)
cmap2 = clr.LinearSegmentedColormap.from_list(
    "custom yellow purple", ["#E7CE60", "#9B76D3"], N=256
)

custom_colors = [
    "#4A8E1A",
    "#FF2A2A",
    "#3889B6",
    "#F9B32E",
    "#9A36A7",
    "#8F4A19",
    "#F67B4D",
    "#227B9B",
]
sns.set_palette(custom_colors)
sns.set_context("paper")
sns.set_style("whitegrid")
plt.tight_layout()


def read_building_blocks(building_block_path):
    return pd.read_csv(building_block_path)["inchi_key"].tolist()



# ----------- PLOT RESULTS -----------


def plot_train_progress_accuracy(train_results_path, save_as):
    sns.set()
    train_progress = pd.read_csv(train_results_path)

    fig = plt.figure()
    plt.plot(
        train_progress["epoch"],
        train_progress["train_accuracy"],
        label="train top-1",
    )
    plt.plot(
        train_progress["epoch"],
        train_progress["valid_accuracy"],
        label="valid top-1",
    )
    plt.plot(
        train_progress["epoch"],
        train_progress["train_top3_accuracy"],
        label="train top-3",
    )
    plt.plot(
        train_progress["epoch"],
        train_progress["valid_top3_accuracy"],
        label="valid top-3",
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
    plt.plot(train_progress["epoch"], train_progress["train_loss"], label="train loss")
    plt.plot(train_progress["epoch"], train_progress["valid_loss"], label="valid loss")
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


# plot final results
def plot_reward_distribution(df, save_as):

    df_rewards = pd.DataFrame(
        {
            "Route reward": df["total_pred_reward"],
            "Route set": f"Predictions, median = {np.round(np.median(df['total_pred_reward']),2)}",
        }
    )
    df_rewards = pd.concat(
        [
            df_rewards,
            pd.DataFrame(
                {
                    "Route reward": df["total_target_reward"],
                    "Route set": f"Targets, median = {np.round(np.median(df['total_target_reward']),2)}",
                }
            ),
        ],
        ignore_index=True,
    )

    fig = plt.figure()
    sns.histplot(df_rewards, x="Route reward", hue="Route set", bins=20)
    # plt.legend()
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()


def plot_length_distribution(df, save_as):

    new_name_pred_len = (
        f"Route length predictions, median = {np.median(df['len_predicted_actions'])}"
    )
    new_name_target_len = (
        f"Route length targets, median = {np.median(df['len_target_actions'])}"
    )

    df_lengths = pd.DataFrame(
        {
            "Route length": df["len_predicted_actions"],
            "Route set": f"Predictions, median = {int(np.median(df['len_predicted_actions']))}",
        }
    )
    df_lengths = pd.concat(
        [
            df_lengths,
            pd.DataFrame(
                {
                    "Route length": df["len_target_actions"],
                    "Route set": f"Targets, median = {int(np.median(df['len_target_actions']))}",
                }
            ),
        ],
        ignore_index=True,
    )

    fig = plt.figure()
    sns.histplot(df_lengths, x="Route length", hue="Route set", bins=10)
    plt.tight_layout()
    plt.savefig(save_as)
    plt.close()


def plot_ted_distribution(df, save_as):
    new_name_solved_ted = (
        f"Tree Edit Distance solved routes, median = {np.median(df['TED to target'])}"
    )
    new_name_unsolved_ted = (
        f"Tree Edit Distance unsolved routes, median = {np.median(df['TED to target'])}"
    )
    df_solved = df[df["route_solved"] == True]
    df_unsolved = df[df["route_solved"] == False]
    df_ted = pd.DataFrame(
        {
            "TED to target": df_unsolved["TED to target"],
            "Route set": f"Unsolved, median={np.round(np.median(df_unsolved['TED to target']),2)}",
        }
    )
    print(len(df_ted))
    df_ted = pd.concat(
        [
            df_ted,
            pd.DataFrame(
                {
                    "TED to target": df_solved["TED to target"],
                    "Route set": f"Solved, median={np.round(np.median(df_solved['TED to target']),2)}",
                }
            ),
        ],
        ignore_index=True,
    )
    print(len(df_ted))
    fig = plt.figure()
    sns.histplot(df_ted, x="TED to target", hue="Route set", bins=10)
    plt.tight_layout()
    fig.legend()
    plt.savefig(save_as)
    plt.close()


def get_stats_table(df):
    solvability = sum(df["route_solved"]) / len(df)
    top1_accuracy = sum(df["TED to target"] == 0) / len(df)

    min_time = min(df["time"])
    max_time = max(df["time"])
    median_time = np.median(df["time"])
    mean_time = np.mean(df["time"])

    df = df[df["route_solved"] == True]
    df.loc[:, "total_pred_reward"] = [
        sum(rewards) for rewards in df["predicted_rewards"]
    ]
    df.loc[:, "total_target_reward"] = [
        sum(rewards[0]) for rewards in df["target_rewards"]
    ]
    mean_route_reward_predictions = np.mean(df["total_pred_reward"])
    median_route_reward_predictions = np.median(df["total_pred_reward"])
    mean_route_reward_targets = np.mean(df["total_target_reward"])
    median_route_reward_targets = np.median(df["total_target_reward"])

    df.loc[:, "len_predicted_actions"] = [
        len(actions) for actions in df["predicted_action_list"]
    ]
    df.loc[:, "len_target_actions"] = [
        len([a for a in actions if a != 0]) for actions in df["target_action_list"]
    ]
    mean_route_length_predictions = np.mean(df["len_predicted_actions"])
    median_route_length_predictions = np.median(df["len_predicted_actions"])
    mean_route_length_targets = np.mean(df["len_target_actions"])
    median_route_length_targets = np.median(df["len_target_actions"])

    mean_TED_predictions = np.mean(df["TED to target"])
    median_TED_predictions = np.median(df["TED to target"])

    mean_n_leafs_prediction = np.mean([len(l) for l in df["leafs"]])
    median_n_leafs_prediction = np.median([len(l) for l in df["leafs"]])

    mean_branches_prediction = np.mean(df["n_branchings"])
    median_branches_prediction = np.median(df["n_branchings"])

    min_time_solved = min(df["time"])
    max_time_solved = max(df["time"])
    median_time_solved = np.median(df["time"])
    mean_time_solved = np.mean(df["time"])

    stats_table = pd.DataFrame(
        {
            "Solvability": solvability,
            "Top1 Accuracy": top1_accuracy,
            "Min time": min_time,
            "Max time": max_time,
            "Mean time": mean_time,
            "Median time": median_time,
            "Min time (solved targets)": min_time_solved,
            "Max time (solved targets)": max_time_solved,
            "Mean time (solved targets)": mean_time_solved,
            "Median time (solved targets)": median_time_solved,
            "Mean route reward predictions": mean_route_reward_predictions,
            "Mean route reward targets": mean_route_reward_targets,
            "Median route reward predictions": median_route_reward_predictions,
            "Median route reward targets": median_route_reward_targets,
            "Mean route length predictions": mean_route_length_predictions,
            "Mean route length targets": mean_route_length_targets,
            "Median route length predictions": median_route_reward_predictions,
            "Median route length targets": median_route_length_targets,
            "Mean TED to target": mean_TED_predictions,
            "Median TED to target": median_TED_predictions,
            "Mean leafs prediction": mean_n_leafs_prediction,
            "Median leafs prediction": median_n_leafs_prediction,
            "Mean branches prediction": mean_branches_prediction,
            "Median branches prediction": median_branches_prediction,
        },
        index=[0],
    )

    return stats_table, df


def check_route_leafs(rxn, bb):
    route_leafs = []
    all_is_bb = True
    is_not_bb = []
    # print(rxn)
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


def read_building_blocks(building_block_path):
    return pd.read_csv(building_block_path)["inchi_key"].tolist()


def check_if_building_block(smiles, building_blocks):
    mol = Chem.MolFromSmiles(smiles)
    inchi_key = Chem.inchi.MolToInchiKey(mol)
    if inchi_key in building_blocks:
        building_block = True
    else:
        building_block = False
    return building_block


def get_average_table(summary_paths):
    dfs = []
    for path in summary_paths:
        dfs.append(pd.read_csv(path))
    properties = [
        "Solvability",
        "Top1 Accuracy",
        "Mean time",
        "Mean time (solved targets)",
        "Mean TED to target",
        "Mean route length predictions",
        "Mean leafs prediction",
        "Mean branches prediction",
    ]

    for prop in properties:
        prop_values = []
        for df in dfs:
            prop_values.append(df[prop])
        print(f"{prop} = {np.mean(prop_values)} +- {np.std(prop_values)}")


def main(result_dir, file_type="json", target_set=None):
    print("in main: ", result_dir, target_set)
    config = utils.read_config(result_dir + "/config.yaml")
    if file_type == "json":
        if target_set == "n1":
            print("n1 loaded")
            routes_path = result_dir + "/predicted_routes_n1.json"
            df = pd.read_json(routes_path)
            print("n1 loaded: ", len(df))
        elif target_set == "n5":
            routes_path = result_dir + "/predicted_routes_n5.json"
            df = pd.read_json(routes_path)
            print("n5 loaded: ", len(df))
        else:
            try:
                routes_path = result_dir + "/predicted_routes.json"
                df = pd.read_json(routes_path)
                print("test targets: ", len(df))
            except:
                routes_path = result_dir + "/pred_routes_train_progress.json"
                df = pd.read_json(routes_path)
        if "epoch" in df.columns:
            latest_epoch = df["epoch"].values[-1]
            df = pd.DataFrame(df[df["epoch"] == latest_epoch]["result"].tolist()[0])
        elif type(df) == list:
            df = df[-1]
    building_blocks = read_building_blocks(config["context"]["building_blocks"])

    is_solved = []
    pred_trees_corrected = []
    for i, row in df.iterrows():
        pred_tree = row["pred_tree"]
        if type(pred_tree) == dict:
            corr_tree, tree_solved = utils.add_in_stock_property_to_trees(
                pred_tree, building_blocks
            )
            pred_trees_corrected.append(corr_tree)
            is_solved.append(tree_solved)
        else:
            pred_trees_corrected.append(None)
            is_solved.append(False)
    print(
        " solved in main",
        sum(is_solved),
        len(is_solved),
        sum(is_solved) / len(is_solved),
    )
    df.loc[:, "route_solved"] = is_solved
    # df = df.dropna()
    df.loc[:, "pred_tree"] = pred_trees_corrected

    stats_table, df = get_stats_table(df)
    stats_table.to_csv(f"{result_dir}/result_summary.csv")

    plot_length_distribution(df, f"{result_dir}/route_length_distribution.png")
    plot_reward_distribution(df, f"{result_dir}/route_reward_distribution.png")
    plot_ted_distribution(df, f"{result_dir}/route_ted_distribution.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Program level arguments
    parser.add_argument(
        "-r",
        "--result_dir",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--target_set",
        default=None,
        type=str,
    )

    args = parser.parse_args()
    main(result_dir=args.result_dir, target_set=args.target_set)
