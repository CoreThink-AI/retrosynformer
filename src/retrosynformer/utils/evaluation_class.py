import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import matplotlib.colors as clr
from rdkit import Chem
from sklearn.metrics import accuracy_score, confusion_matrix

import gzip
import json
from rxnutils.routes import base, comparison
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

TEMPLATE_LIBRARY_CLASSES_PATH = "Paroutes/Uspto-2.1/Nm-class/uspto_template_library-nm.csv" # adjust path accordingly
TEMPLATE_LIBRARY_PATH = "Paroutes/Uspto-2.1/Templates-grouped/uspto_template_library.csv"  # adjust path accordingly
TEMPLATES_DF_PATH = "uspto_routes/uspto_routes_top3000.pickle"  # adjust path accordingly

class Evaluation:
    def __init__(self, result_dir):
        self.result_dir = result_dir
        if type(result_dir) == str:
            self.config = utils.read_config(result_dir + "/config.yaml")
            try:
                self.eval_df = pd.read_json(result_dir + "/predicted_routes.json")
                self.eval_df["run"] = "main"
                self.train_progress_df = pd.read_csv(result_dir + "/train_progress.csv")
            except:
                try:
                    print("trying n1")
                    routes_path = result_dir + "/predicted_routes_n1.json"
                    self.eval_df = pd.read_json(routes_path)
                    print("n1 loaded: ", len(self.eval_df))
                except:
                    try:
                        print("trying n5")
                        routes_path = result_dir + "/predicted_routes_n5.json"
                        self.eval_df = pd.read_json(routes_path)
                        print("n5 loaded: ", len(self.eval_df))
                    except:
                        print("WARNING: no file is loaded!!!")

        elif type(result_dir) == list:
            self.eval_df = pd.DataFrame({})
            for folder in result_dir:
                eval_df_tmp = pd.read_json(folder + "/predicted_routes.json")
                eval_df_tmp["run"] = folder.split("/")[-1]
                self.eval_df = pd.concat([self.eval_df, eval_df_tmp])
            self.config = utils.read_config(folder + "/config.yaml")
        if "epoch" in self.eval_df.columns:
            latest_epoch = self.eval_df["epoch"].values[-1]
            self.eval_df = pd.DataFrame(
                self.eval_df[self.eval_df["epoch"] == latest_epoch]["result"].tolist()[
                    0
                ]
            )
        elif type(self.eval_df) == list:
            self.eval_df = self.eval_df[-1]

        self.eval_df["total_pred_reward"] = [
            sum(r) if type(r) == list else None
            for r in self.eval_df["predicted_rewards"]
        ]
        self.eval_df["total_target_reward"] = [
            sum([p[0] for p in r]) if type(r) == list else None
            for r in self.eval_df["target_rewards"]
        ]
        self.building_blocks = pd.read_csv(self.config["context"]["building_blocks"])["inchi_key"].tolist()

    def read_aizynth(self, aizynth_preds, templates_df, labels=None):

        templates_df = pd.read_pickle(templates_df)
        with gzip.open(aizynth_preds, "rb") as f:
            self.aizynth_data = json.loads(f.read().decode("ascii"))

        aizynth_df = pd.DataFrame(self.aizynth_data["data"])
        aizynth_df["pred_tree"] = [trees[0] for trees in aizynth_df["trees"]]

        aizynth_df = aizynth_df[
            [
                "target",
                "search_time",
                "first_solution_time",
                "first_solution_iteration",
                "number_of_solved_routes",
                "is_solved",
                "number_of_steps",
                "pred_tree",
            ]
        ]
        if labels:
            target_labels = pd.read_csv(labels, header=None)[0]
            aizynth_df["target"] = target_labels

        print("Len aizynth; ", len(aizynth_df))
        aizynth_df = aizynth_df[
            aizynth_df["target"].isin(set(self.eval_df["target"].tolist()))
        ].reset_index()
        print("Len aizynth; ", len(aizynth_df))
        # Add actions to df
        aizy_actions = []
        for i, row in aizynth_df.iterrows():
            tree = row["pred_tree"]
            tree_list = utils.route_to_list(tree)
            actions = []
            target = tree["smiles"]
            for rxn in tree_list:
                try:
                    idx = templates_df[templates_df["hash"] == rxn["hash_corr"]].index[0]
                except:
                    try:
                        idx = templates_df[templates_df["hash"] == rxn["hash"]].index[0]
                    except:
                        print("error:", rxn.keys())
                        idx = templates_df[
                            templates_df["hash"] == rxn["metadata"]["template_hash"]
                        ].index[0]
                actions.append(idx)
            aizy_actions.append(actions)
        aizynth_df["pred_actions"] = aizy_actions

        self.aizynth_df = aizynth_df

    def plot_action_distribution(self, n=10, only_solved_targets=True, save_as=None):
        self.actions_df_target = self.eval_df[["target", "target_action_list"]]
        self.actions_df_retrosyn = self.eval_df[
            [
                "target",
                "route_solved",
                "predicted_action_list",
                "target_action_list",
                "predicted_rewards",
                "target_rewards",
            ]
        ]
        self.actions_df_aizynth = pd.merge(
            self.aizynth_df[["target", "is_solved", "pred_actions"]],
            self.actions_df_target,
            on="target",
            how="left",
        )

        lengths_target = [
            len([t for t in target if t != 0])
            for target in self.actions_df_target["target_action_list"].tolist()
            if type(target) == list
        ]
        print("Len targets: ", np.mean(lengths_target), len(lengths_target))
        if only_solved_targets:
            aizynth_targets = set(
                self.actions_df_aizynth[self.actions_df_aizynth["is_solved"] == True][
                    "target"
                ].tolist()
            )
            retrosyn_targets = set(
                self.actions_df_retrosyn[
                    self.actions_df_retrosyn["route_solved"] == True
                ]["target"].tolist()
            )
            common_targets = aizynth_targets.intersection(retrosyn_targets)

            actions_df_target_solved = self.actions_df_target[
                self.actions_df_target["target"].isin(common_targets)
            ]
            actions_df_retrosyn_solved = self.actions_df_retrosyn[
                self.actions_df_retrosyn["target"].isin(common_targets)
            ]
            actions_df_aizynth_solved = self.actions_df_aizynth[
                self.actions_df_aizynth["target"].isin(common_targets)
            ]

        self.actions_targets = [
            a
            for a in utils.flatten_list(actions_df_target_solved["target_action_list"])
            if a != 0
        ]
        self.actions_retrosyn = utils.flatten_list(
            actions_df_retrosyn_solved["predicted_action_list"].tolist()
        )
        self.actions_aizynth = utils.flatten_list(
            actions_df_aizynth_solved["pred_actions"].tolist()
        )
        c = Counter(
            [
                str(i)
                for i in self.actions_targets
                + self.actions_retrosyn
                + self.actions_aizynth
            ]
        )
        mc_actions = c.most_common()
        actions, counts = zip(*mc_actions)
        top_n_actions = actions[:n]

        # Target actions
        print("Number of target actions: ", len(self.actions_targets))
        print("Number of unique target actions: ", len(set(self.actions_targets)))
        c = Counter([str(i) for i in self.actions_targets])
        mc_actions = c.most_common()
        action_1, count_1 = zip(*mc_actions)
        df_actions = pd.DataFrame(
            {"Template id": action_1, "Count": count_1, "Routes": "Targets"}
        )

        # Retrosyn actions
        print("Number of retrosynformer actions: ", len(self.actions_retrosyn))
        print(
            "Number of unique retrosynformer actions: ", len(set(self.actions_retrosyn))
        )
        c = Counter([str(i) for i in self.actions_retrosyn])
        mc_pred = c.most_common(n)
        action_2, count_2 = zip(*mc_pred)
        df_actions_tmp = pd.DataFrame(
            {
                "Template id": action_2,
                "Count": count_2,
                "Routes": "RetroSynFormer",
                "run": "run",
            }
        )
        df_actions = pd.concat([df_actions, df_actions_tmp], ignore_index=True)

        # AiZynth Actions
        print("Number of aizynth actions: ", len(self.actions_aizynth))
        print("Number of unique aizynth actions: ", len(set(self.actions_aizynth)))
        c = Counter([str(i) for i in self.actions_aizynth])
        mc_pred = c.most_common()
        action_3, count_3 = zip(*mc_pred)
        df_actions_tmp = pd.DataFrame(
            {"Template id": action_3, "Count": count_3, "Routes": "AiZynthFinder"}
        )
        df_actions = pd.concat([df_actions, df_actions_tmp], ignore_index=True)

        fig = plt.figure(figsize=(8, 4))
        sns.barplot(
            data=df_actions[df_actions["Template id"].isin(set(top_n_actions))],
            x="Template id",
            y="Count",
            hue="Routes",
        )
        plt.xlabel("Template id")
        plt.ylabel("Count")
        plt.xticks(rotation=90)
        plt.tight_layout()
        if save_as:
            plt.savefig(save_as)
            plt.close()

    def eval_aizynth(self):

        calculator = comparison.route_distances_calculator("ted")
        aizy_teds = []

        data = self.aizynth_df
        data = data.dropna()
        print(f'Solvability = {sum(data["is_solved"]) / len(data["is_solved"])}')
        print(f'Average time = {np.mean(data["search_time"])}')
        data = pd.merge(
            data, self.eval_df[["target", "target_tree"]], on="target", how="right"
        ).dropna()
        data["route length"] = [
            tree["scores"]["number of reactions"] for tree in data["pred_tree"].tolist()
        ]
        print(
            "Average route length (solved): ",
            np.mean(data[data["is_solved"] == True]["route length"]),
        )
        for i, route in data.iterrows():
            reference = route["target_tree"]

            try:
                ref_route = base.SynthesisRoute(reference)
            except:
                ref_route = base.SynthesisRoute(reference[0])

            route_ = base.SynthesisRoute(route["pred_tree"])
            c, _ = utils.calculate_ted(
                calculator,
                base.SynthesisRoute(route_.reaction_tree),
                [base.SynthesisRoute(ref_route.reaction_tree)],
            )

            aizy_teds.append(c)

        data["TED to target"] = aizy_teds

        print(
            "Avg TED: (solved)",
            np.mean(data[data["is_solved"] == True]["TED to target"]),
        )

        print(f'Top-1 accuracy = {len(data[data["TED to target"] == 0]) / len(data)}')

    def confusion_matrix(self, labels_targets, labels_predictions, save_as=None):

        labels, counts = zip(*Counter(labels_targets).most_common())

        common = [True if i in labels else False for i in labels_targets]
        labels_targets = [i for i, c in zip(labels_targets, common) if c]
        labels_predictions = [i for i, c in zip(labels_predictions, common) if c]

        labels = sorted(set(labels_targets + labels_predictions))
        print("n_labels", len(set(labels)))

        c_matrix = confusion_matrix(labels_targets, labels_predictions)
        fig = plt.figure(figsize=(5, 4))
        sns.heatmap(
            c_matrix, xticklabels=labels, yticklabels=labels, norm=LogNorm(), cmap=cmap2
        )
        plt.tight_layout()
        if save_as:
            plt.savefig(save_as)
            plt.close()

    def calculate_action_accuracy(self, save_as=None):
        actions_df_retrosyn_valid = self.actions_df_retrosyn.dropna()
        target_actions, retrosyn_actions = utils.flatten_and_crop(
            actions_df_retrosyn_valid["target_action_list"].tolist(),
            actions_df_retrosyn_valid["predicted_action_list"].tolist(),
        )
        acc = accuracy_score(target_actions, retrosyn_actions)
        print(len(actions_df_retrosyn_valid))
        print("Action accuracy for retrosynformer: ", acc)
        try:
            target_actions_class = [
                self.idx2class[a].split(";")[-1] for a in target_actions
            ]
            retrosyn_actions_class = [
                self.idx2class[a].split(";")[-1] for a in retrosyn_actions
            ]
            acc = accuracy_score(target_actions_class, retrosyn_actions_class)
            print("Class accuracy for retrosynformer: ", acc)
            self.confusion_matrix(
                labels_targets=target_actions_class,
                labels_predictions=retrosyn_actions_class,
                save_as=save_as,
            )
        except:
            pass

        target_actions = []
        for route in actions_df_retrosyn_valid["target_action_list"].tolist():
            route = [i for i in route if i != 0]
            target_actions.append(route)
        retrosyn_actions = [
            i for i in actions_df_retrosyn_valid["predicted_action_list"].tolist()
        ]
        accuracy = [a == b for a, b in zip(target_actions, retrosyn_actions)]
        print("Ordered route accuracy is: ", sum(accuracy) / len(accuracy))

        accuracy = [set(a) == set(b) for a, b in zip(target_actions, retrosyn_actions)]
        print("Unordered route accuracy is: ", sum(accuracy) / len(accuracy))

        # AiZynthFinder
        df = self.actions_df_aizynth
        df["pred_actions"] = self.actions_df_aizynth[
            "pred_actions"
        ]
        df = df.dropna()
        print(df.head())
        target_actions, aizynth_actions = utils.flatten_and_crop(
            df["target_action_list"].tolist(), df["pred_actions"].tolist()
        )
        acc = accuracy_score(target_actions, aizynth_actions)
        print("Action accuracy for AiZynth: ", acc)

        try:
            target_actions_class = [self.idx2class[a] for a in target_actions]
            aizynth_actions_class = [self.idx2class[a] for a in aizynth_actions]
            assert len(target_actions_class) == len(aizynth_actions_class)
            acc = accuracy_score(target_actions_class, aizynth_actions_class)
            print("Class accuracy for AiZynth: ", acc)
        except:
            pass

        target_actions = []
        for route in df["target_action_list"].tolist():
            route = [i for i in route if i != 0]
            target_actions.append(route)
        retrosyn_actions = [i for i in df["pred_actions"].tolist()]
        accuracy = [a == b for a, b in zip(target_actions, retrosyn_actions)]
        print("Ordered route accuracy is: ", sum(accuracy) / len(accuracy))

        accuracy = [set(a) == set(b) for a, b in zip(target_actions, retrosyn_actions)]
        print("Unordered route accuracy is: ", sum(accuracy) / len(accuracy))

    def plot_reaction_class_distribution(
        self, n=10, only_solved_targets=True, save_as=None
    ):
        idx2class = pd.read_csv("data/classes/template_reaction_classes.csv")
        self.idx2class = {row["index"]: row["NMC2"] for _, row in idx2class.iterrows()}
        class2name = pd.read_csv("data/classes/nextmove_classes.csv")
        class2name = {
            str(row["class"]): row["name"] for _, row in class2name.iterrows()
        }
        class2name["0"] = "0 unknown"
        new_class2name = {}
        for key, value in class2name.items():
            if len(value) > 15:
                split_str = value.split(" ")
                if len((" ").join(split_str[0:3])) > 20:
                    part1 = (" ").join(split_str[0:2])
                    part2 = (" ").join(split_str[2:])
                    value = part1 + "\n" + part2
                else:
                    part1 = (" ").join(split_str[0:3])
                    part2 = (" ").join(split_str[3:])
                    value = part1 + "\n" + part2

            new_class2name[key] = value
        class2name = new_class2name

        all_predicted_classes, all_target_classes, all_aizy_classes = [], [], []
        for ta in self.actions_targets:
            all_target_classes.append(self.idx2class[ta])
        for ra in self.actions_retrosyn:
            all_predicted_classes.append(self.idx2class[ra])
        for aa in self.actions_aizynth:
            all_aizy_classes.append(self.idx2class[aa])

        print(
            "Number of unique reaction classes retrosynformer: ",
            len(set(all_predicted_classes)),
        )
        print("Number of unique reaction classes AiZynth: ", len(set(all_aizy_classes)))
        print(
            "Number of unique reaction classes targets: ", len(set(all_target_classes))
        )

        most_common_class, _ = zip(
            *Counter(
                all_target_classes + all_predicted_classes + all_aizy_classes
            ).most_common(n)
        )

        r_class, count = zip(*Counter(all_target_classes).most_common())
        count = [c for c, r in zip(count, r_class) if r in most_common_class]
        r_class = [r for r in r_class if r in most_common_class]
        df_classes = pd.DataFrame(
            {"Count": count, "Reaction class": r_class, "Routes": "Targets"}
        )

        r_class, count = zip(*Counter(all_predicted_classes).most_common())
        count = [c for c, r in zip(count, r_class) if r in most_common_class]
        r_class = [r for r in r_class if r in most_common_class]
        df_classes = pd.concat(
            (
                df_classes,
                pd.DataFrame(
                    {
                        "Count": count,
                        "Reaction class": r_class,
                        "Routes": "RetroSynFormer",
                    }
                ),
            )
        )

        r_class, count = zip(*Counter(all_aizy_classes).most_common())
        count = [c for c, r in zip(count, r_class) if r in most_common_class]
        r_class = [r for r in r_class if r in most_common_class]
        df_classes = pd.concat(
            (
                df_classes,
                pd.DataFrame(
                    {
                        "Count": count,
                        "Reaction class": r_class,
                        "Routes": "AiZynthFinder",
                    }
                ),
            )
        )

        fig = plt.figure(figsize=(8, 4))
        df_classes["Reaction class"] = [
            class2name[str(cla)] for cla in df_classes["Reaction class"]
        ]
        sns.barplot(
            df_classes, x="Reaction class", y="Count", hue="Routes"
        )
        _ = plt.xticks(rotation=90)
        plt.tight_layout()
        if save_as:
            plt.savefig(save_as)
            plt.close()

    def compare_targets(self):
        all_targets = self.eval_df['target'].tolist()

        aizynth_solved = self.aizynth_df[["target", "is_solved", "pred_tree"]].rename(
            columns={"is_solved": "aizynth_solved", "pred_tree": "aizynth_tree"}
        )
        retrosyn_solved = self.eval_df[
            ["target", "route_solved", "pred_tree", "target_tree"]
        ].rename(
            columns={"route_solved": "retrosyn_solved", "pred_tree": "retrosyn_tree"}
        )
        df_solved = aizynth_solved.merge(retrosyn_solved, on="target")
        df_solved.to_json(f"{self.result_dir}/compare_solved_route.json")

        print(
            "Total # unique targets: ",
            len(
                set(self.aizynth_df["target"].tolist()).union(
                    set(self.eval_df["target"].tolist())
                )
            ),
        )
        print(
            "Number of solved target by only aizynth: ",
            len(
                df_solved[
                    (df_solved["aizynth_solved"] == True)
                    & (df_solved["retrosyn_solved"] == False)
                ]
            ),
        )
        print(
            "Number of solved target only retrosynformer: ",
            len(
                df_solved[
                    (df_solved["aizynth_solved"] == False)
                    & (df_solved["retrosyn_solved"] == True)
                ]
            ),
        )
        print(
            "Number of solved targets by both: ",
            len(
                df_solved[
                    (df_solved["aizynth_solved"] == True)
                    & (df_solved["retrosyn_solved"] == True)
                ]
            ),
        )
        print(
            "Number of solved targets by none: ",
            len(
                df_solved[
                    (df_solved["aizynth_solved"] == False)
                    & (df_solved["retrosyn_solved"] == False)
                ]
            ),
        )
        print('Number of solved target by only aizynth: ', len(set(aizynth_solved).intersection(set(all_targets) - set(retrosyn_solved))))
        print('Number of solved target only retrosynformer: ', len(set(retrosyn_solved).intersection(set(all_targets) - set(aizynth_solved))))
        print('Number of solved targets by both: ', len(set(aizynth_solved).intersection(set(retrosyn_solved))))
        print('Number of solved targets by none: ', len(set(all_targets) - (set(aizynth_solved).union(set(retrosyn_solved)))))

        return df_solved["target"].tolist(), retrosyn_solved, aizynth_solved

# plot final results
def plot_length_distribution(df, df_aizy, save_as=None):

    df = df.dropna()
    df["n_reactions_target"] = [
        len([i for i in tree if i != 0]) for tree in df["target_action_list"]
    ]
    df["n_reactions"] = [
        len([i for i in tree if i != 0]) for tree in df["predicted_action_list"]
    ]

    new_name_pred_len = (
        f"Route length RetroSynFormer, median = {np.median(df['n_reactions'])}"
    )
    new_name_target_len = (
        f"Route length targets, median = {np.median(df['n_reactions_target'])}"
    )

    df_lengths_pred = pd.DataFrame(
        {
            "Route length": df["n_reactions_target"],
            "Route set": f"Targets",  # , \nmedian = {int(np.median(df['n_reactions_target']))}",
        }
    )
    df_lengths_target = pd.DataFrame(
        {
            "Route length": df["n_reactions"],
            "Route set": f"RetroSynFormer",  # , \nmedian = {int(np.median(df['n_reactions']))}",
        }
    )
    df_lengths = pd.concat((df_lengths_target, df_lengths_pred), ignore_index=True)

    try:
        df_aizy = df_aizy[df_aizy["is_solved"] == True]
        df_lengths_aizy = pd.DataFrame(
            {
                "Route length": df_aizy["number_of_steps"].tolist(),
                "Route set": f"AiZynthFinder",  # , \nmedian = {int(np.median(df_aizy['number_of_steps'].tolist()))}",
            }
        )
        df_lengths = pd.concat((df_lengths, df_lengths_aizy), ignore_index=True)
    except:
        pass

    fig = plt.figure(figsize=(4, 3))
    sns.countplot(
        df_lengths,
        x="Route length",
        hue="Route set",
        hue_order=["Targets", "RetroSynFormer", "AiZynthFinder"],
    )  # , multiple='stack')
    plt.ylabel("Count")
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
        plt.close()


def plot_ted_distribution(df, save_as):
    df_solved = df[df["route_solved"] == True]
    df_unsolved = df[df["route_solved"] == False]
    df_ted = pd.DataFrame(
        {
            "TED to target": df_unsolved["TED to target"],
            "Route set": f"Unsolved, median={np.round(np.median(df_unsolved['TED to target']),2)}",
        }
    )
    df_ted = df_ted.append(
        pd.DataFrame(
            {
                "TED to target": df_solved["TED to target"],
                "Route set": f"Solved, median={np.round(np.median(df_solved['TED to target']),2)}",
            }
        ),
        ignore_index=True,
    )
    fig = plt.figure(figsize=(4, 3))
    sns.countplot(
        df_ted, x="TED to target", hue="Route set", bins=10, element="step", fill=False
    )
    plt.tight_layout()
    fig.legend()
    plt.savefig(save_as)
    plt.close()

def plot_reward_distribution(df, save_as):
    print("saving reward plot as: ", save_as)
    df_rewards = pd.DataFrame(
        {
            "Route reward": df["total_target_reward"],
            "Route set": f"Targets",  # , median = {np.round(np.median(df['total_target_reward']),2)}",
        }
    )
    df_rewards = pd.concat(
        (
            df_rewards,
            pd.DataFrame(
                {
                    "Route reward": df["total_pred_reward"],
                    "Route set": f"RetroSynFormer",  # , median = {np.round(np.median(df['total_pred_reward']),2)}",
                }
            ),
        ),
        ignore_index=True,
    )

    fig = plt.figure(figsize=(4, 3))
    sns.histplot(
        df_rewards,
        x="Route reward",
        hue="Route set",
        alpha=1,
        bins=20,
        multiple="stack",
        color=["#4A8E1A", "#3889B6"],
    )
    # plt.legend()
    plt.xlim(-35, 1)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
        print("saved!")
        plt.close()
    sns.set_palette(custom_colors)
