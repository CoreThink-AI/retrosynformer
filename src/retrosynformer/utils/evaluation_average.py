import argparse

import evaluation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_average_table(summary_paths):

    dfs = []
    for path in summary_paths:
        dfs.append(pd.read_csv(path))
    properties = [
        "Solvability",
        "Top1 Accuracy",
        "Mean time",
        "Mean TED to target",
        "Mean route length predictions",
        "Mean leafs prediction",
        "Mean branches prediction",
    ]

    for prop in properties:
        prop_values = []
        for df in dfs:
            prop_values.append(df[prop][0])
        print(
            f"{prop} = {np.round(np.mean(prop_values), 3)} \pm {np.round(np.std(prop_values), 4)}"
        )

        if prop == "Solvability":
            succesrate = prop_values
        if prop == "Top1 Accuracy":
            accuracy = prop_values
        if prop == "Mean time":
            time = prop_values
        if prop == "Mean TED to target":
            ted = prop_values
        if prop == "Mean route length predictions":
            length = prop_values

    return succesrate, accuracy, time, ted, length


def main(path, beam_widths, n_runs, n1=False, n5=False):
    print(path, beam_widths, n_runs, n1, n5)

    both = n1 and n5
    if both:
        n5 = False
    success_rate = []
    accuracy = []
    time = []
    beam_widths_all = []
    lengths = []
    teds = []
    runs = []
    for bw in beam_widths:
        if False:  # if bw == 1:
            paths = [f"{path}/run{i}/" for i in range(1, n_runs + 1)]
        else:
            paths = [
                f"{path}/run{i}/251003_predictions_bw{bw}_sorton_trajectory_prob_n1{n1}_n5{n5}/"
                for i in range(1, n_runs + 1)
            ]

        for tmp_path in paths:
            summary_file = tmp_path + 'result_summary.csv'
            if True:
                print(n1, n5)
                if n1:
                    target_set = "n1"
                elif n5:
                    target_set = "n5"
                else:
                    target_set = None
                print(target_set)
                evaluation.main(tmp_path, target_set=target_set)
        print("Results for BW=", bw)
        paths = [p + "result_summary.csv" for p in paths]
        avg_succesrate, avg_accuracy, avg_time, ted, length = get_average_table(paths)
        success_rate.extend(avg_succesrate)
        accuracy.extend(avg_accuracy)
        time.extend(avg_time)
        runs.extend(['N1' for i in range(len(avg_time))])
        beam_widths_all.extend([bw for _ in range(len(avg_time))])
        teds.extend(ted)
        lengths.extend(length)

        print("N1 = ", n1)
        print("Success rate: ", len(success_rate))
        print("Top 1 accuracy: ", len(accuracy))
        print("Avg time: ", len(time))
        print("run", len([f"run {i}, N1" for i in range(len(time))]))
        print("bw", len(beam_widths_all))

    results_df = pd.DataFrame(
        {
            "Success rate": success_rate,
            "Top-1 accuracy": accuracy,
            "Time": time,
            "TED to target": teds,
            "Predicted route length": lengths,
            "Beam width": beam_widths_all,
            "Run": runs,
        }
    )

    if both:
        n1, n5 = False, True
        success_rate = []
        accuracy = []
        time = []
        beam_widths_all = []
        runs = []
        lengths = []
        teds = []

        for bw in beam_widths:
            if False:
                paths = [f'{path}/run{i}/' for i in range(1,n_runs+1)]
            else:
                paths = [
                    f"{path}/run{i}/251003_predictions_bw{bw}_sorton_trajectory_prob_n1{n1}_n5{n5}/"
                    for i in range(1, n_runs + 1)
                ]

            for tmp_path in paths:
                summary_file = tmp_path + 'result_summary.csv'
                if True:
                    print(n1, n5)
                    if n1:
                        target_set = "n1"
                    elif n5:
                        target_set = "n5"
                    else:
                        target_set = None
                    print(target_set)
                    evaluation.main(tmp_path, target_set=target_set)
            print("Results for BW=", bw)
            paths = [p + "result_summary.csv" for p in paths]
            avg_succesrate, avg_accuracy, avg_time, ted, length = get_average_table(
                paths
            )
            success_rate.extend(avg_succesrate)
            accuracy.extend(avg_accuracy)
            time.extend(avg_time)
            runs.extend([f'N5' for i in range(len(avg_time))])
            beam_widths_all.extend([bw for _ in range(len(avg_time))])
            teds.extend(ted)
            lengths.extend(length)

        results_df_2 = pd.DataFrame(
            {
                "Success rate": success_rate,
                "Top-1 accuracy": accuracy,
                "Time": time,
                "TED to target": teds,
                "Predicted route length": lengths,
                "Beam width": beam_widths_all,
                "Run": runs,
            }
        )
        results_df = pd.concat([results_df, results_df_2])

        print("N5 = ", n5)
        print("Success rate: ", success_rate)
        print("Top 1 accuracy: ", accuracy)
        print("Avg time: ", time)
    if both:
        n1, n5 = True, True

    # if True:#not n1 and not n5:
    # Plot Success rate

    plt.figure(figsize=(4, 3))
    # plt.plot(beam_widths, success_rate, label=[f'run{i}' for i in range(1,n_runs+1)], marker='o')
    sns.lineplot(
        results_df,
        x="Beam width",
        y="Success rate",
        hue="Run",
        marker="o",
        linewidth=1.5,
    )
    plt.ylabel("Sucess rate")
    plt.xlabel("Beam width")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{path}/successrate_N1{n1}_N5{n5}.jpeg", dpi=600)

    # Plot Time
    plt.figure(figsize=(4, 3))
    # plt.plot(beam_widths, accuracy, label=[f'run{i}' for i in range(1,n_runs+1)], marker='o')
    sns.lineplot(
        results_df,
        x="Beam width",
        y="Top-1 accuracy",
        hue="Run",
        marker="o",
        linewidth=1.5,
    )
    plt.ylabel("Top-1 accuracy")
    plt.xlabel("Beam width")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{path}/top1accuracy_N1{n1}_N5{n5}.jpeg", dpi=600)

    # Plot Time
    plt.figure(figsize=(4, 3))
    # plt.plot(beam_widths, accuracy, label=[f'run{i}' for i in range(1,n_runs+1)], marker='o')
    sns.lineplot(
        results_df,
        x="Beam width",
        y="TED to target",
        hue="Run",
        marker="o",
        linewidth=1.5,
    )
    plt.ylabel("TED to target")
    plt.xlabel("Beam width")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{path}/ted_N1{n1}_N5{n5}.jpeg", dpi=600)

    # Plot accuracy
    plt.figure(figsize=(4, 3))
    # plt.plot(beam_widths, time, label=[f'run{i}' for i in range(1,n_runs+1)], marker='o')
    sns.lineplot(
        results_df, x="Beam width", y="Time", hue="Run", marker="o", linewidth=1.5
    )
    plt.ylabel("Average time / target")
    plt.xlabel("Beam width")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/time_N1{n1}_N5{n5}.jpeg", dpi=600)

    # Plot successs vs time
    plt.figure(figsize=(4, 3))
    # plt.plot(time, success_rate, label=[f'run{i}' for i in range(1,n_runs+1)], marker='o')
    sns.lineplot(
        results_df, x="Time", y="Success rate", hue="Run", marker="o", linewidth=1.5
    )
    plt.ylabel("Success rate vs. Average time")
    plt.xlabel("Avg. time (s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path}/successrate_time_N1{n1}_N5{n5}.jpeg", dpi=600)


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Program level arguments

    parser.add_argument(
        "-p",
        "--path",
        type=str,
    )
    parser.add_argument(
        "--beam_widths",
        type=list_of_ints,
    )
    parser.add_argument(
        "--n_runs",
        type=int,
    )
    parser.add_argument(
        "--n1",
        action="store_true",  # If the argument is present, n1 will be True, otherwise False
        help="Set n1 to True",
    )
    parser.add_argument(
        "--n5",
        action="store_true",  # If the argument is present, n5 will be True, otherwise False
        help="Set n5 to True",
    )
    args = parser.parse_args()
    main(
        path=args.path,
        beam_widths=args.beam_widths,
        n_runs=args.n_runs,
        n1=args.n1,
        n5=args.n5,
    )
