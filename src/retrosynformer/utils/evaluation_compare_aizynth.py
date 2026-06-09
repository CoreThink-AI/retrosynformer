import argparse
from evaluation_class import (
    Evaluation,
    plot_length_distribution,
    plot_reward_distribution,
)


def main(pred_path, aizynth_results, aizynth_labels=None):

    templates_df = "/mimer/NOBACKUP/groups/naiss2023-6-290/emmagran/retrosynformer/data/templates/uspto_routes_top3000_general_templates.pickle"

    eval = Evaluation(pred_path)
    eval.read_aizynth(aizynth_results, templates_df, aizynth_labels)
    print("Evaluating AiZynthFinder")
    eval.eval_aizynth()
    print("Comparing to predictions")
    all_targets, retrosyn_solved, aizynth_solved = eval.compare_targets()
    print("Plotting...")
    plot_length_distribution(
        eval.eval_df.dropna(), eval.aizynth_df, save_as=f"{pred_path}/length_bar.png"
    )
    plot_reward_distribution(eval.eval_df.dropna(), save_as=f"{pred_path}/reward.png")
    print("Actions:")
    eval.plot_action_distribution(n=15, save_as=f"{pred_path}/reaction_templates.png")
    print("Classes:")
    #eval.plot_reaction_class_distribution(
    #    n=15, save_as=f"{pred_path}/reaction_classes.png"
    #)
    eval.calculate_action_accuracy(save_as=f"{pred_path}/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Program level arguments

    parser.add_argument("-p", "--pred_path", type=str, default=None)
    parser.add_argument("-a", "--aizynth_results", type=str, default=None)
    parser.add_argument("-l", "--aizynth_labels", type=str, default=None)

    args = parser.parse_args()
    main(
        pred_path=args.pred_path,
        aizynth_results=args.aizynth_results,
        aizynth_labels=args.aizynth_labels,
    )
