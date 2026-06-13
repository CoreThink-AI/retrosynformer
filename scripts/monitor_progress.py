import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)


def update(_):
    rows = [json.loads(line) for line in open("results/train_progress.jsonl")]
    idx = max(
        (i for i in range(1, len(rows)) if rows[i]["epoch"] < rows[i - 1]["epoch"]),
        default=0,
    )
    df = pd.DataFrame(rows[idx:])
    for ax in (ax1, ax2):
        ax.cla()
    ax1.plot(df.epoch, df.train_action_accuracy * 100, label="train acc")
    ax1.plot(df.epoch, df.valid_action_accuracy * 100, label="valid acc")
    ax1.set_ylabel("Action accuracy %")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(df.epoch, df.valid_loss, label="valid loss", color="red")
    ax2.set_ylabel("Valid loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)
    fig.suptitle(
        f"Epoch {df.epoch.iloc[-1]} — valid {df.valid_action_accuracy.iloc[-1]*100:.1f}%"
    )


ani = animation.FuncAnimation(fig, update, interval=30_000)
plt.tight_layout()
plt.show()
