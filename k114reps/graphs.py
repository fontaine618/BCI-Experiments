import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# SETUP
dir = "/experiments/k114reps/"
dir_results = dir + f"results/"
results = pd.read_csv(dir_results + f"train_test_nreps.csv")
cmap = LinearSegmentedColormap.from_list("RdYlGn", [(0, "#d73027"), (0.7, "#ffffbf"), (1, "#1a9850")])
# -----------------------------------------------------------------------------



# =============================================================================
# PLOT RESULTS
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex="all", sharey="all")
for row, predset in enumerate(["training", "testing"]):
    df = results.loc[results["dataset"] == predset]
    for col, metric in enumerate(["acc", "hamming"]):
        mat = pd.pivot(df, index="training_reps", columns="repetition", values=metric).values
        if metric == "hamming":
            axs[row, col].imshow(mat, cmap=cmap.reversed())
        else:
            axs[row, col].imshow(mat, cmap=cmap)
        for _, dfrow in df.iterrows():
            axs[row, col].text(
                dfrow["repetition"] - 1,
                dfrow["training_reps"] - 1,
                int(dfrow[metric]),
                ha="center", va="center"
            )
        axs[row, col].invert_yaxis()
        axs[row, col].set_xticks(range(0, 15))
        axs[row, col].set_yticks(range(0, 15))
        axs[row, col].set_xticklabels(range(1, 16))
        axs[row, col].set_yticklabels(range(1, 16))
        axs[row, col].grid(None)
        fig.colorbar(axs[row, col].get_images()[0], ax=axs[row, col],
                     label=metric.replace("_", " ").capitalize())
        if row == 1:
            axs[row, col].set_xlabel("Testing repetitions")
        axs[row, col].set_title(predset.capitalize())
    axs[row, 0].set_ylabel("Training repetitions")
    axs[row, 0].set_ylim(-0.5, 14.5)
    axs[row, 0].set_xlim(-0.5, 14.5)
plt.tight_layout()
fig.savefig(dir_results + f"prediction_metrics.pdf")
# -----------------------------------------------------------------------------
