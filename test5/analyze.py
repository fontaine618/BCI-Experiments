import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# SETUP
method = "posterior_mean"
dir = "/home/simon/Documents/BCI/experiments/test5/"
dir_figures = dir + "figures/"
dir_results = dir + f"results/{method}/"
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD RESULTS
results = pd.DataFrame()
for n in range(1, 16):
    df = pd.read_csv(dir_results + f"train_nrep{n:02}.csv")
    df.columns = ["testing_reps", "hamming_distance", "accuracy"]
    df["training_reps"] = n
    df["pred_set"] = "train"
    results = pd.concat([df, results],  ignore_index=True)
    df = pd.read_csv(dir_results + f"test_nrep{n:02}.csv")
    df.columns = ["testing_reps", "hamming_distance", "accuracy"]
    df["training_reps"] = n
    df["pred_set"] = "test"
    results = pd.concat([df, results],  ignore_index=True)
# -----------------------------------------------------------------------------


# =============================================================================
# PLOT RESULTS
cmap = LinearSegmentedColormap.from_list("RdYlGn", [(0, "#d73027"), (0.9, "#ffffbf"), (1, "#1a9850")])

fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex="all", sharey="all")
for row, predset in enumerate(["train", "test"]):
    df = results.loc[results["pred_set"] == predset]
    for col, metric in enumerate(["hamming_distance", "accuracy"]):
        mat = pd.pivot(df, index="training_reps", columns="testing_reps", values=metric).values
        print(mat)
        if metric == "hamming_distance":
            axs[row, col].imshow(mat, cmap=cmap.reversed())
        else:
            axs[row, col].imshow(mat, cmap=cmap)
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
    axs[row, 0].set_ylim(1.5, 14.5)
plt.tight_layout()
fig.savefig(dir_figures + f"prediction_{method}.pdf")
# -----------------------------------------------------------------------------
