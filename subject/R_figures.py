import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import pickle
import itertools as it
from source.bffmbci import add_transformed_variables
plt.style.use('seaborn-whitegrid')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SETUP
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = "/home/simfont/Documents/BCI/experiments/subject/chains/"
dir_figures = "/home/simon/Documents/BCI/experiments/subject/figures/"
dir_results = "/home/simon/Documents/BCI/experiments/subject/results/"
os.makedirs(dir_figures, exist_ok=True)

# experiments
subject = "114"
# -----------------------------------------------------------------------------








# =============================================================================
# GATHER RESULTS (K)
results = []
for train_reps in range(3, 10):
    for method in ["_swlda", "_eegnet", ""]:
        file = f"K{subject}_{train_reps}reps{method}.test"
        df = pd.read_csv(dir_results + file, index_col=0)
        results.append(df)
df = pd.concat(results)
# -----------------------------------------------------------------------------






# =============================================================================
# PLOT RESULTS


which_train_reps = [3, 5, 7, 9]
ncol = len(which_train_reps)
metrics = {
    "acc": "Accuracy",
    "bce": "Binary cross-entropy",
    "mean_entropy": "Mean entropy"
}
nrow = len(metrics)

fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*2), sharey="row", sharex="all")
for row, (metric, metric_name) in enumerate(metrics.items()):
    for col, train_reps in enumerate(which_train_reps):
        ax = axes[row, col]
        sns.lineplot(
            data=df[df["training_reps"] == train_reps],
            x="repetition",
            y=metric,
            hue="method",
            ax=ax
        )
        ax.set_title(f"{train_reps} Training repetitions" if row == 0 else "")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("Testing repetitions")
        if row != nrow - 1 or col != ncol - 1:
            ax.legend().remove()
        ax.set_xticks(range(2, 13, 2))

plt.tight_layout()
plt.savefig(dir_figures + f"{subject}_test.pdf")
# -----------------------------------------------------------------------------