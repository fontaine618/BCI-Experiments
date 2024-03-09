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

everys = [5, 4, 3, 2]
for every in everys:
    for method in ["_swlda", "_eegnet", ""]:
        file = f"K{subject}_every{every}{method}.test"
        df = pd.read_csv(dir_results + file, index_col=0)
        df["every"] = every
        results.append(df)
df = pd.concat(results, ignore_index=True)
# -----------------------------------------------------------------------------






# =============================================================================
# PLOT RESULTS


ncol = len(everys)
metrics = {
    "acc": "Accuracy",
    "bce": "Binary cross-entropy",
    "mean_entropy": "Mean entropy"
}
nrow = len(metrics)

fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*2), sharey="row", sharex="all")
for row, (metric, metric_name) in enumerate(metrics.items()):
    for col, every in enumerate(everys):
        ax = axes[row, col]
        sns.lineplot(
            data=df[df["every"] == every],
            x="repetition",
            y=metric,
            hue="method",
            ax=ax
        )
        e_display = {2: 8, 3:5, 4:4, 5:3}[every]
        ax.set_title(f"{e_display} Training repetitions" if row == 0 else "")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("Testing repetitions")
        if row != nrow - 1 or col != ncol - 1:
            ax.legend().remove()
        else:
            ax.legend(title="Method")
        ax.set_xticks(range(2, 13, 2))

plt.tight_layout()
plt.savefig(dir_figures + f"{subject}_every_test.pdf")
# -----------------------------------------------------------------------------