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

# experiment
seeds = range(10)
train_reps = [3, 5, 8]
experiment = list(it.product(seeds, train_reps))


for seed, treps in experiment:
    for method in ["", "_swlda", "_eegnet", "_rf", "_gb", "_svm",]:
        file = f"K{subject}_trn{treps}_seed{seed}{method}.test"
        try:
            df = pd.read_csv(dir_results + file, index_col=0)
            df["train_reps"] = treps
            df["seed"] = seed
            if method == "_swlda":
                df["bce"] = float("nan")
                df["mean_entropy"] = float("nan")
            results.append(df)
        except FileNotFoundError:
            pass
df = pd.concat(results, ignore_index=True)
# -----------------------------------------------------------------------------






# =============================================================================
# PLOT RESULTS

train_reps = [3, 5, 8]

ncol = len(train_reps)
metrics = {
    "acc": "Accuracy",
    "bce": "Binary cross-entropy",
    # "mean_entropy": "Mean entropy",
    "auroc": "AuROC"
}
nrow = len(metrics)

fig, axes = plt.subplots(nrow, ncol, figsize=(ncol*3, nrow*2), sharey="row", sharex="all")
for row, (metric, metric_name) in enumerate(metrics.items()):
    for col, treps in enumerate(train_reps):
        # ax = axes[col]
        ax = axes[row, col]
        crdf = df[df["train_reps"] == treps]

        sns.lineplot(
            data=crdf,
            x="repetition",
            y=metric,
            hue="method",
            style="method",
            ax=ax,
            errorbar=("pi", 100),
            estimator="median",
            # errorbar=("ci", 95),
            # estimator="mean",
            err_kws={"alpha": 0.1}
        )
        ax.set_title(f"{treps} Training repetitions" if row == 0 else "")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("Testing repetitions")
        if row != nrow - 1 or col != ncol - 1:
            ax.legend().remove()
        else:
            ax.legend(title="Method")
        ax.set_xticks(range(2, 13, 2))

plt.tight_layout()
plt.savefig(dir_figures + f"{subject}_random_test.pdf")
# -----------------------------------------------------------------------------