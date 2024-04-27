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
subject = "114"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = "/home/simfont/Documents/BCI/experiments/subject/chains/"
dir_figures = f"/home/simon/Documents/BCI/experiments/subject/figures/K{subject}/"
dir_results = f"/home/simon/Documents/BCI/experiments/subject/results/K{subject}/"
os.makedirs(dir_figures, exist_ok=True)
# -----------------------------------------------------------------------------








# =============================================================================
# GATHER RESULTS (K)
results = []

# experiment
seeds = list(range(10))
train_reps = [3, 5, 7]
experiment = list(it.product(seeds, train_reps))
experiment.append(("odd", 7))


for seed, treps in experiment:
    for method in [
        "_mapinit",
        "_lite_mapinit",
        "_cs",
        "_nbmn",
        "_swlda",
        "_eegnet",
        # "_lr",
        "_rf",
        "_gb",
        "_svm"
    ]:
        file = f"K{subject}_trn{treps}_seed{seed}{method}.test"
        try:
            df = pd.read_csv(dir_results + file, index_col=0)
            df["train_reps"] = treps
            df["seed"] = seed
            if method == "_nbmn":
                df["method"] = "MN-LDA"
            if method == "_cs":
                df["method"] = "SMGP(FR-CS)"
            # if method == "_swlda":
            #     df["bce"] = float("nan")
            results.append(df)
        except FileNotFoundError:
            pass
df = pd.concat(results, ignore_index=True)
# -----------------------------------------------------------------------------


# =============================================================================
# TABLE
dff = df.loc[df["repetition"] == df["train_reps"]]
summary = dff.groupby(["train_reps", "method"]).agg(
    acc_mean=("acc", "median"),
    acc_std=("acc", "std"),
    bce_mean=("bce", "median"),
    bce_std=("bce", "std"),
    n=("acc", "count")
)
summary["acc_std"] /= summary["n"].pow(0.5)
summary["bce_std"] /= summary["n"].pow(0.5)
summary["acc_mean"] *= 100
summary["acc_std"] *= 100
summary["Accuracy (%)"] = summary["acc_mean"].apply(lambda x: f"{x:.1f}") + " (" + summary["acc_std"].apply(lambda x: f"{x:.1f}") + ")"
summary["BCE"] = summary["bce_mean"].apply(lambda x: f"{x:.3f}") + " (" + summary["bce_std"].apply(lambda x: f"{x:.3f}") + ")"
summary = summary[["Accuracy (%)", "BCE"]].reset_index().set_index("method").pivot(columns="train_reps")
summary.columns = summary.columns.reorder_levels([1, 0])
summary = summary.sort_index(axis=1)
print(summary.to_latex())
# -----------------------------------------------------------------------------






# =============================================================================
# BOXPLOT RESULTS 7
metrics = {
    # "hamming": "Hamming distance",
    "acc": "Accuracy",
    "bce": "Binary cross-entropy",
    # "mean_entropy": "Mean entropy",
    # "auroc": "AuROC"
}
ncol = len(metrics)
fig, axes = plt.subplots(1, ncol, figsize=(ncol*3, 3), sharey="row", sharex="none")
for col, (metric, metric_name) in enumerate(metrics.items()):
    ax = axes[col]
    sns.boxplot(
        data=df[(df["train_reps"] == 7) * (df["repetition"] == 7) * (df["seed"] != "odd")],
        y="method",
        x=metric,
        ax=ax,
        flierprops={"marker": "o", "alpha": 0.5},
        fliersize=3,
    )
    dfeven = df[(df["train_reps"] == 7) * (df["repetition"] == 7) * (df["seed"] == "odd")]
    sns.scatterplot(
        data=dfeven,
        y="method",
        x=metric,
        ax=ax,
        color="red",
        marker="s",
        s=20,
        zorder=10
    )
    ax.set_xlabel(metric_name)
    ax.set_ylabel("Method" if not col else "")
    if metric == "bce":
        # y log scale
        ax.set_xscale("log")
    # set yticklabels font to fixed width
    for label in ax.get_yticklabels():
        label.set_fontname("monospace")
plt.tight_layout()
plt.savefig(dir_figures + f"{subject}_random_test_7boxplot.pdf")
# -----------------------------------------------------------------------------



# =============================================================================
# PLOT RESULTS

train_reps = [3, 5, 7]

ncol = len(train_reps)
metrics = {
    "hamming": "Hamming distance",
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
            # errorbar=("pi", 80),
            # estimator="median",
            errorbar=("ci", 95),
            estimator="mean",
            err_kws={"alpha": 0.1}
        )
        ax.set_title(f"{treps} Training repetitions" if row == 0 else "")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("Testing repetitions")
        ax.set_xlim(1,7)
        if row != nrow - 1:
        # if row != nrow - 1 or col != ncol - 1:
            ax.legend().remove()
        else:
            ax.legend(title="Method")
        ax.set_xticks(range(2, 8, 2))

plt.tight_layout()
plt.savefig(dir_figures + f"{subject}_random_test.pdf")
# -----------------------------------------------------------------------------