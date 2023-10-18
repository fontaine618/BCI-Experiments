import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
plt.style.use('seaborn-whitegrid')



# =============================================================================
# SETUP
dir_figures = "/home/simon/Documents/BCI/experiments/latent_dimension/figures/"
dir_results = "/home/simon/Documents/BCI/experiments/latent_dimension/predict/"
dir_swlda = "/home/simon/Documents/BCI/experiments/subjects/predict/"
os.makedirs(dir_figures, exist_ok=True)

Ks = list(range(2, 13))

subject = "114"
# -----------------------------------------------------------------------------




# =============================================================================
# LOAD RESULTS

# load all KXXX.loo files
results = []
for K in Ks:
    try:
        results.append(pd.read_csv(dir_results + f"K{subject}_dim{K}.loo", index_col=0))
    except:
        pass
loo = pd.concat(results, ignore_index=True)
# -----------------------------------------------------------------------------




# =============================================================================
# PLOT RESULTS

# plot using seaborn
# each panel is a subject
# x-axis is the repetition
# y-axis is the bce
# each line/color is a drop_component

metric = "mean_entropy"
metric_label = "Entropy"
metric = "acc"
metric_label = "Accuracy"
metric = "bce"
metric_label = "BCE"
filename = dir_figures + "loo_" + metric + ".pdf"

loo["Drop"] = loo["drop_component"].apply(lambda x: "No" if x == "None" else "Yes")

# plot
g = sns.FacetGrid(
    data=loo,
    col="K",
    col_wrap=4,
    sharex=True,
    sharey=True,
    height=3,
    aspect=1,
    margin_titles=True,
)
g.map(
    sns.lineplot,
    "repetition",
    metric,
    "drop_component",
    style=loo["Drop"],
    size=loo["Drop"],
    palette="tab10",
    linewidth=1,
    alpha=0.5,
)
g.set_axis_labels("Repetition", metric_label)
g.set_titles(col_template="K = {col_name}")
if metric == "bce":
    g.set(ylim=(-100., 0))
if metric == "acc":
    g.set(ylim=(0, 1))
g.add_legend()
# plt.show()
plt.savefig(filename, bbox_inches="tight")
# -----------------------------------------------------------------------------
