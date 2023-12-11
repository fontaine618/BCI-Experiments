import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import itertools as it
from source.bffmbci import BFFMResults
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import BFFModel

plt.style.use('seaborn-whitegrid')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_figures = "/home/simon/Documents/BCI/experiments/latent_dimension/figures/"
dir_results = "/home/simon/Documents/BCI/experiments/latent_dimension/results/"
dir_predict = "/home/simon/Documents/BCI/experiments/latent_dimension/predict/"
dir_chains = "/home/simon/Documents/BCI/experiments/latent_dimension/chains/"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
os.makedirs(dir_figures, exist_ok=True)
os.makedirs(dir_results, exist_ok=True)

# settings
subject = "114"

# experiments
nfolds = 3
Ks = list(range(2, 14))
folds = list(range(nfolds))

# combinations
combinations = it.product(folds, Ks)
# -----------------------------------------------------------------------------


# =============================================================================
# READ FILES
out = list()
for fold, K in combinations:
    file = f"K{subject}_dim{K}_fold{fold}.cv"
    out.append(pd.read_csv(dir_predict + file, index_col=0))
out = pd.concat(out)
out = out.loc[out["repetition"] == 5]
# -----------------------------------------------------------------------------


# =============================================================================
# PLOT RESULTS

filename = dir_figures + "cv.pdf"

# plot using seaborn
# each row is a subject
# x-axis is repetitions
# columsna re different metrics: acc, hamming, mean_entropy
# curves are the method

# first we need to melt the metrics
out = out.melt(
    id_vars=["K", "fold"],
    value_vars=["acc", "hamming", "bce", "mean_entropy"],
    var_name="metric",
    value_name="value"
)

# plot
plt.cla()
sns.set(font_scale=1.)
sns.set_style("whitegrid")
g = sns.relplot(
    data=out,
    x="K",
    y="value",
    hue="fold",
    col="metric",
    # col_wrap=3,
    kind="scatter",
    height=5,
    aspect=1,
    legend="full",
    facet_kws={
        "sharey": False,
        "sharex": True,
    }
)
g.set_titles("")
# add mean across fold
g.map(sns.lineplot, "K", "value", color="black")
# add main title
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(f"Subject {subject} CV(3) results")
for col, name in zip(g.axes[0], ["Accuracy", "Hamming", "BCE", "Entropy"]):
    col.set_ylabel(name)
g.set(xlabel="Latent dimension")
# save
plt.savefig(filename, bbox_inches="tight")
# -----------------------------------------------------------------------------