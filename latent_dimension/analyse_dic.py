import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
from source.bffmbci import BFFMResults
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import BFFModel

plt.style.use('seaborn-whitegrid')
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_figures = "/home/simon/Documents/BCI/experiments/latent_dimension/figures/"
dir_results = "/home/simon/Documents/BCI/experiments/latent_dimension/results/"
dir_chains = "/home/simon/Documents/BCI/experiments/latent_dimension/chains/"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
os.makedirs(dir_figures, exist_ok=True)
os.makedirs(dir_results, exist_ok=True)

Ks = list(range(2, 13))
subject = "114"

out = pd.read_csv(dir_results + f"K{subject}_llk.csv", index_col=0)
out["elpd_loo"] *= -2
out["elpd_loo_se"] *= 2
# -----------------------------------------------------------------------------



# # =============================================================================
# # COMPUTE ICs
# out["p_mllk"] = 2 * (out["mllk_postmean"] - out["mean_mllk"])
# out["p_waic2"] = out["var_mllk"]
# out["DIC"] = -2 * out["mllk_postmean"] + 2 * out["p_mllk"]
# out["WAIC"] = -2 * out["mean_mllk"] + 2 * out["p_waic2"]
# out["Deviance"] = -2 * out["mean_mllk"]
#
# n = 285
# out["Deviance/n"] = -2 * out["mean_mllk"] / n
# out["DIC/n"] = -2 * out["mllk_postmean"] / n + 2 * out["p_mllk"]
# out["WAIC/n"] = -2 * out["mean_mllk"] / n + 2 * out["p_waic2"]
#
# out["AIC"] = -2 * out["mllk_postmean"] + 2 * n * 80 * out["K"]
# # -----------------------------------------------------------------------------



# =============================================================================
# PLOT RESULTS
filename = dir_figures + "ics.pdf"

# melt DIC and WAIC into one column
ics = out.melt(
    id_vars=["K"],
    value_vars=["DIC", "WAIC", "Deviance", "AIC"],
    var_name="name",
    value_name="value"
)
ics["type"] = "IC"
icsbyn = out.melt(
    id_vars=["K"],
    value_vars=["DIC/n", "WAIC/n", "Deviance/n"],
    var_name="name",
    value_name="value"
)
icsbyn["type"] = "IC/n"
penalties = out.melt(
    id_vars=["K"],
    value_vars=["p_mllk", "p_waic2"],
    var_name="name",
    value_name="value"
)
penalties["type"] = "Penalty"
penalties.replace({"p_mllk": "DIC", "p_waic2": "WAIC"}, inplace=True)
# merge
df = pd.concat([ics, penalties, icsbyn], axis=0)

# plot
rawdf = lambda x: x + 16*x*2 + 6*25*x

plt.cla()
g = sns.FacetGrid(
    data=df,
    row="type",
    hue="name",
    sharey=False,
    aspect=2,
    height=2,
    palette="colorblind",
    margin_titles=False,
)
g.map(sns.lineplot, "K", "value")
g.set_axis_labels("Number of components")
for ax, n in zip(g.axes.flat, ["IC value", "Est. df", "IC value"]):
    ax.set_ylabel(n)
g.axes[1][0].axline((2, rawdf(2)), (13, rawdf(13)), color="black", linestyle="--")
g.set_titles("")
g.add_legend()
# plt.show()
g.savefig(filename)
