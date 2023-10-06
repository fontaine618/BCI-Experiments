import sys
import os

import numpy as np
import torch
import time
import pickle
import seaborn as sns

sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci import BFFMResults
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from source.swlda.swlda import swlda_predict

plt.style.use("seaborn-v0_8-whitegrid")

from source.data.k_protocol import KProtocol


# =============================================================================
# Heterogeneity
dir = "/home/simon/Documents/BCI/experiments/tuning/"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_chains = dir + "chains/"
dir_results = dir + "posterior/"
dir_figures = dir + "figures/"

type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8
factor_samples = 10
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = True
n_samples = 100
nchars = 19

K = 8
nreps = 7
seed = 0
cor = 0.5
shrinkage = 7.
heterogeneity = [1., 2., 3., 5., 7., 10., 15., 20.]
xi_var = 1.

h_means = dict()
l_means = dict()
for h in heterogeneity:
    # load posterior
    h_mean = pd.read_csv(dir_results + f"heterogeneity{h}_hmean.csv", index_col=0).values
    l_mean = pd.read_csv(dir_results + f"heterogeneity{h}_lmean.csv", index_col=0).values
    h_means[h] = h_mean.flatten()
    l_means[h] = l_mean.flatten()

# merge together
df = pd.concat([
    pd.DataFrame({"heterogneity": h_means[h], "loading": l_means[h], "gamma": h})
    for h in heterogeneity
])
# clamp heterogeneity
df["heterogneity"] = df["heterogneity"].clip(-10., 10.)
# melt loading and heterogeneity into rows
df = df.melt(id_vars=["gamma"], value_vars=["loading", "heterogneity"])

# plot with seaborn kde plot
# two plots from variable
# x is value
# y is kde density
# hue is h

plt.cla()
g = sns.FacetGrid(
    df,
    row="variable",
    hue="gamma",
    sharex=False,
    sharey=False,
    aspect=2,
    height=2,
    palette="viridis",
    legend_out=True
)
g.map(sns.kdeplot, "value", fill=False)
g.add_legend()
# save
plt.savefig(dir_figures + "heterogeneity_posterior.pdf", bbox_inches="tight")
# -----------------------------------------------------------------------------




# =============================================================================
# Shrinkage
dir = "/home/simon/Documents/BCI/experiments/tuning/"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_chains = dir + "chains/"
dir_results = dir + "posterior/"
dir_figures = dir + "figures/"

type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8
factor_samples = 10
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = True
n_samples = 100
nchars = 19

K = 8
nreps = 7
seed = 0
cor = 0.6
shrinkage = [3., 4., 5., 7., 10.]
xi_var = 1.

h_means = dict()
s_means = dict()
for s in shrinkage:
    # load posterior
    s_mean = pd.read_csv(dir_results + f"shrinkage{s}_smean.csv", index_col=0).values
    l_mean = pd.read_csv(dir_results + f"shrinkage{s}_lmean.csv", index_col=0).values
    s_means[s] = s_mean.flatten()**0.5
    l_means[s] = (l_mean**2).sum(0)**0.5

# merge together
df = pd.concat([
    pd.DataFrame({"shrinkage_factor": s_means[s], "loading_norm": l_means[s], "shrinkage": s, "component": np.arange(8)+1})
    for s in shrinkage
])
# melt loading and heterogeneity into rows
df = df.melt(id_vars=["shrinkage", "component"], value_vars=["shrinkage_factor", "loading_norm"])


plt.cla()
g = sns.relplot(
    df,
    x="component",
    y="value",
    row="variable",
    hue="shrinkage",
    kind="line",
    aspect=2,
    height=2,
    facet_kws=dict(sharex=True, sharey=False),
)
# log scale on y axis
# g.set(yscale="log")
# save
plt.savefig(dir_figures + "shrinkage_posterior.pdf", bbox_inches="tight")
# -----------------------------------------------------------------------------