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
dir_data = "/home/simon/Documents/BCI/experiments/sim_variants/data/"
dir_results = "/home/simon/Documents/BCI/experiments/sim_variants/results/"
dir_figures = "/home/simon/Documents/BCI/experiments/sim_variants/figures/"
os.makedirs(dir_figures, exist_ok=True)

# experiments
seeds = range(1)
Kxs = [8]
Kys = [5]
models = ["LR-DCR", "LR-DC", "LR-SC"]
K = 8

# -----------------------------------------------------------------------------



# =============================================================================
# GATHER RESULTS
results = []
combinations = it.product(seeds, Kxs, Kys, models, models)
for seed, Kx, Ky, mtrue, mfitted in combinations:
    file = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}"
    try:
        icx = pd.read_csv(dir_results + file + ".icx", index_col=0).T
        icx["true"] = mtrue
        icx["fitted"] = mfitted
        icx["seed"] = seed
        icx["Kx"] = Kx
        icx["Ky"] = Ky
        icx["IC"] = "x|y"
        results.append(icx)

        icy = pd.read_csv(dir_results + file + ".icy", index_col=0).T
        icy["true"] = mtrue
        icy["fitted"] = mfitted
        icy["seed"] = seed
        icy["Kx"] = Kx
        icy["Ky"] = Ky
        icy["IC"] = "y|x"
        results.append(icy)
    except:
        pass
ic = pd.concat(results)

# -----------------------------------------------------------------------------




# =============================================================================
# GATHER RESULTS TEST
results = []
combinations = it.product(seeds, Kxs, Kys, models, models)
for seed, Kx, Ky, mtrue, mfitted in combinations:
    file = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}"
    try:
        icx = pd.read_csv(dir_results + file + ".testllk", index_col=0)
        results.append(icx)
    except:
        pass
test = pd.concat(results)
# -----------------------------------------------------------------------------




# =============================================================================
# PLOT RESULTS
fig, axes = plt.subplots(2, 3, figsize=(8, 4), sharex="all", sharey="col")
for i, mtrue in enumerate(models):
    # IC
    k = 0
    metric = "elpd_loo"
    ax = axes[k, i]
    ax.set_xlim(-0.5, 2.5)
    data = ic[(ic["true"] == mtrue) & (ic["IC"] == "x|y")]
    ax.scatter(data["fitted"], data[metric])
    ax.errorbar(data["fitted"], data[metric], yerr=data[metric + "_se"], fmt='o')
    ax.set_title(f"True: {mtrue}" if k == 0 else "")
    ax.set_xlabel("Fitted" if k==1 else "")
    ax.set_ylabel("PSIS-LOO-CV " + ("(x|y)" if k==0 else "(y|x)") if i == 0 else "")
    ax.set_xticklabels(models)
    ax.grid(axis="x", visible=False)
    # Test
    k = 1
    metric = "llk"
    ax = axes[k, i]
    ax.set_xlim(-0.5, 2.5)
    data = test[(test["model_true"] == mtrue)]
    ax.scatter(data["model_fitted"], data[metric])
    ax.errorbar(data["model_fitted"], data[metric], yerr=data[metric + "_se"], fmt='o')
    ax.set_title(f"True: {mtrue}" if k == 0 else "")
    ax.set_xlabel("Fitted" if k==1 else "")
    ax.set_ylabel("Test log-likelihood" if i == 0 else "")
    ax.set_xticklabels(models)
    ax.grid(axis="x", visible=False)

plt.tight_layout()
plt.savefig(dir_figures + "main.pdf")
# -----------------------------------------------------------------------------







# =============================================================================
# PLOT RESULTS
metric = "llk"
fig, axes = plt.subplots(1, 3, figsize=(8, 2), sharex="all")
for i, mtrue in enumerate(models):
    ax = axes[i]
    ax.set_xlim(-0.5, 2.5)
    data = results[(results["model_true"] == mtrue)]
    ax.scatter(data["model_fitted"], data[metric])
    ax.errorbar(data["model_fitted"], data[metric], yerr=data[metric + "_se"], fmt='o')
    ax.set_title(f"True: {mtrue}")
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Test log-likelihood" if i == 0 else "")
    ax.set_xticklabels(models)
    ax.grid(axis="x", visible=False)
plt.tight_layout()
plt.savefig(dir_figures + "llk.pdf")
# -----------------------------------------------------------------------------




# =============================================================================
# PLOT TRUE VALUES
file = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}"
variables = pickle.load(open(dir_data + file + ".variables", "rb"))

# plot laodings
fig, ax = plt.subplots(1, 1, figsize=(4, 6))
sns.heatmap(variables["loadings"].cpu(), ax=ax, cmap="coolwarm", center=0)
ax.set_title("True loadings", fontsize=15, fontweight="bold", ha='left', x=0.)
ax.set_xlabel("Latent component")
ax.set_ylabel("Channels")
ax.set_xticklabels(range(1, 9))
ax.set_yticklabels([])
ax.set_aspect(1)
plt.tight_layout()
plt.savefig(dir_figures + "true_loadings.pdf")

# loading correlation
L = variables["loadings"].cpu()
sd = L.norm(dim=0)
L = L / sd
cor = L.T @ L
fig, ax = plt.subplots(1, 1, figsize=(4, 3.4))
sns.heatmap(cor.abs(), ax=ax, cmap="Reds", center=0.5, vmin=0., vmax=1)
ax.set_title("Loading similarity", fontsize=15, fontweight="bold", ha='left', x=0.)
ax.set_xlabel("Latent component")
ax.set_ylabel("Latent component")
ax.set_xticklabels(range(1, 9))
ax.set_yticklabels(range(1, 9))
ax.set_aspect(1)
plt.tight_layout()
plt.savefig(dir_figures + "true_loading_correlation.pdf")

# prepare variables

variables["smgp_factors.target_signal"] = \
    (1 - variables["smgp_factors.mixing_process"]) * \
    variables["smgp_factors.nontarget_process"] + \
    variables["smgp_factors.mixing_process"] * \
    variables["smgp_factors.target_process"]
variables["smgp_factors.difference_process"] = \
    variables["smgp_factors.target_signal"] - \
    variables["smgp_factors.nontarget_process"]
variables["smgp_scaling.target_signal"] = \
    (1 - variables["smgp_scaling.mixing_process"]) * \
    variables["smgp_scaling.nontarget_process"] + \
    variables["smgp_scaling.mixing_process"] * \
    variables["smgp_scaling.target_process"]
variables["smgp_scaling.difference_process"] = \
    variables["smgp_scaling.target_signal"] - \
    variables["smgp_scaling.nontarget_process"]

# target/nontarger process
target = variables["smgp_factors.target_signal"].cpu()
nontarget = variables["smgp_factors.nontarget_process"].cpu()
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
for k in range(8):
    ax.plot(target[k, :], linestyle="--", color=f"C{k}")
    ax.plot(nontarget[k, :], linestyle="-", color=f"C{k}")
ax.set_title("Factor signals", fontsize=15, fontweight="bold", ha='left', x=0.)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("")
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_xticklabels([0, 200, 400, 600, 800])
ax.legend(loc="upper right")
lines = [plt.Line2D([0], [0], color=f"C{k}") for k in range(8)]
lines += [plt.Line2D([0], [0], linestyle="-", color="black")]
lines += [plt.Line2D([0], [0], linestyle="--", color="black")]
labels = list(range(1, 9)) + ["Non-target", "Target"]
ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
plt.tight_layout()

plt.savefig(dir_figures + "true_target_nontarget.pdf")

# target/nontarger process
target = variables["smgp_scaling.target_signal"].cpu()
nontarget = variables["smgp_scaling.nontarget_process"].cpu()
fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
for k in range(8):
    ax.plot(target[k, :], linestyle="--", color=f"C{k}")
    ax.plot(nontarget[k, :], linestyle="-", color=f"C{k}")
ax.set_title("Scaling signals", fontsize=15, fontweight="bold", ha='left', x=0.)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("")
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_xticklabels([0, 200, 400, 600, 800])
ax.legend(loc="upper right")
lines = [plt.Line2D([0], [0], color=f"C{k}") for k in range(8)]
lines += [plt.Line2D([0], [0], linestyle="-", color="black")]
lines += [plt.Line2D([0], [0], linestyle="--", color="black")]
labels = list(range(1, 9)) + ["Non-target", "Target"]
ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
plt.tight_layout()
plt.savefig(dir_figures + "true_target_nontarget_scaling.pdf")

# differece process
difference = variables["smgp_factors.difference_process"].cpu()
fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
for k in range(8):
    ax.plot(difference[k, :], linestyle="-", color=f"C{k}")
ax.set_title("Difference in factor signal", fontsize=15, fontweight="bold", ha='left', x=0.)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("")
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_xticklabels([0, 200, 400, 600, 800])
plt.tight_layout()
plt.savefig(dir_figures + "true_difference.pdf")

# differece process
difference = variables["smgp_scaling.difference_process"].cpu()
fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
for k in range(8):
    ax.plot(difference[k, :], linestyle="-", color=f"C{k}")
ax.set_title("Difference in scaling signal", fontsize=15, fontweight="bold", ha='left', x=0.)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("")
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_xticklabels([0, 200, 400, 600, 800])
plt.tight_layout()
plt.savefig(dir_figures + "true_difference_scaling.pdf")

# cosine similarity of nontarget signals
L = variables["smgp_factors.nontarget_process"].cpu()
sd = L.norm(dim=1)
L = L / sd[:, None]
cor = L @ L.T
fig, ax = plt.subplots(1, 1, figsize=(4, 3.4))
sns.heatmap(cor.abs(), ax=ax, cmap="Reds", center=0.5, vmin=0., vmax=1)
ax.set_title("Signal similarity", fontsize=15, fontweight="bold", ha='left', x=0.)
ax.set_xlabel("Latent component")
ax.set_ylabel("Latent component")
ax.set_xticklabels(range(1, 9))
ax.set_yticklabels(range(1, 9))
ax.set_aspect(1)
plt.tight_layout()
plt.savefig(dir_figures + "true_nontarget_correlation.pdf")

# -----------------------------------------------------------------------------
