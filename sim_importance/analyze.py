import sys
import os
import torch
import time
import pandas as pd
import pickle
import itertools as it
sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import BFFModel
from source.bffmbci import BFFMResults, importance_statistic
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/experiments/sim_importance/data/"
dir_chains = "/home/simon/Documents/BCI/experiments/sim_importance/chains/"
dir_results = "/home/simon/Documents/BCI/experiments/sim_importance/results/"
dir_figures = "/home/simon/Documents/BCI/experiments/sim_importance/figures/"
os.makedirs(dir_chains, exist_ok=True)
os.makedirs(dir_results, exist_ok=True)
os.makedirs(dir_figures, exist_ok=True)

# experiments
seeds = range(3)
Kxs = [8]
Kys = [5]
Ks = [8]

# combinations
combinations = it.product(seeds, Kxs, Kys, Ks)
# -----------------------------------------------------------------------------

out = list()
corr = list()
for seed, Kx, Ky, K in combinations:
    file_true = f"Kx{Kx}_Ky{Ky}_seed{seed}"
    file = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}"
    file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"
    # truth
    variables = pickle.load(open(dir_data + file_true + ".variables", "rb"))
    Ltrue = variables["loadings"]
    Ltruestd = Ltrue / torch.norm(Ltrue, dim=0)
    # posterior
    posterior = pickle.load(open(dir_results + file + ".posterior", "rb"))
    Lpost = posterior["loadings"]
    Lpoststd = Lpost /torch.norm(Lpost, dim=0)
    # importance
    importance = pd.read_csv(dir_results + file + "_importance.csv", index_col=0)
    importance["component"] = importance["component"].astype(int)
    drop_one = pd.read_csv(dir_results + file + "_bcedrop.csv", index_col=0)
    drop_one = drop_one[drop_one["drop"]!="None"]
    drop_one["drop"] = drop_one["drop"].astype(int)
    drop_one.rename(columns={"drop": "component"}, inplace=True)
    drop_one.drop(columns=["bce"], inplace=True)
    # reordering ?
    corrmat = (Ltruestd.mT @ Lpoststd).abs()
    corr.append(corrmat)
    best_match = corrmat.argmax(0)
    # merge
    df = importance.merge(drop_one, on=["component", "Kx", "Ky", "seed", "K"])
    df["best_match"] = best_match.cpu()
    out.append(df)
df = pd.concat(out)

df["component"] += 1
df["best_match"] += 1



fig, axs = plt.subplots(3, 3, figsize=(8, 6), sharex="row",
                        gridspec_kw={'height_ratios': [4, 4, 6]})
for seed in seeds:
    df_seed = df[df["seed"]==seed]
    # importance
    sns.barplot(
        data=df_seed,
        x="component",
        y="importance",
        ax=axs[0, seed]
    )
    axs[0, seed].set_title(f"Seed {seed}")
    axs[0, seed].set_xlabel("")
    axs[0, seed].set_ylabel("Importance" if seed == 0 else "")
    # bce diff
    sns.barplot(
        data=df_seed,
        x="component",
        y="diff_bce",
        ax=axs[1, seed]
    )
    axs[1, seed].set_xlabel("")
    axs[1, seed].set_ylabel("BCE change" if seed == 0 else "")
    # corr mat
    sns.heatmap(
        corr[seed].cpu(),
        ax=axs[2, seed],
        xticklabels=range(1, 9),
        yticklabels=range(1, 9),
        vmin=0,
        vmax=1,
        cmap="coolwarm",
        cbar=seed==2
    )
    axs[2, seed].set_xlabel("Component")
    axs[2, seed].set_ylabel("Cosine similarity with \nTrue Component" if seed == 0 else "")
    for x, y in zip(df_seed["component"], df_seed["best_match"]):
        axs[2, seed].text(x-0.5, y-0.5, "x", ha="center", va="center", color="white")

plt.tight_layout()
plt.savefig(dir_figures + "importance.pdf")









# =============================================================================
# LOAD CHAIN
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file_chain],
    warmup=0,
    thin=1
)
L = results.chains["loadings"]
Lmean = L.mean((0, 1))
Lstd = L.std((0, 1))
Lmean/Lstd

# procrustes
Lref = L[0, 0, :, :]
U, S, V = torch.linalg.svd(L.mT @ Ltrue)
U = U @ V
Lrot = L @ U


Lmean = Lrot.mean((0, 1))
Lstd = Lrot.std((0, 1))
Lmean/Lstd

# varimax
Lref = _varimax(Lref)[0]
# reorder columns by column norm
Lref = Lref[:, torch.argsort(torch.norm(Lref, dim=0), descending=True)]

# -----------------------------------------------------------------------------


# =============================================================================
# LOAD POSTERIOR
posterior = pickle.load(open(dir_results + file + ".posterior", "rb"))
Lpost = posterior["loadings"]
torch.round(Lpost * 1) / 1
# -----------------------------------------------------------------------------