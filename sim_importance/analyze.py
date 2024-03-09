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
seeds = range(5)
Kxs = [8]
Kys = [5]
Ks = [8]

# combinations
combinations = it.product(seeds, Kxs, Kys, Ks)
# -----------------------------------------------------------------------------

out = list()
corr = list()
for seed, Kx, Ky, K in combinations:
    file_true = f"Kx{Kx}_Ky{Ky}"
    file = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}"
    file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"
    # truth
    variables = pickle.load(open(dir_data + file_true + ".variables", "rb"))
    Ltrue = variables["loadings"]
    Ltruestd = Ltrue / torch.norm(Ltrue, dim=0)
    # posterior
    posterior = pickle.load(open(dir_results + file + ".posterior", "rb"))
    Lpost = posterior["loadings"]
    # print(k, torch.norm(Lpost, dim=0))
    Lpoststd = Lpost / torch.norm(Lpost, dim=0)
    # importance
    importance = pd.read_csv(dir_results + file + "_importance.csv", index_col=0)
    importance["component"] = importance["component"].astype(int)
    drop_one = pd.read_csv(dir_results + file + "_bcedrop.csv", index_col=0)
    drop_one["reference_drop"] = drop_one[drop_one["drop"] == "None"]["bce"].min()
    drop_one = drop_one[drop_one["drop"] != "None"]
    drop_one["drop"] = drop_one["drop"].astype(int)
    drop_one.rename(columns={"drop": "component", "diff_bce": "diff_bce_drop"}, inplace=True)
    drop_one.drop(columns=["bce"], inplace=True)
    just_one = pd.read_csv(dir_results + file + "_bcejust.csv", index_col=0)
    just_one["reference_add"] = just_one[just_one["drop"] == "None"]["bce"].min()
    just_one = just_one[just_one["drop"] != "None"]
    just_one["drop"] = just_one["drop"].astype(int)
    just_one.rename(columns={"drop": "component", "diff_bce": "diff_bce_add"}, inplace=True)
    just_one.drop(columns=["bce"], inplace=True)
    # reordering ?
    corrmat = (Ltruestd.mT @ Lpoststd).abs()
    corr.append(corrmat)
    best_match = corrmat.argmax(0)
    # merge
    df = importance.merge(drop_one, on=["component", "Kx", "Ky", "seed", "K"])
    df = df.merge(just_one, on=["component", "Kx", "Ky", "seed", "K"])
    df["best_match"] = best_match.cpu()
    out.append(df)
df = pd.concat(out)

df["component"] += 1
df["best_match"] += 1

fig, axs = plt.subplots(4, 5, figsize=(12, 8), sharex="all", sharey="row",
                        gridspec_kw={'height_ratios': [4, 4, 4, 6]})
for seed in seeds:
    df_seed = df[df["seed"] == seed]
    df_seed = df_seed.sort_values("best_match", ascending=True)
    df_seed["x"] = range(1, 9)
    # importance
    # sns.barplot(
    #     data=df_seed,
    #     x="component",
    #     y="importance",
    #     ax=axs[0, seed]
    # )
    axs[0, seed].bar(
        x=df_seed["x"]-0.5,
        height=df_seed["importance"],
        color=[f"C{i}" for i in range(8)],
        width=1.
    )
    axs[0, seed].set_title(f"Chain {seed+1}")
    axs[0, seed].set_xlabel("")
    axs[0, seed].set_ylabel("Importance" if seed == 0 else "")
    axs[0, seed].grid(axis="x")
    axs[0, seed].autoscale(tight=True, axis="x")
    # bce diff
    ref = df_seed[df_seed["component"] == 1]["reference_drop"].values[0]
    axs[1, seed].bar(
        x=df_seed["x"]-0.5,
        height=df_seed["diff_bce_drop"],
        bottom=df_seed["reference_drop"],
        color=[f"C{i}" for i in range(8)],
        width=1.
    )
    axs[1, seed].set_xlabel("")
    axs[1, seed].set_ylabel("BCE change (drop one)" if seed == 0 else "")
    axs[1, seed].set_xticks(range(1, 9))
    axs[1, seed].grid(axis="x")
    axs[1, seed].axhline(ref, color="black", linestyle="--")
    axs[1, seed].autoscale(tight=True, axis="x", enable=True, )
    # axs[1, seed].set_ylim(
    #     axs[1, seed].get_ylim()[0],
    #     axs[1, seed].get_ylim()[1] + 0.5
    # )
    # bce diff
    ref = df_seed[df_seed["component"] == 1]["reference_add"].values[0]
    axs[2, seed].bar(
        x=df_seed["x"]-0.5,
        height=df_seed["diff_bce_add"],
        bottom=df_seed["reference_add"],
        color=[f"C{i}" for i in range(8)],
        width=1.
    )
    axs[2, seed].set_xlabel("")
    axs[2, seed].set_ylabel("BCE change (add one)" if seed == 0 else "")
    axs[2, seed].set_xticks(range(1, 9))
    axs[2, seed].grid(axis="x")
    axs[2, seed].axhline(ref, color="black", linestyle="--")
    axs[2, seed].set_ylim(
        axs[2, seed].get_ylim()[0] - 0.5,
        axs[2, seed].get_ylim()[1]
    )
    axs[2, seed].autoscale(tight=True, axis="x")
    # corr mat
    sns.heatmap(
        corr[seed].cpu()[:, df_seed["component"].values-1],
        ax=axs[3, seed],
        xticklabels=range(1, 9),
        yticklabels=range(1, 9),
        vmin=0,
        vmax=1,
        cmap="coolwarm",
        cbar=False
    )
    axs[3, seed].set_xlabel("Estimated component")
    axs[3, seed].set_ylabel("Cosine similarity with \nTrue Component" if seed == 0 else "")
    for x, y in zip(df_seed["x"], df_seed["best_match"]):
        axs[3, seed].text(x - 0.5, y - 0.5, "O", ha="center", va="center", color="white")
    for k in [1, 3, 5]:
        axs[3, seed].axhline(k - 0.5, color="white", linestyle="--")
    top3 = df_seed.sort_values("diff_bce_drop", ascending=True).head(3)["x"].values
    for k in top3:
        # top three components
        axs[3, seed].axvline(k - 0.5, color="white", linestyle="--")

plt.tight_layout()
plt.savefig(dir_figures + "importance.pdf")

# =============================================================================
# Ground truth importance
Kx = 8
Ky = 5
seed = 0
file_true = f"Kx{Kx}_Ky{Ky}_seed{seed}"
variables = pickle.load(open(dir_data + file_true + ".variables", "rb"))
variables["smgp_factors.target_signal"] = \
    (1 - variables["smgp_factors.mixing_process"]) * \
    variables["smgp_factors.nontarget_process"] + \
    variables["smgp_factors.mixing_process"] * \
    variables["smgp_factors.target_process"]
variables["smgp_scaling.target_signal"] = \
    (1 - variables["smgp_scaling.mixing_process"]) * \
    variables["smgp_scaling.nontarget_process"] + \
    variables["smgp_scaling.mixing_process"] * \
    variables["smgp_scaling.target_process"]
L = variables["loadings"]  # nc x ns x E x K
beta_z1 = variables["smgp_factors.target_signal"]  # nc x ns x K x T
beta_z0 = variables["smgp_factors.nontarget_process"]
beta_xi1 = variables["smgp_scaling.target_signal"]
beta_xi0 = variables["smgp_scaling.nontarget_process"]
diff = beta_z1 * beta_xi1.exp() - beta_z0 * beta_xi0.exp()
product = torch.einsum(
    "ek, kt -> ket",
    L,
    diff
)
dnorm = diff.pow(2.).sum(-1).sqrt()
lnorm = L.pow(2.).sum(-2).sqrt()
prod = dnorm * lnorm
prod = prod.reshape(-1, prod.shape[-1])
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD CHAIN
torch.cuda.empty_cache()
Kx = 8
Ky = 5
seed = 0
K = 8
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"
results = BFFMResults.from_files(
    [dir_chains + file_chain],
    warmup=0,
    thin=1
)
L = results.chains["loadings"]
Lmean = L.mean((0, 1))
Lstd = L.std((0, 1))
Lmean / Lstd

# procrustes
Lref = L[0, 0, :, :]
U, S, V = torch.linalg.svd(L.mT @ Ltrue)
U = U @ V
Lrot = L @ U

Lmean = Lrot.mean((0, 1))
Lstd = Lrot.std((0, 1))
Lmean / Lstd

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

# find direction of differences
posterior["smgp_factors.target_signal"] = \
    (1 - posterior["smgp_factors.mixing_process"]) * \
    posterior["smgp_factors.nontarget_process"] + \
    posterior["smgp_factors.mixing_process"] * \
    posterior["smgp_factors.target_process"]
posterior["smgp_factors.difference_process"] = \
    posterior["smgp_factors.target_signal"] - \
    posterior["smgp_factors.nontarget_process"]
posterior["smgp_scaling.target_signal"] = \
    (1 - posterior["smgp_scaling.mixing_process"]) * \
    posterior["smgp_scaling.nontarget_process"] + \
    posterior["smgp_scaling.mixing_process"] * \
    posterior["smgp_scaling.target_process"]
posterior["smgp_scaling.difference_process"] = \
    posterior["smgp_scaling.target_signal"] - \
    posterior["smgp_scaling.nontarget_process"]
beta_z1 = posterior["smgp_factors.target_signal"]  # nc x ns x K x T
beta_z0 = posterior["smgp_factors.nontarget_process"]
beta_xi1 = posterior["smgp_scaling.target_signal"]
beta_xi0 = posterior["smgp_scaling.nontarget_process"]
diff = beta_z1 * beta_xi1.exp() - beta_z0 * beta_xi0.exp()
product = torch.einsum(
    "...ek, ...kt -> ...ket",
    Lpost,
    diff
)
U, S, V = torch.linalg.svd(diff)
# percent explained
S = S.cumsum(0) / S.sum()
# take top 3 (80%)
top = 3
Utop = U[:, :top]
Vtop = V[:, :top]
Ltop = Lpost @ Utop
Ltrue = variables["loadings"][:, [0, 2, 4]]
# compute projection matrices
Ptop = Ltop @ torch.linalg.inv(Ltop.mT @ Ltop) @ Ltop.mT
Ptrue = Ltrue @ torch.linalg.inv(Ltrue.mT @ Ltrue) @ Ltrue.mT
# distance
dist = (Ptop - Ptrue).pow(2.).mean()

Ltrue = variables["loadings"][:, [0, 2, 4]]
Ltruestd = Ltrue / torch.norm(Ltrue, dim=0)
Lpoststd = Lpost / torch.norm(Lpost, dim=0)

B = torch.linalg.inv(Ltruestd.mT @ Ltruestd) @ Ltruestd.mT @ Lpoststd
Proj = Ltruestd @ B
Error = (Lpoststd - Proj).pow(2.).mean(0)
# -----------------------------------------------------------------------------


# =============================================================================
# PLOT LLK ACROSS SEEDS

file_true = f"Kx{Kx}_Ky{Ky}"
variables = pickle.load(open(dir_data + file_true + ".variables", "rb"))
combinations = it.product(seeds, Kxs, Kys, Ks)
llks = dict()
for seed, Kx, Ky, K in combinations:
    file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"
    results = BFFMResults.from_files(
        [dir_chains + file_chain],
        warmup=0,
        thin=1
    )
    llks[seed] = results.chains["log_likelihood.observations"]

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
for seed, llk in llks.items():
    ax.plot(llk.squeeze(0).cpu(), alpha=0.5, color=f"C{seed}", linewidth=0.5)
    # plot smooth
    ax.plot(pd.Series(llk.squeeze(0).cpu()).rolling(25, center=True).mean(),
            label=f"Chain {seed}", color=f"C{seed}")
ax.axhline(variables["log_likelihood.observations"], color="black", linestyle="--")
ax.legend(title="", loc="lower center", ncol=5)
ax.set_xticks([0, 100, 200, 300, 400, 500])
ax.set_xticklabels([5000, 6000, 7000, 8000, 9000, 10000])
ax.set_xlabel("MCMC Iteration")
ax.set_ylabel("Log-likelihood")
plt.tight_layout()
plt.savefig(dir_figures + "llk.pdf")