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
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
out = pd.DataFrame()
for K in Ks:
    file = f"K{subject}_dim{K}_llk_y.csv"
    res = pd.read_csv(dir_results + file, index_col=0).T
    out = pd.concat([out, res], axis=0)
# -----------------------------------------------------------------------------




# =============================================================================
# PLOT RESULTS
filename = dir_figures + "ics_y.pdf"

fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
# fig 1: Bayes factor
ax = axs[0]
ax.axhline(0, color="black", linestyle="--")
log_bf = out["log_bf_loo"]
# log_bf_se = out["log_bf_loo_se"]
K = out["K"][1:]
log_bf = log_bf.diff()[1:]
# log_bf_se = np.sqrt(log_bf_se.values[1:]**2 + log_bf_se.values[:-1]**2)
ax.plot(K, log_bf, color="black")
# ax.fill_between(K, log_bf - 2*log_bf_se, log_bf + 2*log_bf_se, color="black", alpha=0.2)
ax.set_ylabel("log(BF)")
ax.set_title("Bayes factor")
ax.set_xlabel("Comparison K v. K-1")
ax.set_xticks(K)
ax.set_xticklabels(K)
# ax.set_yscale("symlog")

# fig 2: WAIC
K = out["K"].astype(int)
ax = axs[1]
waic = out["elpd_waic"]
waic_se = out["elpd_waic_se"]
ax.plot(K, waic, color="black")
ax.fill_between(K, waic - 2*waic_se, waic + 2*waic_se, color="black", alpha=0.2)
ax.set_ylabel("elpd")
ax.set_title("WAIC")
ax.set_xlabel("Number of components")
ax.set_xticks(K)
ax.set_xticklabels(K)

# fig 3: PSIS-LOO
ax = axs[2]
loo = out["elpd_loo"]
loo_se = out["elpd_loo_se"]
ax.plot(K, loo, color="black")
ax.fill_between(K, loo - 2*loo_se, loo + 2*loo_se, color="black", alpha=0.2)
ax.set_ylabel("elpd")
ax.set_title("PSIS-LOO")
ax.set_xlabel("Number of components")
ax.set_xticks(K)
ax.set_xticklabels(K)

# # fig4: LPPD
# ax = axs[3]
# lppd = out["lppd"]
# ax.plot(K, lppd, color="black")
# ax.set_ylabel("lppd")
# ax.set_title("LPPD")
# ax.set_xlabel("Number of components")
# ax.set_xticks(K)
# ax.set_xticklabels(K)


plt.tight_layout()
plt.savefig(filename)

# -----------------------------------------------------------------------------
