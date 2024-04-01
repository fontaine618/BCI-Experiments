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
for K in range(1, 13):
    file = f"K{subject}_allreps_K{K}"
    icx = pd.read_csv(dir_results + file + ".icx", index_col=0).T
    icx["IC"] = "x|y"
    icy = pd.read_csv(dir_results + file + ".icy", index_col=0).T
    icy["IC"] = "y|x"
    results.append(icx)
    results.append(icy)
K_results = pd.concat(results)
# -----------------------------------------------------------------------------




# =============================================================================
# GATHER RESULTS (V)
results = []
for V in ["LR-DCR", "LR-DC", "LR-SC", "CS"]:
    file = f"K{subject}_allreps_{V}"
    icx = pd.read_csv(dir_results + file + ".icx", index_col=0).T
    icx["IC"] = "x|y"
    icx["Model"] = V
    icy = pd.read_csv(dir_results + file + ".icy", index_col=0).T
    icy["IC"] = "y|x"
    icy["Model"] = V
    results.append(icx)
    results.append(icy)
V_results = pd.concat(results)
# -----------------------------------------------------------------------------




# =============================================================================
# GATHER RESULTS (C)
results = []
for cor in [0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75,]:
    file = f"K{subject}_allreps_3C{cor}"
    icx = pd.read_csv(dir_results + file + ".icx", index_col=0).T
    icx["IC"] = "x|y"
    icx["Cor"] = cor
    icy = pd.read_csv(dir_results + file + ".icy", index_col=0).T
    icy["IC"] = "y|x"
    icy["Cor"] = cor
    results.append(icx)
    results.append(icy)
C_results = pd.concat(results)
# -----------------------------------------------------------------------------




# =============================================================================
# PLOT RESULTS
fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharey="none", sharex="col")

# K selection

# IC x
ax = axes[0, 0]
which = K_results["IC"] == "x|y"
values = K_results[which]["elpd_loo"]
xs = K_results[which]["K"]
ses = K_results[which]["elpd_loo_se"]
ax.errorbar(xs, values, yerr=ses, fmt='.')
ax.set_title("LR-DCR, Correlation = 0.5")
ax.set_ylabel("PSIS-LOO-CV (x|y)")
ax.set_xticks([0, 2, 4, 6, 8, 10, 12])

# IC y
ax = axes[1, 0]
which = K_results["IC"] == "y|x"
values = K_results[which]["elpd_loo"]
xs = K_results[which]["K"]
ses = K_results[which]["elpd_loo_se"]
ax.errorbar(xs, values, yerr=ses, fmt='.')
ax.set_ylabel("PSIS-LOO-CV (y|x)")
ax.set_xlabel("Number of components")
ax.set_xticks([0, 2, 4, 6, 8, 10, 12])

# Model selection

# IC x
ax = axes[0, 1]
which = V_results["IC"] == "x|y"
values = V_results[which]["elpd_loo"]
xs = V_results[which]["Model"]
ses = V_results[which]["elpd_loo_se"]
ax.errorbar(xs, values, yerr=ses, fmt='.')
ax.set_title("K=8, Correlation = 0.5")


# IC y
ax = axes[1, 1]
which = V_results["IC"] == "y|x"
values = V_results[which]["elpd_loo"]
xs = V_results[which]["Model"]
ses = V_results[which]["elpd_loo_se"]
ax.errorbar(xs, values, yerr=ses, fmt='.')
ax.set_xlabel("Model")

# Correlation selection

# IC x
ax = axes[0, 2]
which = C_results["IC"] == "x|y"
values = C_results[which]["elpd_loo"]
xs = C_results[which]["Cor"]
ses = C_results[which]["elpd_loo_se"]
ax.errorbar(xs, values, yerr=ses, fmt='.')
ax.set_title("LR-DCR, K=8")

# IC y
ax = axes[1, 2]
which = C_results["IC"] == "y|x"
values = C_results[which]["elpd_loo"]
xs = C_results[which]["Cor"]
ses = C_results[which]["elpd_loo_se"]
ax.errorbar(xs, values, yerr=ses, fmt='.')
ax.set_xlabel("One-step correlation")
ax.set_xticks([0.4, 0.5, 0.6, 0.7])


plt.tight_layout()
plt.savefig(dir_figures + f"{subject}_selection.pdf")
# -----------------------------------------------------------------------------