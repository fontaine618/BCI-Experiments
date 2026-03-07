import os
import sys
sys.path.insert(0, '/home/simon/Documents/BCI')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
import itertools as it
from source.bffmbci import BFFMResults
plt.style.use('seaborn-whitegrid')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# =============================================================================
# SCRIPT: Display posterior mean functions for K, V, C experiments
#
# This script creates a figure displaying posterior mean functions for all
# components across all three types of experiments:
#   - K experiments: Varying number of latent components (K=2 to 12)
#   - V experiments: Varying covariance models (LR-DCR, LR-DC, LR-SC, CS)
#   - C experiments: Varying one-step correlation (0.35 to 0.8)
#
# Output:
#   - Figure with rows for each setting and columns for each component
#   - Empty panels for settings with fewer components than the maximum
#   - Each panel shows target (yellow) and nontarget (navy) mean responses
#
# Customizations:
#   - Change 'subject' to plot different subjects (default: "114")
#   - Modify TARGET/NONTARGET colors for different color schemes
#   - Adjust figure size by modifying figsize in plt.subplots()
# =============================================================================

# =============================================================================
# SETUP
subject = "117"
dir_chains = f"/home/simon/Documents/BCI/experiments/subject/chains/"
dir_figures = f"/home/simon/Documents/BCI/experiments/subject/figures/"
os.makedirs(dir_figures, exist_ok=True)

# Colors for target/nontarget
TARGET = "#FFCB05"
NONTARGET = "#00274C"

# Stimulus parameters from the data
stimulus_window = 800.0  # ms
n_time_points = 25  # Based on the actual mean process data
time_axis = np.linspace(0, stimulus_window, n_time_points)

# =============================================================================
# LOAD K EXPERIMENTS (K=2 to 12)
K_data = {}
for K in range(2, 13):
    file = f"K{subject}_allreps_K{K}"
    chain_file = dir_chains + file + ".chain"
    try:
        results = BFFMResults.from_files(
            [chain_file],
            warmup=0,
            thin=1
        )
        results.add_transformed_variables()
        K_data[K] = results
        print(f"Loaded K experiment with K={K}")
    except Exception as e:
        print(f"Failed to load K={K}: {e}")

# =============================================================================
# LOAD V EXPERIMENTS (LR-DCR, LR-DC, LR-SC, CS)
V_data = {}
V_list = ["LR-DCR", "LR-DC", "LR-SC",]
for V in V_list:
    file = f"K{subject}_allreps_{V}"
    chain_file = dir_chains + file + ".chain"
    try:
        results = BFFMResults.from_files(
            [chain_file],
            warmup=0,
            thin=1
        )
        results.add_transformed_variables()
        V_data[V] = results
        print(f"Loaded V experiment with V={V}")
    except Exception as e:
        print(f"Failed to load V={V}: {e}")

# =============================================================================
# LOAD C EXPERIMENTS (correlation values)
C_data = {}
C_cors = [0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
for cor in C_cors:
    file = f"K{subject}_allreps_3C{cor}"
    chain_file = dir_chains + file + ".chain"
    try:
        results = BFFMResults.from_files(
            [chain_file],
            warmup=0,
            thin=1
        )
        results.add_transformed_variables()
        C_data[cor] = results
        print(f"Loaded C experiment with cor={cor}")
    except Exception as e:
        print(f"Failed to load C={cor}: {e}")

# =============================================================================
# EXTRACT POSTERIOR MEAN FUNCTIONS
def extract_means(results):
    """Extract posterior mean functions for target and nontarget by component.

    Returns:
        mu0_components: nontarget response per component (components x time)
        mu1_components: target response per component (components x time)
        K: number of components
    """
    try:
        # Get component-wise mean processes (already integrated over channels)
        mu0_comp = results.chains["nontarget_mean_process.componentwise"].squeeze(0)  # iterations x components x time
        mu1_comp = results.chains["target_mean_process.componentwise"].squeeze(0)      # iterations x components x time

        # Average over iterations
        mu0_mean = mu0_comp.mean(0).cpu().numpy()  # components x time
        mu1_mean = mu1_comp.mean(0).cpu().numpy()  # components x time
        K = mu0_mean.shape[0]

        return mu0_mean, mu1_mean, K
    except Exception as e:
        print(f"Error extracting means: {e}")
        return None, None, None

# Extract all means
K_means = {}
for k, results in K_data.items():
    mu0, mu1, kval = extract_means(results)
    if mu0 is not None:
        K_means[k] = (mu0, mu1, kval)

V_means = {}
for v, results in V_data.items():
    mu0, mu1, kval = extract_means(results)
    if mu0 is not None:
        V_means[v] = (mu0, mu1, kval)

C_means = {}
for c, results in C_data.items():
    mu0, mu1, kval = extract_means(results)
    if mu0 is not None:
        C_means[c] = (mu0, mu1, kval)

# =============================================================================
# CREATE FIGURE WITH POSTERIOR MEAN FUNCTIONS
# Determine maximum number of components across all experiments
max_K_K = max([kval for mu0, mu1, kval in K_means.values()]) if K_means else 0
max_K_V = max([kval for mu0, mu1, kval in V_means.values()]) if V_means else 0
max_K_C = max([kval for mu0, mu1, kval in C_means.values()]) if C_means else 0
max_K = max(max_K_K, max_K_V, max_K_C)

print(f"Max K: {max_K}")

# Get one channel to plot (e.g., the first channel)
channel_idx = 0

# Create figure with rows for each setting
n_rows = len(K_means) + len(V_means) + len(C_means)
n_cols = max_K

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2.5*n_rows))
axes = axes.reshape(n_rows, n_cols) if n_rows > 1 else axes.reshape(1, n_cols)

row_idx = 0

# Plot K experiments
for k in sorted(K_means.keys()):
    mu0, mu1, kval = K_means[k]
    for comp_idx in range(n_cols):
        ax = axes[row_idx, comp_idx]

        if comp_idx < kval:
            # mu0 and mu1 are already (components x time)
            ax.plot(time_axis, mu0[comp_idx, :], color=NONTARGET, linewidth=2, label='Nontarget')
            ax.plot(time_axis, mu1[comp_idx, :], color=TARGET, linewidth=2, label='Target')
            ax.set_title(f"K={k}, Comp {comp_idx+1}", fontsize=8)
            if comp_idx == 0:
                ax.legend(fontsize=7)
        else:
            # Empty panel
            ax.set_visible(False)

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("Mean response", fontsize=8)

    row_idx += 1

# Plot V experiments
for v in V_list:
    if v in V_means:
        mu0, mu1, kval = V_means[v]
        for comp_idx in range(n_cols):
            ax = axes[row_idx, comp_idx]

            if comp_idx < kval:
                ax.plot(time_axis, mu0[comp_idx, :], color=NONTARGET, linewidth=2, label='Nontarget')
                ax.plot(time_axis, mu1[comp_idx, :], color=TARGET, linewidth=2, label='Target')
                ax.set_title(f"V={v}, Comp {comp_idx+1}", fontsize=8)
                if comp_idx == 0:
                    ax.legend(fontsize=7)
            else:
                ax.set_visible(False)

            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Time (ms)", fontsize=8)
            ax.set_ylabel("Mean response", fontsize=8)

        row_idx += 1

# Plot C experiments
for cor in sorted(C_means.keys()):
    mu0, mu1, kval = C_means[cor]
    for comp_idx in range(n_cols):
        ax = axes[row_idx, comp_idx]

        if comp_idx < kval:
            ax.plot(time_axis, mu0[comp_idx, :], color=NONTARGET, linewidth=2, label='Nontarget')
            ax.plot(time_axis, mu1[comp_idx, :], color=TARGET, linewidth=2, label='Target')
            ax.set_title(f"C={cor}, Comp {comp_idx+1}", fontsize=8)
            if comp_idx == 0:
                ax.legend(fontsize=7)
        else:
            ax.set_visible(False)

        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Time (ms)", fontsize=8)
        ax.set_ylabel("Mean response", fontsize=8)

    row_idx += 1

plt.tight_layout()
plt.savefig(dir_figures + f"K{subject}_posterior_means.pdf", bbox_inches='tight', dpi=150)
print(f"Saved figure to {dir_figures}K{subject}_posterior_means.pdf")
plt.close()

# =============================================================================
# CREATE FIGURE WITH LOG LIKELIHOOD ALONG THE CHAIN
# Extract log likelihood along chain from all experiments
K_llk_chains = {}
for k, results in K_data.items():
    try:
        # chains["log_likelihood.sum"] has shape (chains, iterations)
        llk_chain = results.chains["log_likelihood.sum"].cpu().numpy()
        K_llk_chains[k] = llk_chain
    except Exception as e:
        print(f"Failed to extract log_likelihood chain for K={k}: {e}")

V_llk_chains = {}
for v, results in V_data.items():
    try:
        llk_chain = results.chains["log_likelihood.sum"].cpu().numpy()
        V_llk_chains[v] = llk_chain
    except Exception as e:
        print(f"Failed to extract log_likelihood chain for V={v}: {e}")

C_llk_chains = {}
for c, results in C_data.items():
    try:
        llk_chain = results.chains["log_likelihood.sum"].cpu().numpy()
        C_llk_chains[c] = llk_chain
    except Exception as e:
        print(f"Failed to extract log_likelihood chain for C={c}: {e}")

# Create figure with log likelihood along chain for K experiments
if K_llk_chains:
    n_k_exp = len(K_llk_chains)
    fig, axes = plt.subplots(n_k_exp, 1, figsize=(12, 3*n_k_exp))
    if n_k_exp == 1:
        axes = [axes]

    for idx, k in enumerate(sorted(K_llk_chains.keys())):
        llk_chain = K_llk_chains[k]
        ax = axes[idx]

        # Plot each chain
        for chain_idx in range(llk_chain.shape[0]):
            ax.plot(llk_chain[chain_idx, :], linewidth=1, alpha=0.7, label=f'Chain {chain_idx+1}')

        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Log Likelihood Sum", fontsize=10)
        ax.set_title(f"K={k}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if llk_chain.shape[0] > 1:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(dir_figures + f"K{subject}_loglikelihood_chain_K.pdf", bbox_inches='tight', dpi=150)
    print(f"Saved figure to {dir_figures}K{subject}_loglikelihood_chain_K.pdf")
    plt.close()

# Create figure with log likelihood along chain for V experiments
if V_llk_chains:
    n_v_exp = len(V_llk_chains)
    fig, axes = plt.subplots(n_v_exp, 1, figsize=(12, 3*n_v_exp))
    if n_v_exp == 1:
        axes = [axes]

    for idx, v in enumerate(V_list):
        if v in V_llk_chains:
            llk_chain = V_llk_chains[v]
            ax = axes[idx]

            # Plot each chain
            for chain_idx in range(llk_chain.shape[0]):
                ax.plot(llk_chain[chain_idx, :], linewidth=1, alpha=0.7, label=f'Chain {chain_idx+1}')

            ax.set_xlabel("Iteration", fontsize=10)
            ax.set_ylabel("Log Likelihood Sum", fontsize=10)
            ax.set_title(f"V={v}", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if llk_chain.shape[0] > 1:
                ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(dir_figures + f"K{subject}_loglikelihood_chain_V.pdf", bbox_inches='tight', dpi=150)
    print(f"Saved figure to {dir_figures}K{subject}_loglikelihood_chain_V.pdf")
    plt.close()

# Create figure with log likelihood along chain for C experiments
if C_llk_chains:
    n_c_exp = len(C_llk_chains)
    fig, axes = plt.subplots(n_c_exp, 1, figsize=(12, 2.5*n_c_exp))
    if n_c_exp == 1:
        axes = [axes]

    for idx, c in enumerate(sorted(C_llk_chains.keys())):
        llk_chain = C_llk_chains[c]
        ax = axes[idx]

        # Plot each chain
        for chain_idx in range(llk_chain.shape[0]):
            ax.plot(llk_chain[chain_idx, :], linewidth=1, alpha=0.7, label=f'Chain {chain_idx+1}')

        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Log Likelihood Sum", fontsize=10)
        ax.set_title(f"C={c}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if llk_chain.shape[0] > 1:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(dir_figures + f"K{subject}_loglikelihood_chain_C.pdf", bbox_inches='tight', dpi=150)
    print(f"Saved figure to {dir_figures}K{subject}_loglikelihood_chain_C.pdf")
    plt.close()



