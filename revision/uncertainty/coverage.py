import numpy as np
import os
import sys
import pickle
sys.path.insert(1, '/home/simon/Documents/BCI')
import torch
import itertools as it
from source.bffmbci import BFFMResults
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')


# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/experiments/revision/uncertainty/data/"
dir_chains = "/home/simon/Documents/BCI/experiments/sim_variants/chains/"
dir_chains = "/home/simon/Documents/BCI/experiments/revision/uncertainty/chains2/"
dir_figures = "/home/simon/Documents/BCI/experiments/revision/uncertainty/figures/"

os.makedirs(dir_figures, exist_ok=True)

# experiments
seeds = range(1)
Kxs = [8]
Kys = [5]
models = ["LR-DCR", "LR-DC", "LR-SC"]
K = 8

# -----------------------------------------------------------------------------
combinations = it.product(seeds, Kxs, Kys, models, models)
for seed, Kx, Ky, mtrue, mfitted in combinations:
    # file
    file_data = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}"
    file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}"
    file_out = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}_mllk"

    # =============================================================================
    # LOAD DATA
    observations = torch.load(dir_data + file_data + ".observations")
    order = torch.load(dir_data + file_data + ".order")
    target = torch.load(dir_data + file_data + ".target")
    variables = pickle.load(open(dir_data + file_data + ".variables", "rb"))
    # -----------------------------------------------------------------------------

    # =============================================================================
    # LOAD RESULTS
    torch.cuda.empty_cache()
    results = BFFMResults.from_files(
        [dir_chains + file_chain + ".chain"],
        warmup=0,
        thin=1
    )
    results.add_transformed_variables()
    # -----------------------------------------------------------------------------

    # =============================================================================
    # FITTED MEAN

    # ground truth
    L_t = variables["loadings"]
    Sigma_t = variables["observation_variance"]
    b0_xi_t = variables["smgp_scaling.nontarget_process"]
    a1_xi_t = variables["smgp_scaling.target_process"]
    zeta_xi_t = variables["smgp_scaling.mixing_process"]
    b1_xi_t = b0_xi_t + zeta_xi_t * (a1_xi_t - b0_xi_t)
    b0_z_t = variables["smgp_factors.nontarget_process"]
    a1_z_t = variables["smgp_factors.target_process"]
    zeta_z_t = variables["smgp_factors.mixing_process"]
    b1_z_t = b0_z_t + zeta_z_t * (a1_z_t - b0_z_t)

    mu0_t = torch.einsum(
        "ek, kt, kt -> et",
        L_t,
        b0_xi_t.exp(),
        b0_z_t
    ).unsqueeze(0)
    mu1_t = torch.einsum(
        "ek, kt, kt -> et",
        L_t,
        b1_xi_t.exp(),
        b1_z_t
    ).unsqueeze(0)

    # estimated
    L_e = results.chains["loadings"].squeeze(0)
    Sigma_e = results.chains["observation_variance"].squeeze(0)
    b0_xi_e = results.chains["smgp_scaling.nontarget_process"].squeeze(0)
    b1_xi_e = results.chains["smgp_scaling.target_signal"].squeeze(0)
    b0_z_e = results.chains["smgp_factors.nontarget_process"].squeeze(0)
    b1_z_e = results.chains["smgp_factors.target_signal"].squeeze(0)

    results.chains["smgp_factors.mixing_process"].squeeze(0).mean(0)
    results.chains["smgp_scaling.mixing_process"].squeeze(0).mean(0)


    mu0_e = torch.einsum(
        "bek, bkt, bkt -> bet",
        L_e,
        b0_xi_e.exp(),
        b0_z_e
    )
    mu1_e = torch.einsum(
        "bek, bkt, bkt -> bet",
        L_e,
        b1_xi_e.exp(),
        b1_z_e
    )

    # coverage
    mu0_e_upper = torch.quantile(mu0_e, 0.975, dim=0, keepdim=True)
    mu0_e_lower = torch.quantile(mu0_e, 0.025, dim=0, keepdim=True)
    mu1_e_upper = torch.quantile(mu1_e, 0.975, dim=0, keepdim=True)
    mu1_e_lower = torch.quantile(mu1_e, 0.025, dim=0, keepdim=True)
    # mu0_e_upper = torch.max(mu0_e, dim=0, keepdim=True)[0]
    # mu0_e_lower = torch.min(mu0_e, dim=0, keepdim=True)[0]
    # mu1_e_upper = torch.max(mu1_e, dim=0, keepdim=True)[0]
    # mu1_e_lower = torch.min(mu1_e, dim=0, keepdim=True)[0]
    # mu0_e_mean = torch.mean(mu0_e, dim=0, keepdim=True)
    # mu0_e_sd = torch.std(mu0_e, dim=0, keepdim=True)
    # mu1_e_mean = torch.mean(mu1_e, dim=0, keepdim=True)
    # mu1_e_sd = torch.std(mu1_e, dim=0, keepdim=True)
    # mu0_e_lower = mu0_e_mean - 10* mu0_e_sd
    # mu0_e_upper = mu0_e_mean + 10 * mu0_e_sd
    # mu1_e_lower = mu1_e_mean - 10 * mu1_e_sd
    # mu1_e_upper = mu1_e_mean + 10 * mu1_e_sd
    coverage0 = torch.mean((mu0_t > mu0_e_lower) * (mu0_t < mu0_e_upper) + 0.0).item()
    coverage1 = torch.mean((mu1_t > mu1_e_lower) * (mu1_t < mu1_e_upper) + 0.0).item()
    width0 = torch.mean(mu0_e_upper - mu0_e_lower).item()
    width1 = torch.mean(mu1_e_upper - mu1_e_lower).item()
    print(
        f"true: {mtrue}, fitted: {mfitted}, nontarget coverage: {round(coverage0, 4)}, target coverage: {round(coverage1, 4)},"
        f"nontarget width: {round(width0, 4)}, target coverage: {round(width1, 4)}")

    # =============================================================================
    # PLOT CONFIDENCE BANDS BY CHANNEL
    n_channels = mu0_t.shape[1]
    n_time = mu0_t.shape[2]

    # Move tensors to CPU for plotting
    mu0_t_np = mu0_t.squeeze(0).cpu().numpy()
    mu1_t_np = mu1_t.squeeze(0).cpu().numpy()
    mu0_e_lower_np = mu0_e_lower.squeeze(0).cpu().numpy()
    mu0_e_upper_np = mu0_e_upper.squeeze(0).cpu().numpy()
    mu1_e_lower_np = mu1_e_lower.squeeze(0).cpu().numpy()
    mu1_e_upper_np = mu1_e_upper.squeeze(0).cpu().numpy()

    time_axis = np.arange(n_time)

    # Plot nontarget and target on same panels
    n_cols = min(4, n_channels)
    n_rows = (n_channels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
    axes = axes.flatten() if n_channels > 1 else [axes]

    for ch in range(n_channels):
        ax = axes[ch]
        # Nontarget confidence band
        ax.fill_between(time_axis, mu0_e_lower_np[ch, :], mu0_e_upper_np[ch, :],
                         alpha=0.2, color='blue', label='Nontarget 95% CI')
        # Target confidence band
        ax.fill_between(time_axis, mu1_e_lower_np[ch, :], mu1_e_upper_np[ch, :],
                         alpha=0.2, color='green', label='Target 95% CI')
        # Truth nontarget
        ax.plot(time_axis, mu0_t_np[ch, :], color='blue', linewidth=2, label='True Nontarget', marker='o',
                markersize=3, markevery=max(1, n_time//20))
        # Truth target
        ax.plot(time_axis, mu1_t_np[ch, :], color='green', linewidth=2, label='True Target', marker='s',
                markersize=3, markevery=max(1, n_time//20))
        ax.set_xlabel('Time')
        ax.set_ylabel('Mean response')
        ax.set_title(f'Channel {ch}')
        ax.grid(True, alpha=0.3)
        if ch == 0:
            ax.legend(loc='best', fontsize=8)

    # Hide extra subplots
    for ch in range(n_channels, len(axes)):
        axes[ch].set_visible(False)

    plt.tight_layout()
    plt.savefig(dir_figures + file_out + "_channels.pdf", bbox_inches='tight')
    plt.close()
    # --------------------------------------------------------------------------
