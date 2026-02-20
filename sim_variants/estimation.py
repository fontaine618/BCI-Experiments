import numpy as np
import os
import sys
import pickle
sys.path.insert(1, '/home/simon/Documents/BCI/src')
import torch
import itertools as it
from source.bffmbci import BFFMResults
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_results = "/home/simon/Documents/BCI/experiments/sim_variants/results/"
dir_chains = "/home/simon/Documents/BCI/experiments/sim_variants/chains2/"
dir_data = "/home/simon/Documents/BCI/experiments/sim_variants/data/"
dir_figures = "/home/simon/Documents/BCI/experiments/sim_variants/figures/"

os.makedirs(dir_figures, exist_ok=True)

# experiments
seeds = range(1)
Kxs = [8]
Kys = [5]
models = ["LR-DCR", "LR-DC", "LR-SC"]
K = 8

# -----------------------------------------------------------------------------

e_mu0 = dict()
e_mu1 = dict()
e_S0 = dict()
e_S1 = dict()

# combinations
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

    # error
    error = (mu0_t - mu0_e).pow(2).mean(1).sqrt()
    size = mu0_t.pow(2).mean(1).sqrt() + 1.
    error = error / size.reshape(1, -1)
    e_mu0[file_chain] = error.mean(0), error.std(0)
    error = (mu1_t - mu1_e).pow(2).mean(1).sqrt()
    size = mu1_t.pow(2).mean(1).sqrt() + 1.
    error = error / size.reshape(1, -1)
    e_mu1[file_chain] = error.mean(0), error.std(0)
    # -----------------------------------------------------------------------------


    # =============================================================================
    # FITTED COVARIANCE

    # ground truth
    S0_t = torch.einsum(
        "ek, kt, fk -> eft",
        L_t,
        b0_xi_t.exp().pow(2.),
        L_t
    ).unsqueeze(0)
    S1_t = torch.einsum(
        "ek, kt, fk -> eft",
        L_t,
        b1_xi_t.exp().pow(2.),
        L_t
    ).unsqueeze(0)
    # make Sigma_t into diagonal matrix
    Sigma_t = Sigma_t.diag_embed().unsqueeze(0).unsqueeze(-1)
    # add to S0_t and S1_t
    S0_t += Sigma_t
    S1_t += Sigma_t


    # estimated
    S0_e = torch.einsum(
        "bek, bkt, bfk -> beft",
        L_e,
        b0_xi_e.exp().pow(2.),
        L_e
    )
    S1_e = torch.einsum(
        "bek, bkt, bfk -> beft",
        L_e,
        b1_xi_e.exp().pow(2.),
        L_e
    )
    # make Sigma_e into diagonal matrix
    Sigma_e = Sigma_e.diag_embed().unsqueeze(-1)
    # add to S0_e and S1_e
    S0_e += Sigma_e
    S1_e += Sigma_e

    # error
    error = (S0_t - S0_e).pow(2).mean((1, 2)).sqrt()
    size = S0_t.pow(2).mean((1, 2)).sqrt()
    error = error / size
    e_S0[file_chain] = error.mean(0), error.std(0)
    error = (S1_t - S1_e).pow(2).mean((1, 2)).sqrt()
    size = S1_t.pow(2).mean((1, 2)).sqrt()
    error = error / size
    e_S1[file_chain] = error.mean(0), error.std(0)
    print(file_chain, S0_t.max(), S1_t.max())


# =============================================================================
# PLOT RESULTS
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# plot mean error
seed = 0
Kx = 8
Ky = 5
K = 8

fig, ax = plt.subplots(2, 3, figsize=(10, 4), sharex=True, sharey="row")
for col, mtrue in enumerate(models):
    for mfitted in models:
        file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}"
        ax[0, col].plot(e_mu0[file_chain][0].cpu().numpy(), label=mfitted)
        ax[0, col].fill_between(
            range(25),
            e_mu0[file_chain][0].cpu().numpy() - e_mu0[file_chain][1].cpu().numpy(),
            e_mu0[file_chain][0].cpu().numpy() + e_mu0[file_chain][1].cpu().numpy(),
            alpha=0.2
        )
        ax[1, col].plot(e_mu1[file_chain][0].cpu().numpy(), label=mfitted)
        ax[1, col].fill_between(
            range(25),
            e_mu1[file_chain][0].cpu().numpy() - e_mu1[file_chain][1].cpu().numpy(),
            e_mu1[file_chain][0].cpu().numpy() + e_mu1[file_chain][1].cpu().numpy(),
            alpha=0.2
        )
    ax[1, col].set_xticks([0, 6, 12, 18, 24])
    ax[1, col].set_xticklabels([0, 200, 400, 600, 800])
    ax[0, col].set_title("True model: " + mtrue)
ax[0, 0].set_ylabel("Rel. error (mean, nontarget)")
ax[1, 0].set_ylabel("Rel. error (mean, target)")
plt.legend(title="Fitted model")
plt.tight_layout()
plt.savefig(dir_figures + f"error_mean.pdf")

# plot covariance error
fig, ax = plt.subplots(2, 3, figsize=(10, 4), sharex=True, sharey="row")
for col, mtrue in enumerate(models):
    for mfitted in models:
        file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}"
        ax[0, col].plot(e_S0[file_chain][0].cpu().numpy(), label=mfitted)
        ax[0, col].fill_between(
            range(25),
            e_S0[file_chain][0].cpu().numpy() - e_S0[file_chain][1].cpu().numpy(),
            e_S0[file_chain][0].cpu().numpy() + e_S0[file_chain][1].cpu().numpy(),
            alpha=0.2
        )
        ax[1, col].plot(e_S1[file_chain][0].cpu().numpy(), label=mfitted)
        ax[1, col].fill_between(
            range(25),
            e_S1[file_chain][0].cpu().numpy() - e_S1[file_chain][1].cpu().numpy(),
            e_S1[file_chain][0].cpu().numpy() + e_S1[file_chain][1].cpu().numpy(),
            alpha=0.2
        )
    ax[1, col].set_xticks([0, 6, 12, 18, 24])
    ax[1, col].set_xticklabels([0, 200, 400, 600, 800])
    ax[0, col].set_title("True model: " + mtrue)
ax[0, 0].set_ylabel("Rel. error (cov., nontarget)")
ax[1, 0].set_ylabel("Rel. error (cov., target)")
plt.legend(title="Fitted model")
plt.tight_layout()
plt.savefig(dir_figures + f"error_covariance.pdf")
# -----------------------------------------------------------------------------
