import numpy as np
import os
import sys
sys.path.insert(1, '/home/simon/Documents/BCI/')
import torch
import itertools as it
import pandas as pd
import pickle
from source.bffmbci import BFFMResults
from torch.distributions import Categorical
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_results = "/experiments/sim_selection/results/"
dir_data = "/experiments/sim_selection/data/"
os.makedirs(dir_results, exist_ok=True)

# experiments
seeds = range(3)
Kxs = [5, 8]
Kys = [3, 5]
Ks = range(1, 11)

# combinations
combinations = it.product(seeds, Kxs, Kys, Ks)

for seed, Kx, Ky, K in combinations:
    maxK = min(K, Kx)

    # file
    file_data = f"Kx{Kx}_Ky{Ky}_seed{seed}"
    file_test = f"Kx{Kx}_Ky{Ky}_seed{1000+seed}"
    file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.chain"

    # dimensions
    n_channels = 16
    n_characters = 19
    n_repetitions = 5
    n_stimulus = (12, 2)
    stimulus_window = 25
    stimulus_to_stimulus_interval = 5
    n_sequences = n_repetitions * n_characters

    # prediction settings
    factor_processes_method = "analytical"
    sample_mean = "harmonic"
    which_first = "sample"
    return_cumulative = False
    n_samples = 1
    factor_samples = 10

    # model
    cor = 0.95
    shrinkage = 3.
    heterogeneity = 3.
    xi_var = 0.1
    sparse = False

    settings = {
        "n_channels": n_channels,
        "stimulus_to_stimulus_interval": stimulus_to_stimulus_interval,
        "stimulus_window": stimulus_window,
        "n_stimulus": (12, 2),
        "nonnegative_smgp": False,
        "scaling_activation": "exp",
        "sparse": sparse,
        "seed": seed,
        "n_characters": n_characters,
        "n_repetitions": n_repetitions,
        "latent_dim": maxK,
        "n_timepoints": 11*stimulus_to_stimulus_interval + stimulus_window
    }

    prior_parameters = {
        "observation_variance": (1., 10.),
        "heterogeneities": heterogeneity,
        "shrinkage_factor": (1., shrinkage),
        "kernel_gp_factor_processes": (cor, 1., 1.),
        "kernel_tgp_factor_processes": (cor, 0.5, 1.),
        "kernel_gp_loading_processes": (cor, 0.1, 1.),
        "kernel_tgp_loading_processes": (cor, 0.5, 1.),
        "kernel_gp_factor": (cor, 1., 1.)
    }
    # -----------------------------------------------------------------------------


    # =============================================================================
    # LOAD DATA
    observations = torch.load(dir_data + file_data + ".observations")
    order = torch.load(dir_data + file_data + ".order")
    target = torch.load(dir_data + file_data + ".target")
    # -----------------------------------------------------------------------------



    # =============================================================================
    # LOAD RESULTS
    with open(dir_data + file_data + ".variables", "rb") as file:
        variables = pickle.load(file)

    # subset to correct latent dimension
    for k in ["loadings"]:
        variables[k] = variables[k][..., :maxK]
    for k in [
        "smgp_scaling.nontarget_process",
        "smgp_scaling.target_process",
        "smgp_scaling.mixing_process",
        "smgp_factors.nontarget_process",
        "smgp_factors.target_process",
        "smgp_factors.mixing_process"
    ]:
        variables[k] = variables[k][:maxK, ...]
    # add a dimension
    variables = {k: v.unsqueeze(0) for k, v in variables.items()}
    results = BFFMResults.single_chain(
        prior=prior_parameters,
        dimensions=settings,
        chain=variables,
        log_likelihood=torch.zeros(1)
    )
    self = results.to_predict(n_samples=n_samples)
    character_idx = torch.arange(n_characters).repeat_interleave(n_repetitions)
    # -----------------------------------------------------------------------------



    # =============================================================================
    # GET PREDICTIVE PROBABILITIES
    llk_long, _ = self.predict(
        order=order,
        sequence=observations,
        factor_samples=factor_samples,
        character_idx=character_idx,
        factor_processes_method=factor_processes_method,
        drop_component=None,
        batchsize=10
    )
    # save
    np.save(
        dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}_mllk_oracle.npy",
        llk_long.cpu().numpy()
    )
    # -----------------------------------------------------------------------------




    # =============================================================================
    # GET PREDICTIVE PROBABILITIES
    llk_long = np.load(dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}_mllk_oracle.npy")
    llk_long = torch.Tensor(llk_long)
    # -----------------------------------------------------------------------------




    # =============================================================================
    # SELECT TARGET
    nchars = 19
    nreps = 5
    # llk_long is ncahrs x nreps x 36 x nsamples
    # reshape to (nchars x nreps) x 36 x nsamples
    llk_long2 = llk_long.reshape(nchars * nreps, 36, n_samples)
    # need to pick out the target character among the 36
    target_ = target.unsqueeze(1).repeat(1, n_samples, 1)
    target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
    target36 = target36.permute(0, 2, 1)
    mllk_long = (target36 * llk_long2).sum(1)
    # -----------------------------------------------------------------------------





    # =============================================================================
    # COMPUTE ICs
    lppd_i = torch.logsumexp(mllk_long, dim=1) - np.log(n_samples)
    lppd = lppd_i.sum().item()

    # Bayes Factor through harmonic mean estimator
    llk = mllk_long
    llk_sum = llk.sum(0)
    log_bf = - torch.logsumexp(-llk_sum, dim=0).item() + np.log(n_samples)

    # store
    out = {
        "K": [K],
        "log_bf": [log_bf],
        "lppd": [lppd],
    }
    print(out)
    # -----------------------------------------------------------------------------




    # =============================================================================
    # SAVE RESULTS
    pd.DataFrame(out).T.to_csv(dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.icx_oracle")
    # -----------------------------------------------------------------------------




    # =============================================================================
    # GET PREDICTIVE PROBABILITIES
    llk_long = np.load(dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}_mllk_oracle.npy")
    llk_long = torch.Tensor(llk_long)
    # -----------------------------------------------------------------------------



    # =============================================================================
    # TRANSFORM TO BCE
    nchars = 19
    nreps = 5
    # llk_long is ncahrs x nreps x 36 x nsamples
    # reshape to (nchars x nreps) x 36 x nsamples
    llk_long2 = llk_long.reshape(nchars * nreps, 36, n_samples)
    # standardize to log probabilities
    llk_long2 = torch.log_softmax(llk_long2, dim=1)
    target_ = target.unsqueeze(1).repeat(1, n_samples, 1)
    target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
    # swap last two dimensions
    target36 = target36.permute(0, 2, 1)
    bce = (target36 * llk_long2).sum(1) # (nchars x nreps) x 36
    mllk_long = bce
    # -----------------------------------------------------------------------------



    # =============================================================================
    # COMPUTE ICs
    lppd_i = torch.logsumexp(mllk_long, dim=1) - np.log(n_samples)
    lppd = lppd_i.sum().item()

    # Bayes Factor through harmonic mean estimator
    llk = mllk_long
    llk_sum = llk.sum(0)
    log_bf = - torch.logsumexp(-llk_sum, dim=0).item() + np.log(n_samples)

    # store
    out = {
        "K": [K],
        "log_bf": [log_bf],
        "lppd": [lppd],
    }
    print(out)
    # -----------------------------------------------------------------------------



    # =============================================================================
    # SAVE RESULTS
    pd.DataFrame(out).T.to_csv(dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.icy_oracle")
    # -----------------------------------------------------------------------------


    # =============================================================================
    # AGGREGATE ACROSS REPETITIONS

    log_prob = self.aggregate(
        llk_long,
        sample_mean=sample_mean,
        which_first=which_first
    )[:, -1, :]  # nchars x 36
    target36 = target36[::nreps, :, 0]
    bce = (target36 * log_prob).sum(1)
    mllk_long = bce
    # -----------------------------------------------------------------------------

    # =============================================================================
    # COMPUTE ICs
    lppd_i = mllk_long
    lppd = lppd_i.sum().item()

    # Bayes Factor through harmonic mean estimator
    llk = mllk_long
    llk_sum = llk.sum(0)
    log_bf = - torch.logsumexp(-llk_sum, dim=0).item() + np.log(n_samples)

    # store
    out = {
        "K": [K],
        "log_bf": [log_bf],
        "lppd": [lppd],
    }
    print(out)
    # -----------------------------------------------------------------------------


    # =============================================================================
    # SAVE RESULTS
    pd.DataFrame(out).T.to_csv(dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.icyagg_oracle")
    # -----------------------------------------------------------------------------




    # =============================================================================
    # LOAD DATA
    observations = torch.load(dir_data + file_test + ".observations")
    order = torch.load(dir_data + file_test + ".order")
    target = torch.load(dir_data + file_test + ".target")

    # GET PREDICTIVE PROBABILITIES
    llk_long, _ = self.predict(
        order=order,
        sequence=observations,
        factor_samples=factor_samples,
        character_idx=character_idx,
        factor_processes_method=factor_processes_method,
        drop_component=None,
        batchsize=10
    )
    # save
    np.save(
        dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}_mllk_oracle_test.npy",
        llk_long.cpu().numpy()
    )

    # PREDICTIONS
    llk_long2 = llk_long.reshape((-1, 36, n_samples)).unsqueeze(1)

    log_prob = self.aggregate(
        llk_long2,
        sample_mean=sample_mean,
        which_first=which_first
    )

    wide_pred_one_hot = self.get_predictions(log_prob, True)

    # METRICS
    nreps = 1
    entropy = Categorical(logits=log_prob).entropy()

    target_ = target.unsqueeze(1)
    hamming = (wide_pred_one_hot != target_).double().sum(2).mean(0) / 2
    acc = (wide_pred_one_hot == target_).all(2).double().mean(0)

    target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
    N = target36.shape[0]
    bce = (target36 * log_prob).sum(-1).mean(0) * N
    bce_se = (target36 * log_prob).sum(-1).std(0).mul(np.sqrt(N))

    df = pd.DataFrame({
        "hamming": hamming.cpu(),
        "acc": acc.cpu(),
        "entropy": entropy.sum(0).cpu(),
        "bce": bce.cpu(),
        "bce_se": bce_se.cpu(),
        "repetition": range(1, nreps + 1),
        "sample_mean": sample_mean,
        "which_first": which_first,
        "method": "BFFMOracle",
        "seed": seed,
        "Kx": Kx,
        "Ky": Ky,
        "K": K,
        "dataset": "test"
    }, index=range(1, nreps + 1))
    print(df)

    df.to_csv(dir_results + f"Kx{Kx}_Ky{Ky}_seed{seed}_K{K}.testoracle")