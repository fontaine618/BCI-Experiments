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

# file
type = "TRN"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

# model
seed = 0
n_iter = 20_000
cor = 0.8
shrinkage = 3.
heterogeneity = 3.
xi_var = 0.003
sparse = False

# prediction settings
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = False
n_samples = 100
factor_samples = 10
# -----------------------------------------------------------------------------


out = dict()
for K in Ks:

    # =============================================================================
    # LOAD DATA
    eeg = KProtocol(
        filename=filename,
        type=type,
        subject=subject,
        session=session,
        window=window,
        bandpass_window=bandpass_window,
        bandpass_order=bandpass_order,
        downsample=downsample,
    )
    # -----------------------------------------------------------------------------




    # =============================================================================
    # INITIALIZE MODEL
    settings = {
        "latent_dim": K,
        "n_channels": eeg.sequence.shape[1],
        "stimulus_to_stimulus_interval": eeg.stimulus_to_stimulus_interval,
        "stimulus_window": eeg.stimulus_window,
        "n_stimulus": (12, 2),
        "n_sequences": eeg.sequence.shape[0],
        "nonnegative_smgp": False,
        "scaling_activation": "exp",
        "sparse": sparse,
        "seed": seed
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

    model = BFFModel(
        sequences=eeg.sequence,
        stimulus_order=eeg.stimulus_order,
        target_stimulus=eeg.target,
        **settings,
        **prior_parameters
    )
    # -----------------------------------------------------------------------------





    # =============================================================================
    # LOAD RESULTS

    file = f"K{subject}_dim{K}.chain"
    torch.cuda.empty_cache()
    results = BFFMResults.from_files(
        [dir_chains + file],
        warmup=0,
        thin=1
    )
    llk = results.chains["log_likelihood.observations"]
    llk_mean = llk.mean().item()

    # get posterior mean
    variables = {
        k: v.mean(dim=(0, 1))
        for k, v in results.chains.items()
    }
    del variables["log_likelihood.observations"]

    # get marginal likelihood for posterior samples
    # and for the mean (last in the list)
    predobj = results.to_predict(n_samples=n_samples)

    # append posterior mean to the list
    for k, v in variables.items():
        if k in predobj.variables.keys():
            predobj.variables[k] = torch.cat([predobj.variables[k], v[None, ...]], dim=0)

    order = eeg.stimulus_order
    sequence = eeg.sequence
    target = eeg.target
    character_idx = eeg.character_idx
    llk_long = predobj.marginal_log_likelihood(
        order=order,
        sequence=sequence,
        target=target,
        batch_size=10 if K > 8 else 25
    )
    mllk = llk_long.sum(0)
    # all but last one are posterior samples to average over
    mllk_mean = mllk[:-1].mean().item()
    # last one is at posterior mean
    mllk_posterior_mean = mllk[-1].item()


    # get likelihood at posterior mean using posterior mean for latent factors

    # set posterior mean
    model.set(**variables)

    # get likelihood at posterior mean
    # to this end, we need to update the local variables
    model.generate_local_variables()

    model.variables["factor_processes"].data = \
        model.variables["factor_processes"].posterior_mean_by_conditionals

    model.variables["observations"].store_log_density()
    llk_postmean = model.variables["observations"].log_density_history[-1]



    # # get likelihood at posterior mean using posterior samples for latent factors
    #
    # # set posterior mean
    # model.set(**variables)
    #
    # # get likelihood at posterior mean
    # # to this end, we need to update the local variables
    # model.generate_local_variables()
    #
    # for iter in range(1000 if K > 8 else 500):
    #     model.variables["factor_processes"].sample()
    #     model.variables["observations"].store_log_density()
    #     if iter % 100 == 0:
    #         print(iter, model.variables["observations"].log_density_history[-1])
    #     llk_postmean = torch.Tensor(model.variables["observations"].log_density_history[-100:]).mean().item()


    out[K] = {"mean_mllk": mllk_mean, "mllk_postmean": mllk_posterior_mean,
              "mean_llk": llk_mean, "llk_postmean": llk_postmean}

    print(K, mllk_mean, mllk_posterior_mean, llk_mean, llk_postmean)
    # -----------------------------------------------------------------------------

    # =============================================================================
    # SAVE RESULTS
    pd.DataFrame(out).T.to_csv(dir_results + f"K{subject}_llk.csv")
    # -----------------------------------------------------------------------------