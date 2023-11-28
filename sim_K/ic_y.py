import pandas as pd
import numpy as np
import os
import sys
import torch
import arviz as az
import itertools as it
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
from source.bffmbci import BFFMResults
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import BFFModel
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_results = "/home/simfont/Documents/BCI/experiments/sim_K/results/"
dir_chains = "/home/simfont/Documents/BCI/experiments/sim_K/chains/"
dir_data = "/home/simfont/Documents/BCI/experiments/sim_K/data/"
os.makedirs(dir_results, exist_ok=True)

# experiments
seeds = range(10)
Ktrues = [2, 5, 8]
Ks = range(2, 11)

# combinations
combinations = it.product(seeds, Ktrues, Ks)

# current experiment from command line
i = int(sys.argv[1])
seed, Ktrue, K = list(combinations)[i]

# file
file_data = f"dim{Ktrue}_seed{seed}"
file_chain = f"dim{Ktrue}_K{K}_seed{seed}.chain"

# model
n_iter = 10_000
cor = 0.8
shrinkage = 3.
heterogeneity = 3.
xi_var = 0.003
sparse = False

# dimensions
n_characters = 19
n_repetitions = 15
n_channels = 16
stimulus_window = 55
stimulus_to_stimulus_interval = 10

# prediction settings
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = False
n_samples = 500
factor_samples = 10
# -----------------------------------------------------------------------------





# =============================================================================
# LOAD DATA
observations = torch.load(dir_data + file_data + ".observations")
order = torch.load(dir_data + file_data + ".order")
target = torch.load(dir_data + file_data + ".target")
# -----------------------------------------------------------------------------



# =============================================================================
# INITIALIZE MODEL
settings = {
    "latent_dim": K,
    "n_channels": observations.shape[1],
    "stimulus_to_stimulus_interval": stimulus_to_stimulus_interval,
    "stimulus_window": stimulus_window,
    "n_stimulus": (12, 2),
    "n_sequences": observations.shape[0],
    "nonnegative_smgp": False,
    "scaling_activation": "exp",
    "sparse": sparse,
    "seed": 0  # NB this is the seed for the chain, not the data generation
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
    sequences=observations,
    stimulus_order=order,
    target_stimulus=target,
    **settings,
    **prior_parameters
)
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file_chain],
    warmup=0,
    thin=1
)
self = results.to_predict(n_samples=n_samples)
# -----------------------------------------------------------------------------



# =============================================================================
# GET PREDICTIVE PROBABILITIES
llk_long = np.load(dir_results + f"dim{Ktrue}_K{K}_seed{seed}.npy")
llk_long = torch.Tensor(llk_long)
# -----------------------------------------------------------------------------




# =============================================================================
# TRANSFORM TO BCE
nchars = 19
nreps = 15
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
# LOO variance estimate
n_obs = llk.shape[0]
log_bf_loo = - torch.logsumexp(-llk_sum + llk, dim=0) + np.log(n_samples-1)
log_bf_loo *= n_obs / (n_obs - 1)
log_bf_loo_mean = log_bf_loo.mean().item()
log_bf_loo_se = (n_samples * torch.var(log_bf_loo)).pow(0.5).item()

# WAIC
vars_lpd = mllk_long.var(dim=1)
waic_i = lppd_i - vars_lpd
waic_se = (n_samples * waic_i.var()).pow(0.5).item()
waic_sum = waic_i.sum().item()
waic_p = vars_lpd.sum().item()

# PSIS-LOO
llk = mllk_long.unsqueeze(0).cpu().numpy()
log_weights, kss = az.psislw(-llk, reff=1.)
log_weights += llk
log_weights = torch.Tensor(log_weights)
loo_lppi_i = torch.logsumexp(log_weights, dim=-1)
loo_lppd = loo_lppi_i.sum().item()
loo_lppd_se = (n_samples * torch.var(loo_lppi_i)).pow(0.5).item()
loo_p = lppd - loo_lppd

# store
out = {
    "K": [K],
    "log_bf": [log_bf], "log_bf_loo": [log_bf_loo_mean], "log_bf_loo_se": [log_bf_loo_se],
    "lppd": [lppd],
    "elpd_loo": [loo_lppd], "elpd_loo_se": [loo_lppd_se], "p_loo": [loo_p],
    "elpd_waic": [waic_sum], "elpd_waic_se": [waic_se], "p_waic": [waic_p],
}
print(out)
# -----------------------------------------------------------------------------




# =============================================================================
# SAVE RESULTS
pd.DataFrame(out).T.to_csv(dir_results + f"dim{Ktrue}_K{K}_seed{seed}.icy")
# -----------------------------------------------------------------------------

