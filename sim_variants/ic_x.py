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
dir_results = "/home/simfont/Documents/BCI/experiments/sim_variants/results/"
dir_chains = "/home/simfont/Documents/BCI/experiments/sim_variants/chains/"
dir_data = "/home/simfont/Documents/BCI/experiments/sim_variants/data/"
os.makedirs(dir_results, exist_ok=True)

# experiments
seeds = range(1)
Kxs = [8]
Kys = [5]
models = ["LR-DCR", "LR-DC", "LR-SC"]
K = 8

# combinations
combinations = it.product(seeds, Kxs, Kys, models, models)

# current experiment from command line
i = int(sys.argv[1])
seed, Kx, Ky, mtrue, mfitted = list(combinations)[i]

# file
file_data = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}"
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mfitted}"
file_mllk = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}_mllk"
file_out = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}"

# prediction settings
n_samples = 100
nchars = 19
nreps = 15
# -----------------------------------------------------------------------------





# =============================================================================
# LOAD DATA
observations = torch.load(dir_data + file_data + ".observations")
order = torch.load(dir_data + file_data + ".order")
target = torch.load(dir_data + file_data + ".target")
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
llk_long = np.load(dir_results + file_mllk + ".npy")
llk_long = torch.Tensor(llk_long)
# -----------------------------------------------------------------------------




# =============================================================================
# SELECT TARGET
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
pd.DataFrame(out).T.to_csv(dir_results + file_out + ".icx")
# -----------------------------------------------------------------------------

