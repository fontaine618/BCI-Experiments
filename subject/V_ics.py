import numpy as np
import os
import sys
sys.path.insert(1, '/home/simon/Documents/BCI/src')
import torch
import itertools as it
import arviz as az
import pandas as pd
from source.bffmbci import BFFMResults
from source.data.k_protocol import KProtocol
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_results = "/home/simon/Documents/BCI/experiments/subject/results/"
dir_chains = "/home/simon/Documents/BCI/experiments/subject/chains/"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
os.makedirs(dir_results, exist_ok=True)


# file
type = "TRN"
subject = "114"#str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

# model
seed = 0
V = "CS"#["LR-DCR", "LR-DC", "LR-SC", "CS"][int(sys.argv[2])]
K = 17 if V == "CS" else 8
n_iter = 20_000
sparse = False

# prediction settings
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
n_samples = 100

# dimensions
n_characters = 19
n_repetitions = 15
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD DATA
eeg = KProtocol(
    filename=dir_data + name + ".mat",
    type=type,
    subject=subject,
    session=session,
    window=window,
    bandpass_window=bandpass_window,
    bandpass_order=bandpass_order,
    downsample=downsample,
)
nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()
order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD RESULTS
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + f"K{subject}_allreps_{V}.chain"],
    warmup=0,
    thin=1
)
self = results.to_predict(n_samples=n_samples)
character_idx = torch.arange(n_characters).repeat_interleave(n_repetitions)
# -----------------------------------------------------------------------------



# =============================================================================
# GET PREDICTIVE PROBABILITIES
llk_long = np.load(dir_results + f"K{subject}_allreps_{V}_mllk.npy")
llk_long = torch.Tensor(llk_long)
log_prob = self.aggregate(
    llk_long,
    sample_mean=sample_mean,
    which_first=which_first
)
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

pd.DataFrame(out).T.to_csv(dir_results + f"K{subject}_allreps_{V}.icx")
# -----------------------------------------------------------------------------





# =============================================================================
# TRANSFORM TO BCE
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

pd.DataFrame(out).T.to_csv(dir_results + f"K{subject}_allreps_{V}.icy")
# -----------------------------------------------------------------------------