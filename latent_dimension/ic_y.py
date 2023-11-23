import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import arviz as az
from source.bffmbci import BFFMResults
from source.data.k_protocol import KProtocol
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_figures = "/home/simfont/Documents/BCI/experiments/latent_dimension/figures/"
dir_results = "/home/simfont/Documents/BCI/experiments/latent_dimension/results/"
dir_chains = "/home/simfont/Documents/BCI/experiments/latent_dimension/chains/"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
os.makedirs(dir_figures, exist_ok=True)
os.makedirs(dir_results, exist_ok=True)

Ks = list(range(2, 14))
K = int(sys.argv[1])

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


# prediction settings
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = False
n_samples = 1000
factor_samples = 10
# -----------------------------------------------------------------------------


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
nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx
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
self = results.to_predict(n_samples=n_samples)
# -----------------------------------------------------------------------------



# =============================================================================
# GET PREDICTIVE PROBABILITIES
llk_long, chars = self.predict(
    order=order,
    sequence=sequence,
    factor_samples=factor_samples,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method,
    drop_component=None
)
# save
np.save(
    dir_results + f"K{subject}_dim{K}_mllk.npy",
    llk_long.cpu().numpy()
)
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
print(out)
# -----------------------------------------------------------------------------




# =============================================================================
# SAVE RESULTS
pd.DataFrame(out).T.to_csv(dir_results + f"K{subject}_dim{K}_llk_y.csv")
# -----------------------------------------------------------------------------

