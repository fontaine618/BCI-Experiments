import os
import sys
import itertools as it

import arviz as az
import numpy as np
import pandas as pd
import torch

sys.path.insert(1, "/home/simon/Documents/BCI")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci import BFFMResults


# =============================================================================
# SETUP
DIR_RESULTS = "/home/simon/Documents/BCI/experiments/sim_gaussianity/results/"
DIR_CHAINS = "/home/simon/Documents/BCI/experiments/sim_gaussianity/chains/"
DIR_DATA = "/home/simon/Documents/BCI/experiments/sim_gaussianity/data/"
os.makedirs(DIR_RESULTS, exist_ok=True)

# experiments
seeds = [0]
Kxs = [8]
Kys = [5]
noise_settings = [
    ("gaussian", None),
    ("student_t", 20.0),
    ("student_t", 10.0),
    ("student_t", 5.0),
    ("student_t", 3.0),
]

# combinations
combinations = it.product(seeds, Kxs, Kys, noise_settings)

i = int(sys.argv[1])
seed, kx, ky, (noise_distribution, df) = list(combinations)[i]
k = 8


def _df_tag(df):
    if df is None:
        return "na"
    return str(df).replace(".", "p")


file_stem = f"Kx{kx}_Ky{ky}_seed{seed}_noise{noise_distribution}_df{_df_tag(df)}"
file_chain = file_stem + f"_K{k}.chain"

n_samples = 100
nchars = 19
nreps = 5
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
target = torch.load(DIR_DATA + file_stem + ".target")
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD RESULTS
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [DIR_CHAINS + file_chain],
    warmup=0,
    thin=1,
)
self = results.to_predict(n_samples=n_samples)
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD PREDICTIVE LOG-LIKELIHOOD SAMPLES
llk_long = np.load(DIR_RESULTS + file_stem + f"_K{k}_mllk.npy")
llk_long = torch.tensor(llk_long)
# -----------------------------------------------------------------------------


# =============================================================================
# SELECT TARGET llk p(x|y)
llk_long = llk_long.reshape(nchars * nreps, 36, n_samples)
target_ = target.unsqueeze(1).repeat(1, n_samples, 1)
target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
target36 = target36.permute(0, 2, 1)
mllk_long = (target36 * llk_long).sum(1)
# -----------------------------------------------------------------------------


# =============================================================================
# COMPUTE ICs
lppd_i = torch.logsumexp(mllk_long, dim=1) - np.log(n_samples)
lppd = lppd_i.sum().item()

llk = mllk_long
llk_sum = llk.sum(0)
log_bf = -torch.logsumexp(-llk_sum, dim=0).item() + np.log(n_samples)
n_obs = llk.shape[0]
log_bf_loo = -torch.logsumexp(-llk_sum + llk, dim=0) + np.log(n_samples - 1)
log_bf_loo *= n_obs / (n_obs - 1)
log_bf_loo_mean = log_bf_loo.mean().item()
log_bf_loo_se = (n_samples * torch.var(log_bf_loo)).pow(0.5).item()

vars_lpd = mllk_long.var(dim=1)
waic_i = lppd_i - vars_lpd
waic_se = (n_samples * waic_i.var()).pow(0.5).item()
waic_sum = waic_i.sum().item()
waic_p = vars_lpd.sum().item()

llk = mllk_long.unsqueeze(0).cpu().numpy()
log_weights, _ = az.psislw(-llk, reff=1.0)
log_weights += llk
log_weights = torch.tensor(log_weights)
loo_lppi_i = torch.logsumexp(log_weights, dim=-1)
loo_lppd = loo_lppi_i.sum().item()
loo_lppd_se = (n_samples * torch.var(loo_lppi_i)).pow(0.5).item()
loo_p = lppd - loo_lppd

out = {
    "K": [k],
    "log_bf": [log_bf],
    "log_bf_loo": [log_bf_loo_mean],
    "log_bf_loo_se": [log_bf_loo_se],
    "lppd": [lppd],
    "elpd_loo": [loo_lppd],
    "elpd_loo_se": [loo_lppd_se],
    "p_loo": [loo_p],
    "elpd_waic": [waic_sum],
    "elpd_waic_se": [waic_se],
    "p_waic": [waic_p],
}
print(out)
# -----------------------------------------------------------------------------


# =============================================================================
# SAVE RESULTS
pd.DataFrame(out).T.to_csv(DIR_RESULTS + file_stem + f"_K{k}.icx")
# -----------------------------------------------------------------------------

