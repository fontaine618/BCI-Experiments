import os
import sys
import itertools as it

import numpy as np
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
seed, kx, ky, (noise_distribution, df), k = list(combinations)[i]


def _df_tag(df):
    if df is None:
        return "na"
    return str(df).replace(".", "p")


file_stem = f"Kx{kx}_Ky{ky}_seed{seed}_noise{noise_distribution}_df{_df_tag(df)}"
file_chain = file_stem + f"_K{k}.chain"

factor_processes_method = "analytical"
n_samples = 100
n_characters = 19
n_repetitions = 5
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
observations = torch.load(DIR_DATA + file_stem + ".observations")
order = torch.load(DIR_DATA + file_stem + ".order")
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
character_idx = torch.arange(n_characters).repeat_interleave(n_repetitions)
# -----------------------------------------------------------------------------


# =============================================================================
# GET PREDICTIVE LOG-LIKELIHOOD SAMPLES
llk_long, _ = self.predict(
    order=order,
    sequence=observations,
    factor_samples=1,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method,
    drop_component=None,
    batchsize=10,
)
np.save(DIR_RESULTS + file_stem + f"_K{k}_mllk.npy", llk_long.cpu().numpy())
# -----------------------------------------------------------------------------

