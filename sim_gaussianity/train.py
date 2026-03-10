import os
import sys
import time
import pickle
import itertools as it

import torch

sys.path.insert(1, "/home/simon/Documents/BCI")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci.bffm import BFFModel


# =============================================================================
# SETUP
DIR_DATA = "/home/simon/Documents/BCI/experiments/sim_gaussianity/data/"
DIR_CHAINS = "/home/simon/Documents/BCI/experiments/sim_gaussianity/chains/"
os.makedirs(DIR_CHAINS, exist_ok=True)

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
file_chain = file_stem + f"_K{k}"

n_iter = 10_000
cor = 0.95
shrinkage = 3.0
heterogeneity = 3.0
xi_var = 0.1
sparse = False
stimulus_window = 25
stimulus_to_stimulus_interval = 5
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
observations = torch.load(DIR_DATA + file_stem + ".observations")
order = torch.load(DIR_DATA + file_stem + ".order")
target = torch.load(DIR_DATA + file_stem + ".target")
# -----------------------------------------------------------------------------


# =============================================================================
# INITIALIZE MODEL
settings = {
    "latent_dim": k,
    "n_channels": observations.shape[1],
    "stimulus_to_stimulus_interval": stimulus_to_stimulus_interval,
    "stimulus_window": stimulus_window,
    "n_stimulus": (12, 2),
    "n_sequences": observations.shape[0],
    "nonnegative_smgp": False,
    "scaling_activation": "exp",
    "sparse": sparse,
    "seed": 0,
}

prior_parameters = {
    "observation_variance": (1.0, 10.0),
    "heterogeneities": heterogeneity,
    "shrinkage_factor": (1.0, shrinkage),
    "kernel_gp_factor_processes": (cor, 1.0, 1.0),
    "kernel_tgp_factor_processes": (cor, 0.5, 1.0),
    "kernel_gp_loading_processes": (cor, xi_var, 1.0),
    "kernel_tgp_loading_processes": (cor, 0.5, 1.0),
    "kernel_gp_factor": (cor, 1.0, 1.0),
}

model = BFFModel(
    sequences=observations,
    stimulus_order=order,
    target_stimulus=target,
    **settings,
    **prior_parameters,
)
# -----------------------------------------------------------------------------


# =============================================================================
# INITIALIZE CHAIN
torch.manual_seed(seed)
status = False
while not status:
    try:
        model.initialize_chain()
        status = True
    except Exception as exc:
        print(exc)
# -----------------------------------------------------------------------------


# =============================================================================
# RUN CHAIN
torch.manual_seed(seed)
t0 = time.time()
t00 = t0
for ii in range(n_iter):
    model.sample()
    if ii % 100 == 0:
        print(
            f"{ii:>10} "
            f"{model.variables['observations']._log_density_history[-1]:>20.4f}"
            f"  dt={time.time() - t00:>20.4f}   elapsed={time.time() - t0:20.4f}"
        )
        t00 = time.time()
# -----------------------------------------------------------------------------


# =============================================================================
# SAVE CHAIN
out = model.results(start=n_iter // 2, thin=10)
with open(DIR_CHAINS + file_chain + ".chain", "wb") as handle:
    pickle.dump(out, handle)
# -----------------------------------------------------------------------------

