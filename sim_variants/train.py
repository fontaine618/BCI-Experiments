import sys
import os
import torch
import time
import pickle
import itertools as it
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.bffmbci.bffm import DynamicRegressionCovarianceRegressionMean
from source.bffmbci.bffm import DynamicCovarianceRegressionMean
from source.bffmbci.bffm import StaticCovarianceRegressionMean
from source.bffmbci.bffm import DynamicRegressionCovarianceStaticMean

# =============================================================================
# SETUP
dir_data = "/home/simfont/Documents/BCI/experiments/sim_variants/data/"
dir_chains = "/home/simfont/Documents/BCI/experiments/sim_variants/chains/"
os.makedirs(dir_chains, exist_ok=True)

# experiments
seeds = range(1)
Kxs = [8]
Kys = [5]
models = ["LR-DCR", "LR-DC", "LR-SC"]

# combinations
combinations = it.product(seeds, Kxs, Kys, models, models)

# current experiment from command line
i = int(sys.argv[1])
seed, Kx, Ky, mtrue, mfitted = list(combinations)[i]

# file
file = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}"
file_chain = f"Kx{Kx}_Ky{Ky}_seed{seed}_model{mtrue}_model{mfitted}"

# model
n_iter = 10_000
cor = 0.95
shrinkage = 3.
heterogeneity = 3.
xi_var = 0.1
sparse = False
K = 8


# dimensions
n_channels = 16
stimulus_window = 25
stimulus_to_stimulus_interval = 5
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
observations = torch.load(dir_data + file + ".observations")
order = torch.load(dir_data + file + ".order")
target = torch.load(dir_data + file + ".target")
variables = pickle.load(open(dir_data + file + ".variables", "rb"))
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
    "kernel_gp_loading_processes": (cor, xi_var, 1.),
    "kernel_tgp_loading_processes": (cor, 0.5, 1.),
    "kernel_gp_factor": (cor, 1., 1.)
}

Model = {
    "LR-DCR": DynamicRegressionCovarianceRegressionMean,
    "LR-DC": DynamicCovarianceRegressionMean,
    "LR-SC": StaticCovarianceRegressionMean,
}[mfitted]

model = Model(
    sequences=observations,
    stimulus_order=order,
    target_stimulus=target,
    **settings,
    **prior_parameters
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
    except Exception as e:
        print(e)
# -----------------------------------------------------------------------------



# =============================================================================
# RUN CHAIN
torch.manual_seed(seed)
t0 = time.time()
t00 = t0
for i in range(n_iter):
    model.sample()
    if i % 100 == 0:
        print(f"{i:>10} "
              f"{model.variables['observations']._log_density_history[-1]:>20.4f}"
              f"  dt={time.time() - t00:>20.4f}   elapsed={time.time() - t0:20.4f}")
        t00 = time.time()
# -----------------------------------------------------------------------------



# =============================================================================
# SAVE CHAIN
out = model.results(
    start=5_000,
    thin=10
)
with open(dir_chains + file_chain + f".chain", "wb") as f:
    pickle.dump(out, f)
# -----------------------------------------------------------------------------

