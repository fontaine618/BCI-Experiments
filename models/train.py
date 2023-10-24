import sys
import os
import torch
import time
import pickle
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import DynamicRegressionCovarianceRegressionMean
from source.bffmbci.bffm import DynamicCovarianceRegressionMean
from source.bffmbci.bffm import StaticCovarianceRegressionMean
from source.bffmbci.bffm import DynamicRegressionCovarianceStaticMean

# =============================================================================
# SETUP
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = "/home/simfont/Documents/BCI/experiments/models/chains/"
os.makedirs(dir_chains, exist_ok=True)

# file
type = "TRN"
subject = "114"
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
K = 8
n_iter = 20_000
cor = 0.8
xi_var = 0.1
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
    "sparse": False,
    "shrinkage": "none",
    "seed": seed
}

prior_parameters = {
    "observation_variance": (1., 10.),
    "heterogeneities": 3.,
    "shrinkage_factor": (1., 3.),
    "kernel_gp_factor_processes": (cor, 1., 1.), # (1-step correlation, variance, power)
    "kernel_tgp_factor_processes": (cor, 1., 1.),
    "kernel_gp_loading_processes": (cor, xi_var, 1.),
    "kernel_tgp_loading_processes": (cor, 0.5, 1.),
    "kernel_gp_factor": (cor, 1., 1.)
}

# choose model
models = [
    # (Class, name, latent_dim)
    (DynamicRegressionCovarianceRegressionMean, "drcrm", K),
    (DynamicCovarianceRegressionMean, "dcrm", K),
    (StaticCovarianceRegressionMean, "scrm", K),
    (DynamicRegressionCovarianceStaticMean, "drcsm", K),
    (StaticCovarianceRegressionMean, "scrmfr", settings["n_channels"]),
]
Class, model_name, K = models[int(sys.argv[1])]
settings["latent_dim"] = K

model = Class(
    sequences=eeg.sequence,
    stimulus_order=eeg.stimulus_order,
    target_stimulus=eeg.target,
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
              f"{model.variables['observations'].log_density_history[-1]:>20.4f}"
              f"  dt={time.time() - t00:>20.4f}   elapsed={time.time() - t0:20.4f}")
        t00 = time.time()
# -----------------------------------------------------------------------------



# =============================================================================
# SAVE CHAIN
out = model.results(
    start=10_000,
    thin=10
)
with open(dir_chains + f"K{subject}_{model_name}.chain", "wb") as f:
    pickle.dump(out, f)
# -----------------------------------------------------------------------------

