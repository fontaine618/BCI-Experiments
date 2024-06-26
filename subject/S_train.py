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

from source.bffmbci.bffm_map import BFFModelMAP
from source.bffmbci.bffm_map import DynamicRegressionCovarianceRegressionMeanMAP
from source.bffmbci.bffm_map import DynamicCovarianceRegressionMeanMAP
from source.bffmbci.bffm_map import StaticCovarianceRegressionMeanMAP

# =============================================================================
# SETUP
type = "TRN"
subject = str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"

dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = f"/home/simfont/Documents/BCI/experiments/subject/chains/K{subject}/"
os.makedirs(dir_chains, exist_ok=True)

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

# model
seed = 0
K = 8
V = "LR-DCR"
cor = 0.5
n_iter = 20_000

# prediction settings
factor_processes_method = "analytical"
n_samples = 100
sample_mean = "arithmetic"
which_first = "sequence"

# dimensions
n_characters = 19
n_repetitions = 15
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
filename = dir_data + name + ".mat"
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
    "sparse": False,
    "seed": seed,
    "shrinkage": "none"
}

prior_parameters = {
    "observation_variance": (1., 10.),
    "heterogeneities": 3.,
    "shrinkage_factor": (2., 3.),
    "kernel_gp_factor_processes": (cor, 1., 2.),
    "kernel_tgp_factor_processes": (cor, 0.5, 2.),
    "kernel_gp_loading_processes": (cor, 0.1, 2.),
    "kernel_tgp_loading_processes": (cor, 0.5, 2.),
    "kernel_gp_factor": (cor, 1., 2.)
}
# -----------------------------------------------------------------------------


# =============================================================================
# MAP INITIALIZATION
ModelMAP = StaticCovarianceRegressionMeanMAP

modelMAP: BFFModelMAP = ModelMAP(
    sequences=eeg.sequence,
    stimulus_order=eeg.stimulus_order,
    target_stimulus=eeg.target,
    **settings,
    **prior_parameters
)

modelMAP.initialize()
modelMAP.fit(lr=0.1, max_iter=2000, tol=1e-8)
variablesMAP = modelMAP.export_variables()
variablesMAP["smgp_scaling.mixing_process"] += 0.1
# -----------------------------------------------------------------------------




# =============================================================================
# INITIALIZE MODEL
Model = {
    "LR-DCR": DynamicRegressionCovarianceRegressionMean,
    "LR-DC": DynamicCovarianceRegressionMean,
    "LR-SC": StaticCovarianceRegressionMean,
}[V]

model = Model(
    sequences=eeg.sequence,
    stimulus_order=eeg.stimulus_order,
    target_stimulus=eeg.target,
    **settings,
    **prior_parameters
)
# -----------------------------------------------------------------------------



# =============================================================================
# INITIALIZE CHAIN TO MAP
model.set(**variablesMAP)
model.clear_history()
# -----------------------------------------------------------------------------



# =============================================================================
# RUN CHAIN
torch.manual_seed(seed)
t0 = time.time()
t00 = t0
for i in range(n_iter):
    model.sample()
    if i % 1 == 0:
        print(f"{i:>10} "
              f"{model.variables['observations']._log_density_history[-1]:>20.4f}"
              f"  dt={time.time() - t00:>20.4f}   elapsed={time.time() - t0:20.4f}")
        t00 = time.time()
# -----------------------------------------------------------------------------



# =============================================================================
# SAVE CHAIN
out = model.results(
    start=10_000,
    thin=10
)
with open(dir_chains + f"K{subject}.chain", "wb") as f:
    pickle.dump(out, f)
# -----------------------------------------------------------------------------

