import sys
import os
import torch
import time
import pickle
import itertools as it
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import DynamicRegressionCovarianceRegressionMean

# =============================================================================
# SETUP
type = "TRN"
subject = str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = f"/home/simfont/Documents/BCI/experiments/subject/chains/K{subject}/"
os.makedirs(dir_chains, exist_ok=True)
filename = dir_data + name + ".mat"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

# model
lite = True
seed = 0
K = 5 if lite else 8
V = "LR-SC" if lite else "LR-DCR"
cor = 0.50
n_iter = 20_000


# experiment
seeds = range(10)
train_reps = [3, 5, 8]
experiment = list(it.product(train_reps, seeds))
seed, train_reps = experiment[int(sys.argv[2])]

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
# subset training reps
torch.manual_seed(seed)
reps = torch.randperm(15) + 1
training_reps = reps[:train_reps].cpu().tolist()
testing_reps = reps[train_reps:].cpu().tolist()
eeg = eeg.repetitions(training_reps)
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

cor = 0.5
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

model = DynamicRegressionCovarianceRegressionMean(
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
model.initialize_chain(weighted=True)
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
    start=10_000,
    thin=10
)
with open(dir_chains + f"K{subject}_trn{train_reps}_seed{seed}{'_lite' if lite else ''}.chain", "wb") as f:
    pickle.dump(out, f)
# -----------------------------------------------------------------------------

