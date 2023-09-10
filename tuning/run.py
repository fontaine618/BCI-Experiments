import sys
import os
import torch
import time
import pickle

sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import BFFModel

dir = "/home/simfont/Documents/BCI/experiments/tuning/"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"

type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8
seed = 0
K = 8
n_iter = 20_000
nreps = 7
# cor = [0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99][int(sys.argv[1]) % 7]
# shrinkage = [3., 4., 5., 7., 10.][int(sys.argv[1]) // 7]
cor = [0.2, 0.3, 0.4, 0.5, 0.6][int(sys.argv[1]) % 5]
shrinkage = [3., 4., 5., 7., 10.][int(sys.argv[1]) // 5]

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

# subset to first repetitions
eeg = eeg.repetitions(range(1, nreps+1))

settings = {
    "latent_dim": K,
    "n_channels": eeg.sequence.shape[1],
    "stimulus_to_stimulus_interval": eeg.stimulus_to_stimulus_interval,
    "stimulus_window": eeg.stimulus_window,
    "n_stimulus": (12, 2),
    "n_sequences": eeg.sequence.shape[0],
    "nonnegative_smgp": False,
    "scaling_activation": "exp",
    "seed": seed
}

prior_parameters = {
    "observation_variance": (1., 10.),
    "heterogeneities": 3.,
    "shrinkage_factor": (2., shrinkage),
    "kernel_gp_factor_processes": (cor, 1., 1.),
    "kernel_tgp_factor_processes": (cor, 0.5, 1.),
    "kernel_gp_loading_processes": (cor, 0.1, 1.),
    "kernel_tgp_loading_processes": (cor, 0.5, 1.),
    "kernel_gp_factor": (cor, 1., 1.)
}

model = BFFModel(
    sequences=eeg.sequence,
    stimulus_order=eeg.stimulus_order,
    target_stimulus=eeg.target,
    **settings,
    **prior_parameters
)



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
dir_out = dir + "chains/"
os.makedirs(dir_out, exist_ok=True)
out = model.results()
with open(dir_out + f"seed{seed}_nreps{nreps}_cor{cor}_shrinkage{shrinkage}.chain", "wb") as f:
    pickle.dump(out, f)
# -----------------------------------------------------------------------------

