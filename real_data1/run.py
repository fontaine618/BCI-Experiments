import sys
import os
import torch
import time
import pickle

sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from src.data.k_protocol import KProtocol
from src.bffmbci.bffm import BFFModel


dir = "/home/simfont/Documents/BCI/experiments/real_data1/"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"

filename = dir_data + "K114_001_BCI_TRN.mat"

type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8
seed = 0
K = 5
n_iter = 20_000
cor = [0.95, 0.97, 0.99, 0.995, 0.999][int(sys.argv[1])]

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
    "shrinkage_factor": (2., 5.),
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
    if i % 1 == 0:
        print(f"{i:>10} "
              f"{model.variables['observations'].log_density_history[-1]:>20.4f}"
              f"  dt={time.time() - t00:>20.4f}   elapsed={time.time() - t0:20.4f}")
        t00 = time.time()
# -----------------------------------------------------------------------------




# =============================================================================
# SAVE CHAIN
# NB: use .chain to avoid VC-ing large objects
dir_out = dir + "chains/" + name + "/"
os.makedirs(dir_out, exist_ok=True)
out = model.results()
with open(dir_out + f"seed0_K5_cor{cor*1000}.chain", "wb") as f:
    pickle.dump(out, f)
# -----------------------------------------------------------------------------

