import sys
import os
import torch
import time
import pickle
import pandas as pd

sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import BFFModel
from source.bffmbci import BFFMResults

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
cor = 0.5
shrinkage = [2., 5., 10.][int(sys.argv[1]) % 3]
reverse = [True, False][int(sys.argv[1]) // 3]
heterogeneity = 3.
xi_var = 0.003
sparse = False


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
    "seed": seed,
    "sparse": sparse
}

prior_parameters = {
    "observation_variance": (1., 10.),
    "heterogeneities": heterogeneity,
    "shrinkage_factor": (2., shrinkage),
    "kernel_gp_factor_processes": (cor, xi_var, 1.),
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
        model.initialize_chain(reverse=reverse)
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
dir_chains = dir + "chains/"
os.makedirs(dir_chains, exist_ok=True)
out = model.results()
file = f"shrinkage{shrinkage}_{'reversed' if reverse else 'original'}"
with open(dir_chains + file + ".chain", "wb") as f:
    pickle.dump(out, f)
# -----------------------------------------------------------------------------




# =============================================================================
# POSTERIOR
dir_posterior = dir + "posterior/"
os.makedirs(dir_posterior, exist_ok=True)

torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file + ".chain"],
    warmup=10_000,
    thin=1
)

s_chain = results.chains["shrinkage_factor"]
l_chain = results.chains["loadings"]
# get posterior mean
s_mean = s_chain.mean(dim=(0, 1))
l_mean = l_chain.mean(dim=(0, 1))
# save to csv
pd.DataFrame(s_mean.cpu().numpy()).to_csv(dir_posterior + file + "_smean.csv")
pd.DataFrame(l_mean.cpu().numpy()).to_csv(dir_posterior + file + "_lmean.csv")
# -----------------------------------------------------------------------------

