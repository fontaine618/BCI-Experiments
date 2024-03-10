import numpy as np
import os
import sys
import torch

sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci import BFFMResults
from source.data.k_protocol import KProtocol


# =============================================================================
# SETUP
type = "TRN"
subject = str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"

dir_results = f"/home/simfont/Documents/BCI/experiments/subject/results/K{subject}/"
dir_chains = f"/home/simfont/Documents/BCI/experiments/subject/chains/K{subject}/"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
os.makedirs(dir_results, exist_ok=True)

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
nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()
order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD RESULTS
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + f"K{subject}.chain"],
    warmup=0,
    thin=1
)
character_idx = torch.arange(n_characters).repeat_interleave(n_repetitions)
# -----------------------------------------------------------------------------

# =============================================================================
# GET PREDICTIVE PROBABILITIES (FULL)
self = results.to_predict(n_samples=n_samples)
llk_long, _ = self.predict(
    order=order,
    sequence=sequence,
    factor_samples=1,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method,
    drop_component=None,
    batchsize=20
)
# save
np.save(
    dir_results + f"K{subject}_mllk_full.npy",
    llk_long.cpu().numpy()
)
# -----------------------------------------------------------------------------

# =============================================================================
# GET PREDICTIVE PROBABILITIES
drop_components = list(range(K))
self = results.to_predict(n_samples=n_samples)
llk_long, _ = self.predict(
    order=order,
    sequence=sequence,
    factor_samples=1,
    character_idx=character_idx,
    factor_processes_method=factor_processes_method,
    drop_component=drop_components,
    batchsize=20
)
# save
np.save(
    dir_results + f"K{subject}_mllk_null.npy",
    llk_long.cpu().numpy()
)
# -----------------------------------------------------------------------------

for k in range(K):
    self = results.to_predict(n_samples=n_samples)
    # =============================================================================
    # GET DROP ONE PREDICTIVE PROBABILITIES
    drop_components = [k]
    llk_long, _ = self.predict(
        order=order,
        sequence=sequence,
        factor_samples=1,
        character_idx=character_idx,
        factor_processes_method=factor_processes_method,
        drop_component=drop_components,
        batchsize=20
    )
    # save
    np.save(
        dir_results + f"K{subject}_mllk_drop{k}.npy",
        llk_long.cpu().numpy()
    )
    # -----------------------------------------------------------------------------


    self = results.to_predict(n_samples=n_samples)
    # =============================================================================
    # GET DROP ONE PREDICTIVE PROBABILITIES
    drop_components = list(range(K))
    drop_components.remove(k)
    llk_long, _ = self.predict(
        order=order,
        sequence=sequence,
        factor_samples=1,
        character_idx=character_idx,
        factor_processes_method=factor_processes_method,
        drop_component=drop_components,
        batchsize=20
    )
    # save
    np.save(
        dir_results + f"K{subject}_mllk_just{k}.npy",
        llk_long.cpu().numpy()
    )
    # -----------------------------------------------------------------------------


