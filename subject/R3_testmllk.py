import numpy as np
import os
import sys
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
import torch
import itertools as it
from source.bffmbci import BFFMResults
from source.data.k_protocol import KProtocol
torch.set_default_tensor_type(torch.cuda.FloatTensor)


# =============================================================================
# SETUP
dir_results = "/home/simfont/Documents/BCI/experiments/subject/results/"
dir_chains = "/home/simfont/Documents/BCI/experiments/subject/chains/"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
os.makedirs(dir_results, exist_ok=True)


# file
type = "TRN"
subject = str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

# model
seed = 0
K = 8
V = ["LR-DCR", "LR-DC", "LR-SC"][0]
cor = [0.35, 0.40, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8][3]
n_iter = 20_000

# experiment
seeds = range(10)
train_reps = [3, 5, 8]
experiment = list(it.product(seeds, train_reps))
seed, train_reps = experiment[int(sys.argv[2])]

# prediction settings
factor_processes_method = "analytical"
n_samples = 100

# dimensions
n_characters = 19
n_repetitions = 15
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
eeg = KProtocol(
    filename=dir_data + name + ".mat",
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
training_reps = (torch.randperm(15)[:train_reps].cpu() + 1 ).tolist()
testing_reps = [i for i in range(1, 16) if i not in training_reps]
eeg = eeg.repetitions(testing_reps)

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
    [dir_chains + f"K{subject}_trn{train_reps}_seed{seed}.chain"],
    warmup=0,
    thin=1
)
self = results.to_predict(n_samples=n_samples)
# -----------------------------------------------------------------------------



# =============================================================================
# GET PREDICTIVE PROBABILITIES
llk_long, chars = self.predict(
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
    dir_results + f"K{subject}_trn{train_reps}_seed{seed}_testmllk.npy",
    llk_long.cpu().numpy()
)
# -----------------------------------------------------------------------------


