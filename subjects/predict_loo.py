import sys
import os
import torch
import time
import pickle

sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci import BFFMResults
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from source.bffmbci.bffm import BFFModel

plt.style.use("seaborn-v0_8-whitegrid")

from source.data.k_protocol import KProtocol

# =============================================================================
# SETUP
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = "/home/simfont/Documents/BCI/experiments/subjects/"
dir_results = "/home/simfont/Documents/BCI/experiments/predict/"
os.makedirs(dir_results, exist_ok=True)


# file
type = "TRN"
subject = ["114"][int(sys.argv[1])]
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"
file = f"K{subject}.chain"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8


# model
seed = 0
K = 5
n_iter = 20_000
cor = 0.8
shrinkage = 5.
heterogeneity = 3.
xi_var = 0.003

# settings
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = True
n_samples = 100
nchars = 19
nreps = 15
factor_samples = 10

out = []
# -----------------------------------------------------------------------------



# =============================================================================
# testing accuracy
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

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx

for c in [None, 0, 1, 2, 3, 4, 5]:
    torch.cuda.empty_cache()
    results = BFFMResults.from_files(
        [dir_chains + file],
        warmup=0,  # already warmed up
        thin=1
    )
    self = results.to_predict(n_samples=n_samples)

    llk_long, chars = self.predict(
        order=order,
        sequence=sequence,
        factor_samples=factor_samples,
        character_idx=character_idx,
        factor_processes_method=factor_processes_method,
        drop_component=c
    )

    log_prob = self.aggregate(
        llk_long,
        sample_mean=sample_mean,
        which_first=which_first
    )

    wide_pred_one_hot = self.get_predictions(log_prob, True)

    entropy = Categorical(logits=log_prob).entropy()

    target_ = target[::nreps, :].unsqueeze(1).repeat(1, nreps, 1)
    hamming = (wide_pred_one_hot != target_).double().sum(2).sum(0) / 2
    acc = (wide_pred_one_hot == target_).all(2).double().sum(0)

    target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
    bce = (target36 * log_prob).sum(-1).mean(0)

    df = pd.DataFrame({
        "hamming": hamming.cpu(),
        "acc": acc.cpu(),
        "mean_entropy": entropy.mean(0).abs().cpu(),
        "bce": bce.cpu(),
        "dataset": "training",
        "repetition": range(1, nreps + 1),
        "subject": subject,
        "sample_mean": sample_mean,
        "which_first": which_first,
        "drop_component": c+1 if c is not None else "None",
        "method": "BFFM",
    }, index=range(1, nreps + 1))
    print(df)

    out.append(df)

df = pd.concat(out)
os.makedirs(dir_results, exist_ok=True)
df.to_csv(dir_results + f"K{subject}.loo")
print(df)
# -----------------------------------------------------------------------------
