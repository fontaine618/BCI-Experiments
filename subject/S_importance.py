import numpy as np
import pandas as pd
import os
import sys
import torchmetrics
import torch

sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci import BFFMResults, importance_statistic
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
self = results.to_predict(n_samples=n_samples)
# -----------------------------------------------------------------------------




# =============================================================================
# COMPUTE STATISTIC FROM POSTERIOR
post = importance_statistic(results.chains)
# -----------------------------------------------------------------------------



components = ["full"] + [f"_drop{k}" for k in range(K)]
out = {}
for c in components:

    # =============================================================================
    # GET DROP ONE PREDICTIVE PROBABILITIES
    filename = dir_results + f"K{subject}_mllk{c}.npy"
    llk_long = torch.Tensor(np.load(filename))
    # -----------------------------------------------------------------------------


    # =============================================================================
    # PREDICTIONS
    log_prob = self.aggregate(
        llk_long,
        sample_mean=sample_mean,
        which_first=which_first
    ) # n_chars x n_reps x 36
    wide_pred_one_hot = self.get_predictions(log_prob, True) # n_chars x n_reps x 12
    # -----------------------------------------------------------------------------

    # =============================================================================
    # METRICS

    # accuracy & hamming
    target_wide = target.view(nchars, nreps, -1)
    accuracy = (wide_pred_one_hot == target_wide).all(2).double().mean(0)
    hamming = (wide_pred_one_hot != target_wide).double().sum(2).mean(0) / 2

    # binary cross-entropy
    target_char = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_wide), 36)
    bce = - (target_char * log_prob).sum(2).mean(0)

    # auc
    target_char_int = torch.argmax(target_char, -1)
    auc = torch.Tensor([
        torchmetrics.functional.classification.multiclass_auroc(
            preds=log_prob[:, c, :],
            target=target_char_int[:, c],
            num_classes=36,
            average="weighted"
        ) for c in range(nreps)
    ])

    # save
    out[c] = {
        "hamming": hamming[-1].item(),
        "acc": accuracy[-1].item(),
        "bce": bce[-1].item(),
        "auroc": auc[-1].item(),
    }
    # -----------------------------------------------------------------------------


drop_hamming = [out[f"_drop{k}"]["hamming"] - out[f"_full"]["hamming"] for k in range(K)]
drop_acc = [-out[f"_drop{k}"]["acc"] + out[f"_full"]["acc"] for k in range(K)]
drop_bce = [out[f"_drop{k}"]["bce"] - out[f"_full"]["bce"] for k in range(K)]
drop_auroc = [-out[f"_drop{k}"]["auroc"] + out[f"_full"]["auroc"] for k in range(K)]

df = pd.DataFrame({
    "component": range(K),
    "drop_hamming": drop_hamming,
    "drop_acc": drop_acc,
    "drop_bce": drop_bce,
    "drop_auroc": drop_auroc,
    "posterior": post
}, index=range(K))
df.to_csv(
    dir_results + f"K{subject}_importance.csv"
)