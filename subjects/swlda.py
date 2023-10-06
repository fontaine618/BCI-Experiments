import sys
import torch
import numpy as np
import os
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from source.swlda.swlda import swlda, swlda_predict

# =============================================================================
# SETUP
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = "/home/simfont/Documents/BCI/experiments/subjects/chains/"
dir_results = "/home/simfont/Documents/BCI/experiments/subjects/predict/"
os.makedirs(dir_results, exist_ok=True)

# file
type = "TRN"
subject = ["114"][int(sys.argv[1])]
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8
# -----------------------------------------------------------------------------


# =============================================================================
# load data
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
# -----------------------------------------------------------------------------


# =============================================================================
# load data
response = eeg.stimulus.cpu().numpy()
type = eeg.stimulus_data["type"].values
trn = eeg.stimulus_data["repetition"] > -1
trn = eeg.stimulus_data.index[trn]
trnX = response[trn, ...]
trny = type[trn]
trnstim = eeg.stimulus_data.loc[trn]

whichchannels, restored_weights, bias = swlda(
    responses=trnX,
    type=trny,
    sampling_rate=1000,
    response_window=[0, response.shape[1] - 1],
    decimation_frequency=1000,
    max_model_features=150,
    penter=0.1,
    premove=0.15
)

Bmat = torch.zeros((16, 25))
Bmat[restored_weights[:, 0] - 1, restored_weights[:, 1] - 1] = torch.Tensor(restored_weights[:, 3])
Bmat.cpu()

# save bmat
np.save(dir_chains + f"K{subject}.swlda", Bmat.cpu().numpy())
# -----------------------------------------------------------------------------




# =============================================================================
# PREDICTIONS
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = "/home/simfont/Documents/BCI/experiments/subjects/chains/"
dir_results = "/home/simfont/Documents/BCI/experiments/subjects/predict/"
os.makedirs(dir_results, exist_ok=True)

# file
type = "FRT"
subject = ["114"][int(sys.argv[1])]
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8
# -----------------------------------------------------------------------------




# =============================================================================
# load data
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
# -----------------------------------------------------------------------------


# =============================================================================
# swLDA predictions
Bmat = np.load(dir_chains + f"K{subject}.swlda.npy")
# prepare data
response = eeg.stimulus.cpu().numpy()
type = eeg.stimulus_data["type"].values
trn = eeg.stimulus_data["repetition"] > -1
trn = eeg.stimulus_data.index[trn]
trnX = response[trn, ...]
trny = type[trn]
trnstim = eeg.stimulus_data.loc[trn]
# get predictions
trn_pred, trn_agg_pred, trn_cum_pred = swlda_predict(Bmat, trnX, trnstim, eeg.keyboard)
# get true
rowtrue = eeg.stimulus_data.loc[(eeg.stimulus_data["source"] < 7) & (eeg.stimulus_data["type"] == 1)]
rowtrue = rowtrue.groupby(["character"]).head(1).sort_values(["character"]).reset_index()
rowtrue = rowtrue[["character", "source"]].rename(columns={"source": "row"})
coltrue = eeg.stimulus_data.loc[(eeg.stimulus_data["source"] > 6) & (eeg.stimulus_data["type"] == 1)]
coltrue = coltrue.groupby(["character"]).head(1).sort_values(["character"]).reset_index()
coltrue = coltrue[["character", "source"]].rename(columns={"source": "col"})
true = rowtrue.merge(coltrue, on=["character"], how="outer").reset_index()
true["char"] = eeg.keyboard[true["row"] - 1, true["col"] - 7]

# metrics
trndf = trn_cum_pred.join(true.set_index("character"), on="character", how="left", rsuffix="_true", lsuffix="_pred")
trndf["hamming"] = (trndf["row_pred"] == trndf["row_true"]).astype(float) \
                   + (trndf["col_pred"] == trndf["col_true"]).astype(float)
trndf["acc"] = (trndf["char_pred"] == trndf["char_true"]).astype(float) / 2.
trndf = trndf.groupby("repetition").agg({"hamming": "mean", "acc": "mean"})
trndf["dataset"] = "FRT"
trndf["method"] = "swLDA"
trndf.reset_index(inplace=True)
trndf["subject"] = subject
trndf.to_csv(dir_results + f"K{subject}.frtswlda")
# -----------------------------------------------------------------------------
