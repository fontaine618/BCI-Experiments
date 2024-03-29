import sys
import os
import torch
import time
import pickle
import itertools as it
import pandas as pd
import torchmetrics
sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm_map import BFFModelMAP
from source.bffmbci.bffm_map import DynamicRegressionCovarianceRegressionMeanMAP
from source.bffmbci.bffm_map import DynamicCovarianceRegressionMeanMAP
from source.bffmbci.bffm_map import StaticCovarianceRegressionMeanMAP
from torch.distributions import Categorical
from source.nb_mn import NaiveBayesMatrixNormal

# =============================================================================
# SETUP
type = "TRN"
subject = "114" #str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_results = f"/home/simon/Documents/BCI/experiments/subject/results/K{subject}/"
os.makedirs(dir_results, exist_ok=True)
filename = dir_data + name + ".mat"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

# model
lite = False
seed = 0
K = 8
V = "LR-SC" #if lite else "LR-DCR"
cor = 0.50


# experiment
seeds = range(10)
train_reps = [3] #, 5, 8]
experiment = list(it.product(seeds, train_reps))
train_reps, seed = 3, 0 #experiment[int(sys.argv[2])]

# -----------------------------------------------------------------------------


for seed, train_reps in experiment:
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
    nchars = eeg.stimulus_data["character"].nunique()
    nreps = eeg.stimulus_data["repetition"].nunique()
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

    ModelMAP = {
        "LR-DCR": DynamicRegressionCovarianceRegressionMeanMAP,
        "LR-DC": DynamicCovarianceRegressionMeanMAP,
        "LR-SC": StaticCovarianceRegressionMeanMAP,
    }["LR-SC"]

    model: BFFModelMAP = ModelMAP(
        sequences=eeg.sequence,
        stimulus_order=eeg.stimulus_order,
        target_stimulus=eeg.target,
        **settings,
        **prior_parameters
    )
    # -----------------------------------------------------------------------------



    # =============================================================================
    # GET LOADING INITIALIZATION
    K0 = -(K // -2) # ceiling division
    K1 = K - K0
    X = eeg.stimulus
    y = torch.Tensor(eeg.stimulus_data["type"].values)
    nbmn = NaiveBayesMatrixNormal(25, 16)
    nbmn.fit(X, y)
    loadings = nbmn.construct_loadings(K0, K1)
    # -----------------------------------------------------------------------------



    # =============================================================================
    # FIT MODEL
    model.initialize(loadings=None)
    model.fit(lr=0.1, max_iter=2000, tol=1e-8)
    # -----------------------------------------------------------------------------



    # =============================================================================
    # GET PREDICTIONS
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
    eeg = eeg.repetitions(testing_reps)
    nchars = eeg.stimulus_data["character"].nunique()
    nreps = eeg.stimulus_data["repetition"].nunique()

    # test set
    model.update_data(
        sequences=eeg.sequence,
        stimulus_order=eeg.stimulus_order,
        target_stimulus=eeg.target
    )


    log_proba = model.predict(method="maximize").detach()

    # aggregate
    log_proba_wide = log_proba.reshape(nchars, nreps, -1)
    cum_log_proba = log_proba_wide.cumsum(1)
    cum_log_proba -= torch.logsumexp(cum_log_proba, 2, keepdim=True)

    wide_pred = cum_log_proba.argmax(2)
    print(eeg.keyboard.flatten()[wide_pred.cpu()])
    wide_pred_one_hot = model.combinations[wide_pred, :]
    # -----------------------------------------------------------------------------


    # =============================================================================
    # METRICS

    # entropy
    entropy = Categorical(logits=cum_log_proba).entropy()
    mean_entropy = entropy.mean(0)

    # accuracy & hamming
    target_wide = eeg.target.view(nchars, nreps, -1)
    accuracy = (wide_pred_one_hot == target_wide).all(2).double().mean(0)
    hamming = (wide_pred_one_hot != target_wide).double().sum(2).mean(0) / 2

    # binary cross-entropy
    ips = torch.einsum("...i,ji->...j", target_wide.double(), model.combinations.double())
    idx = torch.argmax(ips, -1)

    target_char = torch.nn.functional.one_hot(idx, 36)
    bce = - (target_char * cum_log_proba).sum(2).mean(0)

    # auc
    target_char_int = torch.argmax(target_char, -1)
    auc = torch.Tensor([
        torchmetrics.functional.classification.multiclass_auroc(
            preds=cum_log_proba[:, c, :],
            target=target_char_int[:, c],
            num_classes=36,
            average="weighted"
        ) for c in range(nreps)
    ])

    # save
    df = pd.DataFrame({
        "hamming": hamming.cpu(),
        "acc": accuracy.cpu(),
        "mean_entropy": mean_entropy.abs().cpu(),
        "bce": bce.cpu(),
        "auroc": auc.cpu(),
        "dataset": name + "_test",
        "repetition": range(1, nreps + 1),
        "training_reps": train_reps,
        "method": V + "-MAP",
        "K": K,
        "cor": cor
    }, index=range(1, nreps + 1))
    df.to_csv(dir_results + f"K{subject}_trn{train_reps}_seed{seed}_map.test")
    # -----------------------------------------------------------------------------