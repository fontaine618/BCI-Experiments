import sys
import os
import torch
import time
import pickle
import pandas as pd
import numpy as np
import scipy.special
import itertools as it
import torchmetrics
sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from torch.distributions import Categorical
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn import metrics
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


# experiment
seeds = range(10)
train_reps = [7] #, 5, 8]
experiment = list(it.product(seeds, train_reps))
experiment.append(("even", 7))
experiment.append(("odd", 7))

# method, suffix = "LogReg", "lr"
# method, suffix = "XGBoost", "xgb"
# method, suffix = "SVM", "svm"
# method, suffix = "RF", "rf"
method, suffix = "GB", "gb"
# -----------------------------------------------------------------------------


for seed, train_reps in experiment:

    # seed, train_reps = "odd", 7
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
    if isinstance(seed, int):
        torch.manual_seed(seed)
        reps = torch.randperm(15) + 1
        training_reps = reps[:train_reps].cpu().tolist()
        testing_reps = reps[train_reps:].cpu().tolist()
    elif seed == "even":
        training_reps = list(range(1, 16, 2))
        testing_reps = list(range(2, 17, 2))
    elif seed == "odd":
        training_reps = list(range(2, 17, 2))
        testing_reps = list(range(1, 16, 2))
    else:
        raise ValueError("Seed not recognized")
    eeg = eeg.repetitions(training_reps)
    # -----------------------------------------------------------------------------




    # =============================================================================
    # TRAIN MODEL

    X = eeg.stimulus.cpu().numpy()
    y = eeg.stimulus_data["type"].values
    kf = KFold(n_splits=5, shuffle=True)

    if method == "LogReg":
        model = LogisticRegression()
        Cs = np.logspace(-6, 0, 100)
        scores = np.zeros(len(Cs))
        for i, C in enumerate(Cs):
            model.C = C
            scores[i] = cross_val_score(model, X.reshape(X.shape[0], -1), y, cv=kf, scoring="neg_log_loss").mean()
        model.C = Cs[np.argmax(scores)]
        model.fit(X.reshape(X.shape[0], -1), y)
    elif method == "SVM":
        model = SVC(kernel="linear", probability=True)
        Cs = np.logspace(-6, 0, 20)
        scores = np.zeros(len(Cs))
        for i, C in enumerate(Cs):
            print(i, C)
            model.C = C
            scores[i] = cross_val_score(model, X.reshape(X.shape[0], -1), y, cv=kf, scoring="neg_log_loss").mean()
        model.C = Cs[np.argmax(scores)]
        model.fit(X.reshape(X.shape[0], -1), y)
    elif method == "RF":
        model = RandomForestClassifier(random_state=0)
        model.fit(X.reshape(X.shape[0], -1), y)
    elif method == "GB":
        model = GradientBoostingClassifier(random_state=0)
        model.fit(X.reshape(X.shape[0], -1), y)
    else:
        raise ValueError("Method not recognized")

    # -----------------------------------------------------------------------------


    # =============================================================================
    # TEST
    del eeg
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
    eeg = eeg.repetitions(testing_reps)
    X = eeg.stimulus.cpu().numpy()
    y = eeg.stimulus_data["type"].values
    trnstim = eeg.stimulus_data

    nchars = eeg.stimulus_data["character"].nunique()
    nreps = eeg.stimulus_data["repetition"].nunique()


    # get prediction
    log_proba = model.predict_log_proba(X.reshape(X.shape[0], -1))[:, 1]
    trnstim["log_proba"] = log_proba

    # to key probabilities
    log_prob = np.zeros((nchars, nreps, 36))
    for c in trnstim["character"].unique():
        cum_log_proba = np.zeros((6, 6))
        for j, r in enumerate(trnstim["repetition"].unique()):
            idx = (trnstim["character"] == c) & (trnstim["repetition"] == r)
            log_proba = trnstim.loc[idx, "log_proba"].values
            stim = trnstim.loc[idx, "source"].values
            log_proba_mat = np.zeros((6, 6))
            for i, s in enumerate(stim):
                if s < 7:
                    log_proba_mat[s-1, :] += log_proba[i]
                else:
                    log_proba_mat[:, s-7] += log_proba[i]
            log_proba_mat -= scipy.special.logsumexp(log_proba_mat)
            cum_log_proba += log_proba_mat
            log_prob[c-1, j, :] = cum_log_proba.flatten().copy()

    log_prob = torch.Tensor(log_prob)
    log_prob -= torch.logsumexp(log_prob, dim=-1, keepdim=True)
    # clamp -Inf to min value
    log_prob[log_prob == -np.inf] = log_prob[log_prob != -np.inf].min()

    Js = (6, 6)
    combinations = torch.cartesian_prod(*[torch.arange(J) for J in Js])
    to_add = torch.cumsum(torch.Tensor([0] + list(Js))[0:-1], 0).reshape(1, -1)
    combinations = combinations + to_add
    combinations = torch.nn.functional.one_hot(combinations.long(), sum(Js)).sum(1)

    wide_pred = log_prob.argmax(2)
    eeg.keyboard.flatten()[wide_pred.cpu()]
    wide_pred_one_hot = combinations[wide_pred, :]
    # -----------------------------------------------------------------------------






    # =============================================================================
    # METRICS

    # entropy
    entropy = Categorical(logits=log_prob).entropy()
    mean_entropy = entropy.mean(0)

    # accuracy & hamming
    target_wide = eeg.target.view(nchars, nreps, -1)
    accuracy = (wide_pred_one_hot == target_wide).all(2).double().mean(0)
    hamming = (wide_pred_one_hot != target_wide).double().sum(2).mean(0) / 2

    # binary cross-entropy
    ips = torch.einsum("...i,ji->...j", target_wide.double(), combinations.double())
    idx = torch.argmax(ips, -1)

    target_char = torch.nn.functional.one_hot(idx, 36)
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
    df = pd.DataFrame({
        "hamming": hamming.cpu(),
        "acc": accuracy.cpu(),
        "mean_entropy": entropy.mean(0).abs().cpu(),
        "bce": bce.cpu(),
        "auroc": auc.cpu(),
        "dataset": name + "_test",
        "repetition": range(1, nreps + 1),
        "training_reps": train_reps,
        "method": method,
    }, index=range(1, nreps + 1))
    print(df)
    df.to_csv(dir_results + f"K{subject}_trn{train_reps}_seed{seed}_{suffix}.test")
    # -----------------------------------------------------------------------------

    del eeg
    del model