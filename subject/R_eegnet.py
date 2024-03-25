import sys
import os
import torch
import time
import pickle
import numpy as np
import pandas as pd
import torchmetrics
import itertools as it
sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from source.eegnet.eegnet import EEGNet
from torch.distributions import Categorical


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

from tensorflow.keras import utils as np_utils

# =============================================================================
# SETUP
type = "TRN"
subject = "178" #str(sys.argv[1])
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
downsample = 1

# experiment
seeds = range(10)
train_reps = [3, 5, 8]
experiment = list(it.product(seeds, train_reps))
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
    # -----------------------------------------------------------------------------





    # =============================================================================
    # TRAIN EEGNet
    response = eeg.stimulus.cpu().numpy()
    trny = eeg.stimulus_data["type"].values
    trnX = response
    trnstim = eeg.stimulus_data


    trnX = trnX.reshape((trnX.shape[0], trnX.shape[1], trnX.shape[2], 1))
    trny = np_utils.to_categorical(trny, 2)


    nb_classes=2
    Chans=response.shape[1]
    Samples=response.shape[2]
    dropoutRate=0.5
    kernLength=32
    F1=4
    D=2
    F2=16
    norm_rate=0.25
    dropoutType="Dropout"


    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)
    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 4))(block2)
    block2 = dropoutType(dropoutRate)(block2)
    flatten = Flatten(name='flatten')(block2)
    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)
    eegnet = Model(inputs=input1, outputs=softmax)

    eegnet.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['crossentropy']
    )

    eegnet.fit(
        trnX,
        trny,
        epochs=100,
        batch_size=64,
        validation_split=0.2
    )
    # -----------------------------------------------------------------------------






    # =============================================================================
    # TEST
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


    nchars = eeg.stimulus_data["character"].nunique()
    nreps = eeg.stimulus_data["repetition"].nunique()

    response = eeg.stimulus.cpu().numpy()
    trny = eeg.stimulus_data["type"].values
    trnX = response
    trnstim = eeg.stimulus_data

    trnX = trnX.reshape((trnX.shape[0], trnX.shape[1], trnX.shape[2], 1))
    trny = np_utils.to_categorical(trny, 2)


    proba = eegnet.predict(trnX)[:, 1]
    log_proba = np.log(proba)

    trnstim["log_proba"] = log_proba

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
            cum_log_proba += log_proba_mat
            log_prob[c-1, j, :] = cum_log_proba.flatten().copy()

    log_prob = torch.Tensor(log_prob)
    log_prob -= torch.logsumexp(log_prob, dim=-1, keepdim=True)

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
        "method": "EEGNet",
    }, index=range(1, nreps + 1))
    df.to_csv(dir_results + f"K{subject}_trn{train_reps}_seed{seed}_eegnet.test")
    # -----------------------------------------------------------------------------

    del eeg