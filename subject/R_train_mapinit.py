import sys
import os
import torch
import time
import pickle
import itertools as it
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from source.bffmbci.bffm import BFFModel
from source.bffmbci.bffm import DynamicRegressionCovarianceRegressionMean
from source.bffmbci.bffm import DynamicCovarianceRegressionMean
from source.bffmbci.bffm import StaticCovarianceRegressionMean
from source.bffmbci.bffm_map import BFFModelMAP
from source.bffmbci.bffm_map import DynamicRegressionCovarianceRegressionMeanMAP
from source.bffmbci.bffm_map import DynamicCovarianceRegressionMeanMAP
from source.bffmbci.bffm_map import StaticCovarianceRegressionMeanMAP
from source.nb_mn import NaiveBayesMatrixNormal

# =============================================================================
# SETUP
type = "TRN"
subject = str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
dir_data = "/home/simfont/Documents/BCI/K_Protocol/"
dir_chains = f"/home/simfont/Documents/BCI/experiments/subject/chains/K{subject}/"
os.makedirs(dir_chains, exist_ok=True)
filename = dir_data + name + ".mat"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

# model
lite = True
seed = 0
K = 8
V = "LR-SC" if lite else "LR-DCR"
cor = 0.50
n_iter = 20_000


# experiment
seeds = range(10)
train_reps = [3, 5, 7]
experiment = list(it.product(train_reps, seeds))
experiment.append((7, "even"))
experiment.append((7, "odd"))
train_reps, seed = experiment[int(sys.argv[2])]

# -----------------------------------------------------------------------------


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
    training_reps = list(range(2, 16, 2))
    testing_reps = list(range(3, 17, 2))
elif seed == "odd":
    training_reps = list(range(3, 16, 2))
    testing_reps = list(range(2, 16, 2))
else:
    raise ValueError("Seed not recognized")
eeg = eeg.repetitions(training_reps)
# -----------------------------------------------------------------------------


# =============================================================================
# PARAMETERS
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
# -----------------------------------------------------------------------------



# =============================================================================
# MAP INITIALIZATION
ModelMAP = {
    "LR-DCR": DynamicRegressionCovarianceRegressionMeanMAP,
    "LR-DC": DynamicCovarianceRegressionMeanMAP,
    "LR-SC": StaticCovarianceRegressionMeanMAP,
}["LR-SC"]

modelMAP: BFFModelMAP = ModelMAP(
    sequences=eeg.sequence,
    stimulus_order=eeg.stimulus_order,
    target_stimulus=eeg.target,
    **settings,
    **prior_parameters
)

modelMAP.initialize()
modelMAP.fit(lr=0.1, max_iter=2000, tol=1e-8)
variablesMAP = modelMAP.export_variables()
# -----------------------------------------------------------------------------



# =============================================================================
# INITIALIZE MODEL
Model = {
    "LR-DCR": DynamicRegressionCovarianceRegressionMean,
    "LR-DC": DynamicCovarianceRegressionMean,
    "LR-SC": StaticCovarianceRegressionMean,
}[V]

model = Model(
    sequences=eeg.sequence,
    stimulus_order=eeg.stimulus_order,
    target_stimulus=eeg.target,
    **settings,
    **prior_parameters
)
# -----------------------------------------------------------------------------



# =============================================================================
# INITIALIZE CHAIN TO MAP
model.set(**variablesMAP)
model.clear_history()
# -----------------------------------------------------------------------------



# =============================================================================
# RUN CHAIN
torch.manual_seed(seed if isinstance(seed, int) else 0)
t0 = time.time()
t00 = t0
for i in range(n_iter):
    model.sample()
    if i % 100 == 0:
        print(f"{i:>10} "
              f"{model.variables['observations']._log_density_history[-1]:>20.4f}"
              f"  dt={time.time() - t00:>20.4f}   elapsed={time.time() - t0:20.4f}")
        t00 = time.time()
# -----------------------------------------------------------------------------



# =============================================================================
# SAVE CHAIN
out = model.results(
    start=10_000,
    thin=10
)
with open(dir_chains + f"K{subject}_trn{train_reps}_seed{seed}{'_lite' if lite else ''}_mapinit.chain", "wb") as f:
    pickle.dump(out, f)
# -----------------------------------------------------------------------------

