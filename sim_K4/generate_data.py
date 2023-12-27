import sys
import os
import torch
import time
import pickle
import itertools as it
sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.bffmbci.bffm import BFFModel

# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/experiments/sim_K4/data/"

# file
file_post = f"K114_dim10.postmean"

# dimensions
n_channels = 16
n_characters = 19
n_repetitions = 5
n_stimulus = (12, 2)
stimulus_window = 25
stimulus_to_stimulus_interval = 5
n_sequences = n_repetitions * n_characters

# experiments
seeds = range(3)
Kxs = [5, 8]
Kys = [3, 5]
# Ks = range(1, 11)

# combinations
combinations = it.product(seeds, Kxs, Kys)

# model
cor = 0.8
shrinkage = 3.
heterogeneity = 3.
xi_var = 0.1
sparse = False

prior_parameters = {
    "observation_variance": (1., 10.),
    "heterogeneities": heterogeneity,
    "shrinkage_factor": (1., shrinkage),
    "kernel_gp_factor_processes": (cor, 1., 1.),
    "kernel_tgp_factor_processes": (cor, 0.5, 1.),
    "kernel_gp_loading_processes": (cor, 0.1, 1.),
    "kernel_tgp_loading_processes": (cor, 0.5, 1.),
    "kernel_gp_factor": (cor, 1., 1.)
}
# -----------------------------------------------------------------------------



# =============================================================================
# GENERATE DATA
for seed, Kx, Ky in combinations:


    # =============================================================================
    # INITIALIZE MODEL FOR DATA GENERATION
    settings = {
        "n_channels": n_channels,
        "stimulus_to_stimulus_interval": stimulus_to_stimulus_interval,
        "stimulus_window": stimulus_window,
        "n_stimulus": (12, 2),
        "nonnegative_smgp": False,
        "scaling_activation": "exp",
        "sparse": sparse,
        "seed": seed,
        "n_characters": n_characters,
        "n_repetitions": n_repetitions,
        "latent_dim": Kx,
    }

    model = BFFModel.generate_from_dimensions(
        **settings,
        **prior_parameters
    )
    # -----------------------------------------------------------------------------


    # =============================================================================
    # PUT POST MEAN VALUES THEN REGENERATE DATA
    variables = pickle.load(open(dir_data + file_post, "rb"))
    # to torch
    variables = {
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in variables.items()
    }
    # subset to correct latent dimension
    for k in ["heterogeneities", "shrinkage_factor", "loadings"]:
        variables[k] = variables[k][..., :Kx]
    for k in [
        "smgp_scaling.nontarget_process",
        "smgp_scaling.target_process",
        "smgp_scaling.mixing_process",
        "smgp_factors.nontarget_process",
        "smgp_factors.target_process",
        "smgp_factors.mixing_process"
    ]:
        variables[k] = variables[k][:Kx, ...]
    # sort dimensions
    order = variables["loadings"].pow(2.).sum(0).argsort(descending=True)
    for k in [
        "smgp_scaling.nontarget_process",
        "smgp_scaling.target_process",
        "smgp_scaling.mixing_process",
        "smgp_factors.nontarget_process",
        "smgp_factors.target_process",
        "smgp_factors.mixing_process"
    ]:
        variables[k] = variables[k][order, :]
    for k in ["heterogeneities", "shrinkage_factor", "loadings"]:
        variables[k] = variables[k][..., order]
    # set predictive components to first Ky
    for k in [
        "smgp_scaling.mixing_process",
        "smgp_factors.mixing_process",
    ]:
        for dim in range(Kx):
            if dim > Ky:
                variables[k][dim, ...] = 0.
    # reduce signal, otherwise it is too easy
    for k in [
        "smgp_scaling.mixing_process",
        "smgp_factors.mixing_process",
    ]:
        variables[k] *= 0.5
    # add noise
    variables["observation_variance"] = variables["observation_variance"] + 5.
    # put in model
    model.set(**variables)
    # generate data
    torch.manual_seed(seed)
    model.generate_local_variables()
    model.variables["observations"].generate()
    # -----------------------------------------------------------------------------


    # =============================================================================
    # SAVE
    name = f"Kx{Kx}_Ky{Ky}_seed{seed}"
    torch.save(model.variables["observations"].data, dir_data + name + ".observations")
    torch.save(model.variables["sequence_data"].order.data, dir_data + name + ".order")
    torch.save(model.variables["sequence_data"].target.data, dir_data + name + ".target")
    with open(dir_data + name + ".variables", "wb") as f:
        pickle.dump(variables, f)
    # -----------------------------------------------------------------------------
