import sys
import os
import torch
import time
import pickle
sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.bffmbci.bffm import BFFModel

# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/experiments/sim_K/data/"

# file
file_post = f"K114_dim10.postmean"

# dimensions
n_channels = 16
n_characters = 19
n_repetitions = 15
n_stimulus = (12, 2)
stimulus_window = 55
stimulus_to_stimulus_interval = 10
n_sequences = n_repetitions * n_characters

for seed in range(10):
    for Ktrue in [2, 5, 8]:

        # model
        cor = 0.8
        shrinkage = 3.
        heterogeneity = 3.
        xi_var = 0.003
        sparse = False
        # -----------------------------------------------------------------------------


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
            "seed": seed
        }

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

        model = BFFModel.generate_from_dimensions(
            latent_dim=Ktrue,
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
            variables[k] = variables[k][..., :Ktrue]
        for k in [
            "smgp_scaling.nontarget_process",
            "smgp_scaling.target_process",
            "smgp_scaling.mixing_process",
            "smgp_factors.nontarget_process",
            "smgp_factors.target_process",
            "smgp_factors.mixing_process"
        ]:
            variables[k] = variables[k][:Ktrue, ...]
        # put in model
        model.set(**variables)
        # generate data
        model.generate_local_variables()
        model.variables["observations"].generate()
        # -----------------------------------------------------------------------------


        # =============================================================================
        # SAVE
        torch.save(model.variables["observations"].data, dir_data + f"dim{Ktrue}_seed{seed}.observations")
        torch.save(model.variables["sequence_data"].order.data, dir_data + f"dim{Ktrue}_seed{seed}.order")
        torch.save(model.variables["sequence_data"].target.data, dir_data + f"dim{Ktrue}_seed{seed}.target")
        # -----------------------------------------------------------------------------
