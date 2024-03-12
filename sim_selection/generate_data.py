import sys
import torch
import pickle
import itertools as it
import numpy as np
sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.bffmbci.bffm import BFFModel

# =============================================================================
# SETUP
dir_data = "/experiments/sim_selection/data/"

# dimensions
n_channels = 16
n_characters = 19
n_repetitions = 5
n_stimulus = (12, 2)
stimulus_window = 25
stimulus_to_stimulus_interval = 5
n_sequences = n_repetitions * n_characters
n_timepoints = 11 * stimulus_to_stimulus_interval + stimulus_window

# experiments
seeds = range(3)
Kxs = [5, 8]
Kys = [3, 5]
# Ks = range(1, 11)

# combinations
combinations = it.product(seeds, Kxs, Kys)

# model
cor = 0.95
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
    "kernel_gp_loading_processes": (cor, xi_var, 1.),
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
    # DEFINE TRUE GENERATING VALUES
    variables = dict()
    torch.manual_seed(seed)

    variables["observation_variance"] = 5. + 5. * torch.rand(n_channels)

    L = torch.randint(-1, 2, (n_channels, Kx)) * 1.
    Lcolnorm = L.pow(2.).sum(0)
    Lorder = Lcolnorm.sort(descending=True)[1]
    L = L[:, Lorder]
    Lcolnorm = L.pow(2.).sum(0).sqrt()
    L /= Lcolnorm
    colnorm = torch.linspace(20., 3., Kx)
    L *= colnorm
    variables["loadings"] = L

    t = torch.arange(stimulus_window)
    damp = torch.sigmoid(3 * t - 5) * (1. - torch.sigmoid(t - stimulus_window + 8))

    def sine(t, period=8, tshift=3, yshift=0.):
        return yshift + torch.sin((t-tshift)*(2.*torch.pi)/period)

    def bump(t, center=8, width=3):
        p = torch.sigmoid(5.*(t-center)/width)
        return 4. * p * (1-p)

    periods = torch.randint(8, 20, (Kx, ))
    tshifts = torch.randint(0, 20, (Kx, ))
    centers = torch.randint(5, 15, (Kx, ))
    widths = torch.randint(5, 12, (Kx, ))

    nontarget = torch.vstack([
        damp * sine(
            t,
            period,
            tshift,
            0.5 * torch.randn((1,)).item()
        )
        for period, tshift in zip(periods, tshifts)
    ])

    target = torch.vstack([
        damp * sine(
            t,
            period,
            tshift,
            0.5 * torch.randn((1,)).item()
        )
        for period, tshift in zip(periods, tshifts)
    ])

    mixing = torch.vstack([
        bump(t, center, width)
        for center, width in zip(centers, widths)
    ])

    variables["smgp_scaling.nontarget_process"] = nontarget * 0.
    variables["smgp_scaling.target_process"] = target * 0.1
    variables["smgp_scaling.mixing_process"] = mixing
    variables["smgp_factors.nontarget_process"] = nontarget
    variables["smgp_factors.target_process"] = target
    variables["smgp_factors.mixing_process"] = mixing

    # set predictive components to first Ky
    for k in [
        "smgp_scaling.mixing_process",
        "smgp_factors.mixing_process",
    ]:
        for dim in range(Kx):
            if dim >= Ky:
                variables[k][dim, ...] = 0.

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


    # =============================================================================
    # Test set
    seed += 1000
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

