import os
import sys
import pickle
import itertools as it

import torch

sys.path.insert(1, "/home/simon/Documents/BCI")
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci.bffm import BFFModel


# =============================================================================
# SETUP
DIR_DATA = "/home/simon/Documents/BCI/experiments/sim_gaussianity/data/"
os.makedirs(DIR_DATA, exist_ok=True)

# dimensions
n_channels = 16
n_characters = 19
n_repetitions = 5
stimulus_window = 25
stimulus_to_stimulus_interval = 5

# experiments
seeds = [0]
Kxs = [8]
Kys = [5]
noise_settings = [
    ("gaussian", None),
    ("student_t", 20.0),
    ("student_t", 10.0),
    ("student_t", 5.0),
    ("student_t", 3.0),
]

# combinations
combinations = it.product(seeds, Kxs, Kys, noise_settings)

# model
cor = 0.95
shrinkage = 3.0
heterogeneity = 3.0
xi_var = 0.1
sparse = False

prior_parameters = {
    "observation_variance": (1.0, 10.0),
    "heterogeneities": heterogeneity,
    "shrinkage_factor": (1.0, shrinkage),
    "kernel_gp_factor_processes": (cor, 1.0, 1.0),
    "kernel_tgp_factor_processes": (cor, 0.5, 1.0),
    "kernel_gp_loading_processes": (cor, xi_var, 1.0),
    "kernel_tgp_loading_processes": (cor, 0.5, 1.0),
    "kernel_gp_factor": (cor, 1.0, 1.0),
}
# -----------------------------------------------------------------------------


def _df_tag(df):
    if df is None:
        return "na"
    return str(df).replace(".", "p")


def _name(seed, kx, ky, noise_distribution, df):
    return (
        f"Kx{kx}_Ky{ky}_seed{seed}_noise{noise_distribution}_df{_df_tag(df)}"
    )


# =============================================================================
# GENERATE DATA
for seed, kx, ky, (noise_distribution, df) in combinations:
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
        "latent_dim": kx,
    }

    model = BFFModel.generate_from_dimensions(
        **settings,
        **prior_parameters,
    )

    variables = {}
    torch.manual_seed(7)

    variables["observation_variance"] = 5.0 + 5.0 * torch.rand(n_channels)

    loadings = torch.randint(-1, 2, (n_channels, kx)).float()
    order = loadings.pow(2.0).sum(0).sort(descending=True)[1]
    loadings = loadings[:, order]
    loadings = loadings / loadings.pow(2.0).sum(0).sqrt()
    loadings = loadings * torch.linspace(20.0, 3.0, kx)
    variables["loadings"] = loadings

    t = torch.arange(stimulus_window)
    damp = torch.sigmoid(3 * t - 5) * (1.0 - torch.sigmoid(t - stimulus_window + 8))

    def sine(time, period=8, tshift=3, yshift=0.0):
        return yshift + torch.sin((time - tshift) * (2.0 * torch.pi) / period)

    def bump(time, center=8, width=3):
        p = torch.sigmoid(5.0 * (time - center) / width)
        return 4.0 * p * (1 - p)

    periods = torch.randint(8, 20, (kx,))
    tshifts = torch.randint(0, 20, (kx,))
    centers = torch.randint(5, 15, (kx,))
    widths = torch.randint(5, 12, (kx,))

    nontarget = torch.vstack(
        [
            damp * sine(time=t, period=period, tshift=tshift, yshift=0.5 * torch.randn((1,)).item())
            for period, tshift in zip(periods, tshifts)
        ]
    )
    target = torch.vstack(
        [
            damp * sine(time=t, period=period, tshift=tshift, yshift=0.5 * torch.randn((1,)).item())
            for period, tshift in zip(periods, tshifts)
        ]
    )
    mixing = torch.vstack([bump(time=t, center=center, width=width) for center, width in zip(centers, widths)])


    # readjust signal strength
    mixing[0, :] *= 0.65
    target[2, :] *= -1

    variables["smgp_scaling.nontarget_process"] = nontarget.detach().clone() * -0.1
    variables["smgp_scaling.target_process"] = target.detach().clone() * 0.4
    variables["smgp_scaling.mixing_process"] = mixing.detach().clone()
    variables["smgp_factors.nontarget_process"] = nontarget.detach().clone() * 0.3
    variables["smgp_factors.target_process"] = target.detach().clone()
    variables["smgp_factors.mixing_process"] = mixing.detach().clone()

    # set predictive components to first Ky
    for k in [
        "smgp_scaling.mixing_process",
        "smgp_factors.mixing_process",
    ]:
        for dim in range(kx):
            if dim >= ky or (dim % 2 == 1):
                variables[k][dim, ...] = 0.

    model.set(**variables)

    # Train split.
    torch.manual_seed(seed)
    model.generate_local_variables()
    model.variables["observations"].generate(
        noise_distribution=noise_distribution,
        df=df,
    )
    name = _name(seed, kx, ky, noise_distribution, df)
    mean = model.variables["observations"].mean().detach().clone()

    torch.save(model.variables["observations"].data, DIR_DATA + name + ".observations")
    torch.save(model.variables["sequence_data"].order.data, DIR_DATA + name + ".order")
    torch.save(model.variables["sequence_data"].target.data, DIR_DATA + name + ".target")
    torch.save(mean, DIR_DATA + name + ".mean")
    with open(DIR_DATA + name + ".variables", "wb") as handle:
        pickle.dump(variables, handle)

    # Test split.
    seed_test = seed + 1000
    torch.manual_seed(seed_test)
    model.generate_local_variables()
    model.variables["observations"].generate(
        noise_distribution=noise_distribution,
        df=df,
    )
    name = _name(seed_test, kx, ky, noise_distribution, df)
    mean = model.variables["observations"].mean().detach().clone()

    torch.save(model.variables["observations"].data, DIR_DATA + name + ".observations")
    torch.save(model.variables["sequence_data"].order.data, DIR_DATA + name + ".order")
    torch.save(model.variables["sequence_data"].target.data, DIR_DATA + name + ".target")
    torch.save(mean, DIR_DATA + name + ".mean")
    with open(DIR_DATA + name + ".variables", "wb") as handle:
        pickle.dump(variables, handle)

