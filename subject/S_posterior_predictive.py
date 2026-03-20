import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import torch

try:
    plt.style.use('seaborn-whitegrid')
except OSError:
    plt.style.use('seaborn-v0_8-whitegrid')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci import BFFMResults
from source.bffmbci.bffm import DynamicRegressionCovarianceRegressionMean
from source.bffmbci.bffm import CompoundSymmetryCovarianceRegressionMean
from source.data.k_protocol import KProtocol


# =============================================================================
# SETUP
type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"

dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_chains = f"/home/simon/Documents/BCI/experiments/subject/chains/K{subject}/"
dir_figures = f"/home/simon/Documents/BCI/experiments/subject/figures/K{subject}/"
os.makedirs(dir_figures, exist_ok=True)

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8

# model
seed = 0
K = 8
cor = 0.5
n_iter = 20_000
Model = DynamicRegressionCovarianceRegressionMean

# dimensions
n_characters = 19
n_repetitions = 15
n_posterior_predictive_samples = 100

# colors
TARGET = "#FFCB05"
NONTARGET = "#00274C"
POSITIVE = "#00A398"
NEGATIVE = "#EF4135"

# file
file_chain = f"K{subject}.chain"

# channels
channels = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

channel_positions = {
    'F3': (0, 1),
    'Fz': (0, 2),
    'F4': (0,3),
    'T7': (1, 0),
    'C3': (1, 1),
    'Cz': (1, 2),
    'C4': (1, 3),
    'T8': (1, 4),
    'CP3': (2, 1),
    'CP4': (2, 3),
    'P3': (3, 1),
    'Pz': (3, 2),
    'P4': (3, 3),
    'PO7': (4, 1),
    'Oz': (4, 2),
    'PO8': (4, 3),
}
# -----------------------------------------------------------------------------



# =============================================================================
# LOAD DATA
filename = dir_data + name + ".mat"
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
# LOAD POSTERIOR
torch.cuda.empty_cache()
posterior = BFFMResults.from_files(
    [dir_chains + file_chain],
    warmup=0,
    thin=1
)
# posterior.add_transformed_variables()
# sample ids at random
torch.manual_seed(seed)
sample_ids = torch.randperm(posterior.n_samples)[:n_posterior_predictive_samples]
# -----------------------------------------------------------------------------


# =============================================================================
# UTILITY FUNCTIONS
def sequence_to_stimulus(X):
    T = eeg.stimulus_window
    E = eeg.sequence.shape[1]
    d = eeg.stimulus_to_stimulus_interval
    R = eeg.sequence.shape[0]
    L = 12
    # take the subblock of X of length T every d for L times
    # into a new tensor of shape (R*L, E, T)
    Xlong = torch.stack([X[:, :, i*d:i*d+T] for i in range(L)], dim=1).reshape(R*L, E, T)
    return Xlong


def subset_to(Xlong, Ylong, y=0):
    idx = (Ylong == y).nonzero().squeeze()
    return Xlong[idx]


def integrated_autocovariance_matrix(X, lag=1):
    # X is of shape (n_samples, n_channels, n_timepoints)
    # return the autocovariance matrix of shape (n_channels, n_channels)
    n_samples, n_channels, n_timepoints = X.shape
    X = X - X.mean(dim=0, keepdim=True)
    cov = torch.zeros(n_channels, n_channels)
    for i in range(n_samples):
        for t in range(n_timepoints - lag):
            cov += torch.ger(X[i, :, t], X[i, :, t+lag])
    cov /= (n_samples * (n_timepoints - lag))
    return cov


def covariance_matrix(X):
    # X is of shape (n_samples, n_channels, n_timepoints)
    # return covariance over time of shape (n_channels, n_channels, n_timepoints)
    n_samples, n_channels, n_timepoints = X.shape
    X = X - X.mean(dim=0, keepdim=True)
    cov = torch.einsum("nct,ndt->cdt", X, X)
    cov /= n_samples
    return cov


def correlation_matrix(X, eps=1e-8):
    # X is of shape (n_samples, n_channels, n_timepoints)
    # return correlation over time of shape (n_channels, n_channels, n_timepoints)
    n_samples, n_channels, n_timepoints = X.shape
    X = X - X.mean(dim=0, keepdim=True)
    cov = torch.einsum("nct,ndt->cdt", X, X)
    cov /= n_samples
    var = cov[torch.arange(n_channels), torch.arange(n_channels), :].clamp_min(eps)
    std = torch.sqrt(var)
    corr = cov / (std[:, None, :] * std[None, :, :] + eps)
    return corr


def channelwise_autocorrelation(X, max_lag=10, eps=1e-8):
    # X is of shape (n_samples, n_channels, n_timepoints)
    # return pooled empirical ACF of shape (n_channels, max_lag + 1)
    n_samples, n_channels, n_timepoints = X.shape
    if max_lag >= n_timepoints:
        raise ValueError(f"max_lag ({max_lag}) must be < n_timepoints ({n_timepoints})")

    # Center each sample/channel across replications, then pool over sample-time pairs.
    Xc = X - X.mean(dim=0, keepdim=True)
    var0 = (Xc * Xc).mean(dim=(0, 2))

    acf_lags = []
    for lag in range(max_lag + 1):
        num = (Xc[:, :, :n_timepoints-lag] * Xc[:, :, lag:]).mean(dim=(0, 2))
        acf_lags.append(num / (var0 + eps))

    return torch.stack(acf_lags, dim=-1)


def residual_ecdf(X, value_grid):
    # X is of shape (n_samples, n_channels, n_timepoints)
    # value_grid is 1D tensor of evaluation points shared across datasets
    n_samples, n_channels, n_timepoints = X.shape
    X = X - X.mean(dim=0, keepdim=True)
    flat = X.permute(1, 0, 2).reshape(n_channels, -1)
    ecdf = (flat[:, :, None] <= value_grid[None, None, :]).float().mean(dim=1)
    return ecdf


def make_ecdf_grid(X, n_points=101):
    return torch.linspace(-25., 25., n_points)
# -----------------------------------------------------------------------------



# =============================================================================
# STATISTICS
X = eeg.sequence
W = eeg.stimulus_order
Y = eeg.target
# eeg.sequence is 285 x 16 x 80
# eeg.stimulus_order is 285 x 12 and takes values 0--11
# eeg.target.shape is 285 x 12 and takes values 0/1
Xlong = sequence_to_stimulus(X)
Ylong = torch.Tensor(eeg.stimulus_data["type"].values)
# eeg.stimulus is 285*12 x 16 x 25
# Ylong is 285*12

# statistics are function that take X, W, Y, Ylong, Xlong and return a tensor

acf_max_lag = 10
x_target = subset_to(Xlong, Ylong, y=1)
x_nontarget = subset_to(Xlong, Ylong, y=0)
ecdf_grid_target = make_ecdf_grid(x_target, n_points=101)
ecdf_grid_nontarget = make_ecdf_grid(x_nontarget, n_points=101)
mean_target = lambda X, W, Y, Xlong, Ylong: subset_to(Xlong, Ylong, y=1).mean(dim=0)
mean_nontarget = lambda X, W, Y, Xlong, Ylong: subset_to(Xlong, Ylong, y=0).mean(dim=0)
q5_target = lambda X, W, Y, Xlong, Ylong: subset_to(Xlong, Ylong, y=1).quantile(0.05, dim=0)
q95_target = lambda X, W, Y, Xlong, Ylong: subset_to(Xlong, Ylong, y=1).quantile(0.95, dim=0)
q5_nontarget = lambda X, W, Y, Xlong, Ylong: subset_to(Xlong, Ylong, y=0).quantile(0.05, dim=0)
q95_nontarget = lambda X, W, Y, Xlong, Ylong: subset_to(Xlong, Ylong, y=0).quantile(0.95, dim=0)
acf_target = lambda X, W, Y, Xlong, Ylong: channelwise_autocorrelation(
    subset_to(Xlong, Ylong, y=1), max_lag=acf_max_lag
)
acf_nontarget = lambda X, W, Y, Xlong, Ylong: channelwise_autocorrelation(
    subset_to(Xlong, Ylong, y=0), max_lag=acf_max_lag
)
ecdf_target = lambda X, W, Y, Xlong, Ylong: residual_ecdf(
    subset_to(Xlong, Ylong, y=1), value_grid=ecdf_grid_target
)
ecdf_nontarget = lambda X, W, Y, Xlong, Ylong: residual_ecdf(
    subset_to(Xlong, Ylong, y=0), value_grid=ecdf_grid_nontarget
)
corr_target = lambda X, W, Y, Xlong, Ylong: correlation_matrix(subset_to(Xlong, Ylong, y=1))
corr_nontarget = lambda X, W, Y, Xlong, Ylong: correlation_matrix(subset_to(Xlong, Ylong, y=0))
statistics = {
    "channelwise_mean_target": mean_target,
    "channelwise_mean_nontarget": mean_nontarget,
    # "channelwise_q5_target": q5_target,
    # "channelwise_q95_target": q95_target,
    # "channelwise_q5_nontarget": q5_nontarget,
    # "channelwise_q95_nontarget": q95_nontarget,
    # "channelwise_acf_target": acf_target,
    # "channelwise_acf_nontarget": acf_nontarget,
    # "channelwise_ecdf_target": ecdf_target,
    # "channelwise_ecdf_nontarget": ecdf_nontarget,
    "channelwise_corr_target": corr_target,
    "channelwise_corr_nontarget": corr_nontarget,
}
# -----------------------------------------------------------------------------



# =============================================================================
# INITIALIZE MODEL
model = Model(
    sequences=eeg.sequence,
    stimulus_order=eeg.stimulus_order,
    target_stimulus=eeg.target,
    **settings,
    **prior_parameters
)
# -----------------------------------------------------------------------------




# =============================================================================
# POSTERIOR PREDICTIVE STATISTICS
def compute_posterior_predictive_statistic(X, W, Y, Xlong, Ylong):
    return {name: stat(X, W, Y, Xlong, Ylong) for name, stat in statistics.items()}
# True data (already in model)
true_statistic = compute_posterior_predictive_statistic(X, W, Y, Xlong, Ylong)
# Posterior resampled
posterior_predictive_statistics = []
for i, sample_id in enumerate(sample_ids):
    print(f"Computing posterior predictive statistic for sample {sample_id} ({i}/{n_posterior_predictive_samples})")
    model.set(**posterior.get(sample_id))
    model.generate_local_variables()
    model.variables["observations"].generate()
    X = model.variables["observations"].data
    Xlong = sequence_to_stimulus(X)
    posterior_predictive_statistics.append(
        compute_posterior_predictive_statistic(X, W, Y, Xlong, Ylong)
    )
# Concatenate along posterior samples into a dictionary of tensors of shape (n_posterior_predictive_samples, ...)
posterior_predictive_statistics = {
    name: torch.stack([stat[name] for stat in posterior_predictive_statistics], dim=0)
    for name in statistics.keys()
}
# Posterior predictive p-values
posterior_predictive_p_values = {
    name: ((posterior_predictive_statistics[name] - true_statistic[name]) > 0).float().mean(dim=0)
    for name in statistics.keys()
}
# -----------------------------------------------------------------------------




# =============================================================================
# PLOT 1: posterior predictive mean and quantile functions (mean, q5, q95)
# Arrange channels in a grid according to channel_positions (row, col)
which = "target"
if which not in {"target", "nontarget"}:
    raise ValueError(f"Unsupported value for `which`: {which}")

base_sampling_rate = 256.0
if which == "target":
    n_timepoints = true_statistic["channelwise_mean_target"].shape[-1]
else:
    n_timepoints = true_statistic["channelwise_mean_nontarget"].shape[-1]
time_ms = np.arange(n_timepoints) * (1000.0 * downsample / base_sampling_rate)

fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=True, sharey=True)
used_positions = set(channel_positions.values())
legend_position = (0, 0)
for r in range(5):
    for c in range(5):
        if (r, c) not in used_positions and (r, c) != legend_position:
            axes[r, c].axis("off")

legend_ax = axes[legend_position]
legend_ax.axis("off")

pp_mean_color = TARGET
for channel, (row, col) in channel_positions.items():
    ax = axes[row, col]
    channel_idx = channels.index(channel)

    if which == "target":
        true_mean = true_statistic["channelwise_mean_target"][channel_idx, :]
        true_q5 = true_statistic["channelwise_q5_target"][channel_idx, :]
        true_q95 = true_statistic["channelwise_q95_target"][channel_idx, :]
        pp_mean = posterior_predictive_statistics["channelwise_mean_target"][:, channel_idx, :]
        pp_q5 = posterior_predictive_statistics["channelwise_q5_target"][:, channel_idx, :]
        pp_q95 = posterior_predictive_statistics["channelwise_q95_target"][:, channel_idx, :]
    else:
        true_mean = true_statistic["channelwise_mean_nontarget"][channel_idx, :]
        true_q5 = true_statistic["channelwise_q5_nontarget"][channel_idx, :]
        true_q95 = true_statistic["channelwise_q95_nontarget"][channel_idx, :]
        pp_mean = posterior_predictive_statistics["channelwise_mean_nontarget"][:, channel_idx, :]
        pp_q5 = posterior_predictive_statistics["channelwise_q5_nontarget"][:, channel_idx, :]
        pp_q95 = posterior_predictive_statistics["channelwise_q95_nontarget"][:, channel_idx, :]

    # Light posterior-predictive draws
    ax.plot(time_ms, pp_mean.cpu().T, color=pp_mean_color, alpha=0.06, linewidth=2)
    ax.plot(time_ms, pp_q5.cpu().T, color=NEGATIVE, alpha=0.08, linewidth=2)
    ax.plot(time_ms, pp_q95.cpu().T, color=POSITIVE, alpha=0.08, linewidth=2)

    # True curves in stronger colors
    ax.plot(time_ms, true_mean.cpu(), color="black", linewidth=2, linestyle="--")
    ax.plot(time_ms, true_q5.cpu(), color="black", linewidth=2, linestyle="--")
    ax.plot(time_ms, true_q95.cpu(), color="black", linewidth=2, linestyle="--")

    ax.set_title(channel)
    if row == 4:
        ax.set_xlabel("Time (ms)")
    if col == 0:
        ax.set_ylabel("Amplitude")

legend_handles = [
    Line2D([0], [0], color=pp_mean_color, linewidth=2, alpha=0.8, label="PP mean"),
    Line2D([0], [0], color=NEGATIVE, linewidth=2, alpha=0.8, label="PP q5"),
    Line2D([0], [0], color=POSITIVE, linewidth=2, alpha=0.8, label="PP q95"),
    Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Observed"),
]
legend_ax.legend(handles=legend_handles, loc="center", frameon=True)
fig.suptitle(f"Posterior predictive ({which})", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig(f"{dir_figures}posterior_predictive_marginal_{which}.pdf")


# =============================================================================
# PLOT 2: posterior predictive ACF by channel

which = "target"
if which == "target":
    true_acf = true_statistic["channelwise_acf_target"]
    pp_acf = posterior_predictive_statistics["channelwise_acf_target"]
else:
    true_acf = true_statistic["channelwise_acf_nontarget"]
    pp_acf = posterior_predictive_statistics["channelwise_acf_nontarget"]

lag_axis = np.arange(pp_acf.shape[-1])
acf_color = TARGET

fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=True, sharey=True)
used_positions = set(channel_positions.values())
legend_position = (0, 0)
for r in range(5):
    for c in range(5):
        if (r, c) not in used_positions and (r, c) != legend_position:
            axes[r, c].axis("off")

legend_ax = axes[legend_position]
legend_ax.axis("off")

for channel, (row, col) in channel_positions.items():
    ax = axes[row, col]
    channel_idx = channels.index(channel)

    ax.plot(lag_axis, pp_acf[:, channel_idx, :].cpu().T, color=acf_color, alpha=0.08, linewidth=2)
    ax.plot(lag_axis, true_acf[channel_idx, :].cpu(), color="black", linewidth=2, linestyle="--")

    ax.set_title(channel)
    ax.set_ylim(-0.1, 1.0)
    if row == 4:
        ax.set_xlabel("Lag")
    if col == 0:
        ax.set_ylabel("ACF")

legend_handles = [
    Line2D([0], [0], color=acf_color, linewidth=2, alpha=0.8, label="PP ACF"),
    Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Observed ACF"),
]
legend_ax.legend(handles=legend_handles, loc="center", frameon=True)
fig.suptitle(f"Posterior predictive ACF ({which})", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig(f"{dir_figures}posterior_predictive_acf_{which}.pdf")


# =============================================================================
# PLOT 2b: posterior predictive ECDF by channel
which = "target"
if which == "target":
    true_ecdf = true_statistic["channelwise_ecdf_target"]
    pp_ecdf = posterior_predictive_statistics["channelwise_ecdf_target"]
    ecdf_color = TARGET
    ecdf_x = ecdf_grid_target.cpu().numpy()
else:
    true_ecdf = true_statistic["channelwise_ecdf_nontarget"]
    pp_ecdf = posterior_predictive_statistics["channelwise_ecdf_nontarget"]
    ecdf_color = NONTARGET
    ecdf_x = ecdf_grid_nontarget.cpu().numpy()

fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=True, sharey=True)
used_positions = set(channel_positions.values())
legend_position = (0, 0)
for r in range(5):
    for c in range(5):
        if (r, c) not in used_positions and (r, c) != legend_position:
            axes[r, c].axis("off")

legend_ax = axes[legend_position]
legend_ax.axis("off")

for channel, (row, col) in channel_positions.items():
    ax = axes[row, col]
    channel_idx = channels.index(channel)

    ax.plot(ecdf_x, pp_ecdf[:, channel_idx, :].cpu().T, color=ecdf_color, alpha=0.08, linewidth=2)
    ax.plot(ecdf_x, true_ecdf[channel_idx, :].cpu(), color="black", linewidth=2, linestyle="--")

    ax.set_title(channel)
    ax.set_ylim(0.0, 1.0)
    if row == 4:
        ax.set_xlabel("Residual value")
    if col == 0:
        ax.set_ylabel("ECDF")

legend_handles = [
    Line2D([0], [0], color=ecdf_color, linewidth=2, alpha=0.8, label="PP ECDF"),
    Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Observed ECDF"),
]
legend_ax.legend(handles=legend_handles, loc="center", frameon=True)
fig.suptitle(f"Posterior predictive ECDF ({which})", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.97))
# plt.show()
plt.savefig(f"{dir_figures}posterior_predictive_ecdf_{which}.pdf")


# =============================================================================
# PLOT 3: channel-pair correlation over time (upper: target, lower: nontarget)
channel_subset = ["Fz", "Cz", "T7", "T8", "PO7", "PO8"]
channel_subset = ["PO7", "PO8", "Pz", "C3", "C4"]
subset_idx = [channels.index(ch) for ch in channel_subset]
n_subset = len(channel_subset)

true_corr_target = true_statistic["channelwise_corr_target"][subset_idx][:, subset_idx, :]
pp_corr_target = posterior_predictive_statistics["channelwise_corr_target"][:, subset_idx][:, :, subset_idx, :]
true_corr_nontarget = true_statistic["channelwise_corr_nontarget"][subset_idx][:, subset_idx, :]
pp_corr_nontarget = posterior_predictive_statistics["channelwise_corr_nontarget"][:, subset_idx][:, :, subset_idx, :]

fig, axes = plt.subplots(n_subset, n_subset, figsize=(1.8 * n_subset, 1.8 * n_subset), sharex=True, sharey=True)
legend_diag_idx = n_subset // 2
legend_ax = axes[legend_diag_idx, legend_diag_idx]
for i in range(n_subset):
    for j in range(n_subset):
        ax = axes[i, j]

        if i == j:
            if i == legend_diag_idx:
                ax.axis("off")
            else:
                ax.axis("off")
            continue

        if i < j:
            pp_curve = pp_corr_target[:, i, j, :]
            true_curve = true_corr_target[i, j, :]
            color = TARGET
        else:
            pp_curve = pp_corr_nontarget[:, i, j, :]
            true_curve = true_corr_nontarget[i, j, :]
            color = NONTARGET
        ax.axhline(y=0, color="black", linestyle="-")
        ax.plot(time_ms, pp_curve.cpu().T, color=color, alpha=0.08, linewidth=1.5)
        ax.plot(time_ms, true_curve.cpu(), color="black", linewidth=2, linestyle="--")
        ax.set_ylim(-0.2, 1.0)

        if i == 0:
            ax.set_title(channel_subset[j])
        # Diagonal is hidden, so place the first row label on the first visible panel.
        if j == 0 or (i == 0 and j == 1):
            ax.set_ylabel(channel_subset[i])
        # Diagonal is hidden, so place the first column title on the first visible panel.
        if j == 0 and i == 1:
            ax.set_title(channel_subset[0])
        if i == n_subset - 1:
            ax.set_xlabel("Time (ms)")

legend_handles = [
    Line2D([0], [0], color=TARGET, linewidth=2, alpha=0.8, label="PP target"),
    Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Observed"),
    Line2D([0], [0], color=NONTARGET, linewidth=2, alpha=0.8, label="PP nontarget"),
]
if legend_ax is not None:
    legend_ax.legend(handles=legend_handles, loc="center", frameon=True)
# fig.suptitle("Posterior predictive correlation over time (upper target, lower nontarget)", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 1.))
plt.savefig(f"{dir_figures}posterior_predictive_correlation_triangular.pdf")
# -----------------------------------------------------------------------------
