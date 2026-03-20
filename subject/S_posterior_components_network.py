import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    plt.style.use('seaborn-whitegrid')
except OSError:
    plt.style.use('seaborn-v0_8-whitegrid')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.bffmbci import BFFMResults
import networkx as nx

# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_chains = "/home/simon/Documents/BCI/experiments/subject/chains/"
dir_results = "/home/simon/Documents/BCI/experiments/subject/results/"

# experiments
subject = "114"
dir_figures = f"/home/simon/Documents/BCI/experiments/subject/figures/K{subject}/"
os.makedirs(dir_figures, exist_ok=True)
K = 8

# colors
TARGET = "#FFCB05"
NONTARGET = "#00274C"
POSITIVE = "#00A398"
NEGATIVE = "#EF4135"

# file
file_chain = f"K{subject}/K{subject}.chain"

# channels
channels = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

channel_positions = {
    'F3': (1.25, 4.05),
    'Fz': (2, 4),
    'F4': (2.75, 4.05),
    'T7': (0.1, 3),
    'C3': (1.05, 3),
    'Cz': (2, 3),
    'C4': (2.95, 3),
    'T8': (3.9, 3),
    'CP3': (1.1, 2.45),
    'CP4': (2.9, 2.45),
    'P3': (1.25, 1.95),
    'Pz': (2, 2),
    'P4': (2.75, 1.95),
    'PO7': (1., 1.3),
    'Oz': (2, 0.95),
    'PO8': (3., 1.3),
}

extra_channels = {
    "F1": (1.625, 4),
    "F2": (2.375, 4),
    "F5": (0.875, 4.125),
    "F6": (3.125, 4.125),
    "F7": (0.5, 4.25),
    "F8": (3.5, 4.25),
    "F9": (0.125, 4.5),
    "F10": (3.875, 4.5),

    "FT9": (-0.25, 3.875),
    "FT7": (0.2, 3.75),
    "FC5": (0.65, 3.625),
    "FC3": (1.1, 3.55),
    "FC1": (1.5, 3.5),
    "FCz": (2, 3.5),
    "FC2": (2.5, 3.5),
    "FC4": (2.9, 3.55),
    "FC6": (3.35, 3.625),
    "FT8": (3.8, 3.75),
    "FT10": (4.25, 3.875),

    "C5": (0.6, 3.),
    "C1": (1.5, 3.),
    "C2": (2.5, 3.),
    "C6": (3.4, 3.),

    "TP9": (-0.25, 2.125),
    "TP7": (0.2, 2.25),
    "CP5": (0.65, 2.375),
    "CP1": (1.5, 2.5),
    "CPz": (2, 2.5),
    "CP2": (2.5, 2.5),
    "CP6": (3.35, 2.375),
    "TP8": (3.8, 2.25),
    "TP10": (4.25, 2.125),

    "P9": (0.125, 1.5),
    "P7": (0.5, 1.75),
    "P5": (0.875, 1.875),
    "P1": (1.625, 2.),
    "P2": (2.375, 2.),
    "P6": (3.125, 1.875),
    "P8": (3.5, 1.75),
    "P10": (3.875, 1.5),

    "PO9": (0.625, 1.),
    "PO3": (1.5, 1.42),
    "POz": (2., 1.45),
    "PO4": (2.5, 1.42),
    "PO10": (3.375, 1.),

    "O1": (1.5, 1),
    "O2": (2.5, 1),

    "O9": (1.25, 0.7),
    "O10": (2.75, 0.7),
    "Iz": (2., 0.55),

    "AF3": (1.5, 4.45),
    'AFz': (2, 4.5),
    "AF4": (2.5, 4.45),

    "AF7": (0.9, 4.6),
    "Fp1": (1.4, 4.9),
    "Fpz": (2., 5.),
    "Fp2": (2.6, 4.9),
    "AF8": (3.1, 4.6),
}

xrange = (-0.5, 4.5)
yrange = (0.5, 4.5)
# -----------------------------------------------------------------------------


# =============================================================================
# PLOT NODES
file = f"K{subject}_nodes"
fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(5, 4.8),
    sharex="all",
    sharey="all"
)


def draw_head(ax, facecolor="lightgrey"):
    # draw ears
    ear1 = plt.Circle((-0.2, 3.), 0.5, edgecolor='black', fill=True, facecolor=facecolor)
    ear2 = plt.Circle((4.2, 3.), 0.5, edgecolor='black', fill=True, facecolor=facecolor)
    ax.add_artist(ear1)
    ax.add_artist(ear2)
    # draw nose as triangle
    nose = plt.Polygon([[1.5, 5.], [2.5, 5.], [2., 5.75]], edgecolor='black', fill=True, facecolor=facecolor)
    ax.add_artist(nose)
    # draw scalp
    circle = plt.Circle((2, 3), 2.5, edgecolor='black', fill=True, facecolor=facecolor)
    ax.add_artist(circle)


draw_head(ax)

for cname, (x, y) in channel_positions.items():
    ax.plot(x, y, "o", markersize=20, color="black", fillstyle='full', markerfacecolor="black")
    ax.text(x, y, cname, ha="center", va="center", color="white", size=8, fontweight="bold")

for cname, (x, y) in extra_channels.items():
    ax.plot(x, y, "o", markersize=20, color="black", fillstyle='full', markerfacecolor="white")
    ax.text(x, y, cname, ha="center", va="center", size=8, color="grey")


def blank_canvas(ax):
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-0.75, 4.75)
    ax.set_ylim(0., 6.)
    # remove box around
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # remove grid
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


blank_canvas(ax)
plt.tight_layout()
plt.savefig(dir_figures + file + ".pdf")
# ...existing code...


# =============================================================================
# LOAD POSTERIOR
torch.cuda.empty_cache()
results = BFFMResults.from_files(
    [dir_chains + file_chain],
    warmup=0,
    thin=1
)
results.add_transformed_variables()
importance = pd.read_csv(
    dir_results + f"K{subject}/K{subject}_importance.csv",
    index_col=0
)
# the order was reverese when computed
importance["drop_bce"] *= -1
importance["drop_acc"] *= -1
importance["drop_auroc"] *= -1

# channel order
# order = importance.sort_values("posterior", ascending=False)["component"].values
# order = importance.sort_values("drop_auroc", ascending=True)["component"].values
# order = importance.sort_values("drop_acc", ascending=True)["component"].values
order = importance.sort_values("drop_bce", ascending=True)["component"].values
beta_z1 = results.chains["smgp_factors.target_signal.rescaled"]
beta_z0 = results.chains["smgp_factors.nontarget_process.rescaled"]
beta_xi1 = results.chains["smgp_scaling.target_signal"]
beta_xi0 = results.chains["smgp_scaling.nontarget_process"]
diff = beta_z1 * beta_xi1.exp() - beta_z0 * beta_xi0.exp()
mean0 = beta_z0 * beta_xi0.exp()
mean1 = beta_z1 * beta_xi1.exp()
diff_xi = results.chains["smgp_scaling.difference_process"]
diff_xi_scaled = (beta_xi1.exp().pow(2.) - beta_xi0.exp().pow(2.)) * results.chains["loadings.norm"].movedim(2, 3).pow(2.)
mean_diff = mean1 - mean0

# fitted mean
L = results.chains["loadings.norm_one"]  # nc x ns x E x K
Lnorm = results.chains["loadings.norm"] # nc x ns x K
channel_wise_mean_diff = torch.einsum(
    "cbek, cbkt -> cbket",
    L,
    diff
)
channel_wise_mean0 = torch.einsum(
    "cbek, cbkt -> cbket",
    L,
    mean0
)
channel_wise_mean1 = torch.einsum(
    "cbek, cbkt -> cbket",
    L,
    mean1
)

# fitted covariance
L = results.chains["loadings"]
cov1 = torch.einsum(
    "cbek, cbfk, cbkt -> cbeft",
    L,
    L,
    beta_xi1.exp().pow(2.)
)
cov0 = torch.einsum(
    "cbek, cbfk, cbkt -> cbeft",
    L,
    L,
    beta_xi0.exp().pow(2.)
)

channel_wise_cov_diff = (cov1 - cov0)

# fitted correlation
sd1 = cov1.diagonal(dim1=-2, dim2=-3).pow(0.5).permute(0, 1, 3, 2)
sd0 = cov0.diagonal(dim1=-2, dim2=-3).pow(0.5).permute(0, 1, 3, 2)
cor1 = cov1 / sd1.unsqueeze(-2) / sd1.unsqueeze(-3)
cor0 = cov0 / sd0.unsqueeze(-2) / sd0.unsqueeze(-3)
cor1 = torch.nan_to_num(cor1, nan=0.0, posinf=0.0, neginf=0.0)
cor0 = torch.nan_to_num(cor0, nan=0.0, posinf=0.0, neginf=0.0)
channel_wise_cor_diff = cor1 - cor0
channel_wise_cor_diff = torch.nan_to_num(channel_wise_cor_diff, nan=0.0, posinf=0.0, neginf=0.0)


def difference_to_previous_timepoint(values: torch.Tensor) -> torch.Tensor:
    if values.shape[-1] < 2:
        raise ValueError("Need at least two timepoints to compute lag-1 differences")
    out = torch.zeros_like(values)
    out[..., 1:] = values[..., 1:] - values[..., :-1]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


nontarget_mean_prev_diff = difference_to_previous_timepoint(channel_wise_mean0)
nontarget_cor_prev_diff = difference_to_previous_timepoint(cor0)
target_mean_prev_diff = difference_to_previous_timepoint(channel_wise_mean1)
target_cor_prev_diff = difference_to_previous_timepoint(cor1)
lag1_diff_mean = target_mean_prev_diff - nontarget_mean_prev_diff
lag1_diff_cor = torch.nan_to_num(target_cor_prev_diff - nontarget_cor_prev_diff, nan=0.0, posinf=0.0, neginf=0.0)

# compute yrange
ymin = min(mean0.min(), mean1.min()).item()
ymax = max(mean0.max(), mean1.max()).item()

# -----------------------------------------------------------------------------


def confidance_band(
        samples: torch.Tensor,
        level: float = 0.95,
):
    X = samples.reshape(-1, samples.shape[-1])
    m = X.mean(0)
    alpha = 1. - level
    dim = X.shape[-1]
    Xmin = X.quantile(alpha / 2, 0)
    Xmax = X.quantile(1. - alpha / 2, 0)
    qs = reversed(torch.linspace(alpha/(2*dim), alpha/2, 500))
    for q in qs:
        Xmin = X.quantile(q, 0)
        Xmax = X.quantile(1. - q, 0)
        prop = ((Xmin.reshape(1, -1) <= X) & (X <= Xmax.reshape(1, -1))).all(1).float().mean()
        if prop >= level:
            break
    return m, Xmin, Xmax


# =============================================================================
# PLOT COMPONENTS (MEAN ONLY)
L = results.chains["loadings.norm_one"]
file = f"K{subject}_components_mean"
fig, axes = plt.subplots(
    nrows=4+1,
    ncols=4,
    figsize=(10, 9),
    gridspec_kw={
        "width_ratios": [1, 1, 1, 1],
        "height_ratios": [1.5, 1, 1.5, 1, 0.5]
    },
    sharex="row",
    sharey="row"
)
TOP = 8
for j in range(TOP):
    k = order[j]
    col = j % 4
    row = 2 * (j // 4)
    Lnormk = Lnorm[:, :, 0, k].mean((0, 1)).item()

    # network plot
    ax = axes[row, col]
    draw_head(ax, "white")
    component = L[0, :, :, k].mean(0).reshape(-1, 1)
    component_sd = L[0, :, :, k].std(0).reshape(-1, 1)
    excludes = component.abs() / component_sd < 1.96
    colors = [POSITIVE if c > 0 else NEGATIVE for c in component]
    sizes = component.abs().pow(1.).cpu().numpy() * 250
    x = [channel_positions[c][0] for c in channels]
    y = [channel_positions[c][1] for c in channels]
    ax.scatter(x, y, c=colors, s=sizes)
    blank_canvas(ax)


    # scaling difference plot
    row += 1
    ax = axes[row, col]
    ax.set_title(f"Component {k+1}\n"
                 f"Loading norm: {Lnormk:.2f}\n"
                 f"BCE Change: {importance['drop_bce'][k]:.2f}")
    t = np.arange(25) * 31.25
    nontarget_mean, nontarget_min, nontarget_max = confidance_band(mean0[:, :, k, :])
    target_mean, target_min, target_max = confidance_band(mean1[:, :, k, :])

    # highlet where differential
    _, diff_min, diff_max = confidance_band(mean_diff[:, :, k, :], 0.95)
    differential = diff_min.gt(0.) | diff_max.lt(0.)
    for x, d in zip(t, differential):
        if d:
            ax.axvline(x, color="black", linestyle="-", alpha=0.1, linewidth=6.25)

    ax.axhline(0, color="k", linestyle="--")
    ax.fill_between(
        t,
        nontarget_min.cpu().numpy(),
        nontarget_max.cpu().numpy(),
        alpha=0.5,
        color=NONTARGET
    )
    ax.plot(t, nontarget_mean.cpu().numpy(), color=NONTARGET)
    ax.fill_between(
        t,
        target_min.cpu().numpy(),
        target_max.cpu().numpy(),
        alpha=0.5,
        color=TARGET
    )
    ax.plot(t, target_mean.cpu().numpy(), color=TARGET)
    ax.set_xlim(0, 24*31.25)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([0, 150, 300, 450, 600, 750])

    ax.set_xlabel("Time (ms)")
# ADD LEGEND
axes[1, 0].set_ylabel("Mean process")
axes[3, 0].set_ylabel("Mean process")
gs = axes[4, 0].get_gridspec()
for ax in axes[4, :]:
    ax.remove()
axlegend = fig.add_subplot(gs[4, :], frameon=False)
axlegend.axis("off")

# loadings legend
values = torch.Tensor([-1., -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.]).cpu()
xs = torch.arange(-5, 6, 1).cpu()
colors = [POSITIVE if c > 0 else NEGATIVE for c in values]
sizes = values.abs().pow(1.).cpu().numpy() * 250
y = [0 for c in values]
axlegend.scatter(xs, y, c=colors, s=sizes)
for i, v in enumerate(values):
    axlegend.text(xs[i].item(), -0.8, f"{v.item():.2f}", ha="center", va="center")
axlegend.text(-6., 0., "Std. loading", ha="right", va="center")

# curve legends
xs = torch.Tensor([7, 8]).cpu()
for y, which, color, draw_line, alpha in zip(
        [0.8, -0.8, 0.],
        ["Nontarget", "Differential", "Target"],
        [NONTARGET, "black", TARGET],
        [True, False, True],
        [0.5, 0.1, 0.5]
):
    if draw_line:
        axlegend.plot(xs, [y, y], color=color)
    axlegend.fill_between(xs, [y - 0.2, y - 0.2], [y + 0.2, y + 0.2], color=color, alpha=alpha)
    axlegend.text(8.5, y, which, va="center")

# remove box around
axlegend.spines['top'].set_visible(False)
axlegend.spines['right'].set_visible(False)
axlegend.spines['left'].set_visible(False)
axlegend.spines['bottom'].set_visible(False)
# remove grid
axlegend.grid(False)
axlegend.set_xticks([])
axlegend.set_yticks([])

axlegend.set_xlim(-10, 13)
axlegend.set_ylim(-1, 1)

plt.tight_layout()
plt.savefig(dir_figures + file + ".pdf")

