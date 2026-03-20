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

xrange = (-0.5, 4.5)
yrange = (0.5, 4.5)
# -----------------------------------------------------------------------------


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


def plot_mean_and_correlation_over_time(
        file: str,
        node_values: torch.Tensor,
        edge_values: torch.Tensor,
        node_label: str,
        edge_label: str,
        node_legend_values: torch.Tensor,
        edge_legend_values: torch.Tensor,
        node_scale: float = 25.,
        edge_scale: float = 50.,
        difference_titles: bool = False,
):
    if node_scale < 0 or edge_scale < 0:
        raise ValueError("node_scale and edge_scale must be non-negative")
    ts = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ncols = 5
    nrows = len(ts) // ncols
    edge_alpha_scale = min(edge_scale / 10., 5.)
    fig, axes = plt.subplots(
        nrows=nrows + 1,
        ncols=ncols,
        figsize=(12, nrows * 2.5 + 1),
        gridspec_kw={
            "width_ratios": [1] * ncols,
            "height_ratios": [1] * nrows + [0.5]
        },
        sharex="all",
        sharey="all"
    )
    for plot_idx, t in enumerate(ts):
        col = plot_idx % ncols
        row = plot_idx // ncols
        time = round(t * 31.25)
        previous_time = round((t - 1) * 31.25)

        ax = axes[row, col]
        draw_head(ax, "white")

        network = edge_values[0, :, :, :, t].mean(0)
        network_sd = edge_values[0, :, :, :, t].std(0)
        exclude = torch.zeros_like(network, dtype=torch.bool)
        valid = network_sd.gt(0)
        exclude[valid] = network[valid].abs() / network_sd[valid] < 1.96
        network = network.clone()
        network[~torch.isfinite(network)] = 0
        network[exclude] = 0

        component = node_values[0, :, :, :, t].sum(1).mean(0).reshape(-1, 1)
        colors = [POSITIVE if c > 0 else NEGATIVE for c in component]
        sizes = component.abs().pow(1.).cpu().numpy() * node_scale

        edgelist = torch.tril_indices(16, 16, offset=-1)
        weights = network[edgelist[0], edgelist[1]]
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        alphas = weights.abs().pow(1.).cpu().numpy().clip(0., 0.2) * edge_alpha_scale
        G = nx.from_edgelist(edgelist.T.cpu().numpy(), create_using=nx.DiGraph)
        for edge_idx, (u, v) in enumerate(G.edges()):
            G.edges[u, v]['weight'] = weights[edge_idx].item()
        G = nx.relabel_nodes(G, {idx: channels[idx] for idx in range(16)})
        nx.draw_networkx_nodes(
            G,
            pos=channel_positions,
            node_size=sizes,
            node_color=colors,
            edgecolors='k',
            linewidths=0,
            ax=ax
        )
        nx.draw_networkx_edges(
            G,
            pos=channel_positions,
            width=weights.abs().pow(1.).cpu().numpy() * edge_scale,
            edge_color=[POSITIVE if w > 0 else NEGATIVE for w in weights],
            connectionstyle='arc3, rad=0.1',
            arrowsize=20,
            arrowstyle='-',
            alpha=alphas,
            ax=ax
        )

        blank_canvas(ax)
        if difference_titles:
            ax.set_title(f"{previous_time}→{time} ms")
        else:
            ax.set_title(f"{time} ms")

    gs = axes[nrows, 0].get_gridspec()
    for ax in axes[nrows, :]:
        ax.remove()
    axlegend = fig.add_subplot(gs[nrows, :], frameon=False)
    axlegend.axis("off")

    xs = torch.arange(-5, 6, 1).cpu()
    node_colors = [POSITIVE if c > 0 else NEGATIVE for c in node_legend_values]
    node_sizes = node_legend_values.abs().pow(1.).cpu().numpy() * node_scale
    axlegend.scatter(xs, [0 for _ in xs], c=node_colors, s=node_sizes)
    for idx, value in enumerate(node_legend_values):
        axlegend.text(xs[idx].item(), -0.5, f"{value.item():.1f}", ha="center", va="center")

    edge_colors = [POSITIVE if c > 0 else NEGATIVE for c in edge_legend_values]
    edge_width = edge_legend_values.abs().pow(1.).cpu().numpy() * edge_scale
    edge_alpha = edge_legend_values.abs().pow(1.).cpu().numpy().clip(0., 0.2) * edge_alpha_scale
    for x, color, width, alpha, value in zip(xs, edge_colors, edge_width, edge_alpha, edge_legend_values):
        axlegend.plot([x - 0.2, x + 0.2], [1., 1.], c=color, lw=width, alpha=alpha)
        axlegend.text(x.item(), 0.5, f"{value.item():.2f}", ha="center", va="center")

    axlegend.text(-6., 0., node_label, ha="right", va="center")
    axlegend.text(-6., 1., edge_label, ha="right", va="center")
    axlegend.spines['top'].set_visible(False)
    axlegend.spines['right'].set_visible(False)
    axlegend.spines['left'].set_visible(False)
    axlegend.spines['bottom'].set_visible(False)
    axlegend.grid(False)
    axlegend.set_xticks([])
    axlegend.set_yticks([])
    axlegend.set_xlim(-10, 10)
    axlegend.set_ylim(-1, 1.5)

    plt.tight_layout()
    plt.savefig(dir_figures + file + ".pdf")
    plt.close(fig)


# =============================================================================
# PLOT MEAN AND CORRELATION DIFF
diff_node_scale = 25.
diff_edge_scale = 50.
plot_mean_and_correlation_over_time(
     file=f"K{subject}_mean_and_cov_over_time",
     node_values=channel_wise_mean_diff,
     edge_values=channel_wise_cor_diff,
     node_label="Difference in mean",
     edge_label="Difference in correlation",
     node_legend_values=torch.Tensor([-10., -5., -2., -1., -0.5, 0., 0.5, 1., 2., 5., 10.]).cpu(),
     edge_legend_values=torch.Tensor([-0.2, -0.1, -0.05, -0.02, -0.01, 0., 0.01, 0.02, 0.05, 0.1, 0.2]).cpu(),
    node_scale=diff_node_scale,
    edge_scale=diff_edge_scale,
)


# =============================================================================
# PLOT NONTARGET DIFFERENCE TO PREVIOUS TIMEPOINT
nontarget_prev_diff_node_scale = 25.
nontarget_prev_diff_edge_scale = 50.
plot_mean_and_correlation_over_time(
     file=f"K{subject}_nontarget_prev_diff_mean_and_cov_over_time",
     node_values=nontarget_mean_prev_diff,
     edge_values=nontarget_cor_prev_diff,
     node_label="Lag-1 nontarget mean diff",
     edge_label="Lag-1 nontarget corr. diff",
     node_legend_values=torch.Tensor([-10., -5., -2., -1., -0.5, 0., 0.5, 1., 2., 5., 10.]).cpu(),
     edge_legend_values=torch.Tensor([-0.2, -0.1, -0.05, -0.02, -0.01, 0., 0.01, 0.02, 0.05, 0.1, 0.2]).cpu(),
    node_scale=nontarget_prev_diff_node_scale,
    edge_scale=nontarget_prev_diff_edge_scale,
    difference_titles=True,
 )

# =============================================================================
# PLOT TARGET DIFFERENCE TO PREVIOUS TIMEPOINT
target_prev_diff_node_scale = 25.
target_prev_diff_edge_scale = 50.
plot_mean_and_correlation_over_time(
     file=f"K{subject}_target_prev_diff_mean_and_cov_over_time",
     node_values=target_mean_prev_diff,
     edge_values=target_cor_prev_diff,
     node_label="Lag-1 target mean diff",
     edge_label="Lag-1 target corr. diff",
     node_legend_values=torch.Tensor([-10., -5., -2., -1., -0.5, 0., 0.5, 1., 2., 5., 10.]).cpu(),
     edge_legend_values=torch.Tensor([-0.2, -0.1, -0.05, -0.02, -0.01, 0., 0.01, 0.02, 0.05, 0.1, 0.2]).cpu(),
    node_scale=target_prev_diff_node_scale,
    edge_scale=target_prev_diff_edge_scale,
    difference_titles=True,
 )

