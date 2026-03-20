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
beta_xi1 = results.chains["smgp_scaling.target_signal"]
beta_xi0 = results.chains["smgp_scaling.nontarget_process"]
L = results.chains["loadings.norm_one"]  # nc x ns x E x K
Lnorm = results.chains["loadings.norm"] # nc x ns x K

# Weight scaling process by squared loading norm for each component.
loading_norm = Lnorm[:, :, 0, :].unsqueeze(-1)
scaling0 = beta_xi0.exp()
scaling1 = beta_xi1.exp()
scaling_diff = beta_xi1 - beta_xi0

# scaling yrange
scaling_ymin = min(scaling0.min(), scaling1.min()).item()
scaling_ymax = max(scaling0.max(), scaling1.max()).item()

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
# PLOT COMPONENTS (SCALING ONLY)
L = results.chains["loadings.norm_one"]
file = f"K{subject}_components_scaling"
fig, axes = plt.subplots(
    nrows=4+1,
    ncols=4,
    figsize=(10, 10),
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
    component_samples = L[0, :, :, k]

    rank_one_samples = torch.einsum("be,bf->bef", component_samples, component_samples)
    network = rank_one_samples.mean(0)
    network_sd = rank_one_samples.std(0)
    exclude = torch.zeros_like(network, dtype=torch.bool)
    valid = network_sd.gt(0)
    exclude[valid] = network[valid].abs() / network_sd[valid] < 1.96
    network = network.clone()
    network[~torch.isfinite(network)] = 0
    network[exclude] = 0

    edgelist = torch.tril_indices(16, 16, offset=-1)
    weights = network[edgelist[0], edgelist[1]]
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    edge_scale = 30.
    edge_alpha_scale = 3.
    alphas = weights.abs().cpu().numpy().clip(0.0, 0.2) * edge_alpha_scale
    G = nx.from_edgelist(edgelist.T.cpu().numpy(), create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, {idx: channels[idx] for idx in range(16)})
    nx.draw_networkx_nodes(
        G,
        pos=channel_positions,
        node_size=18,
        node_color="black",
        linewidths=0,
        ax=ax
    )
    nx.draw_networkx_edges(
        G,
        pos=channel_positions,
        width=weights.abs().cpu().numpy() * edge_scale,
        edge_color=[POSITIVE if w > 0 else NEGATIVE for w in weights],
        connectionstyle='arc3, rad=0.1',
        arrowsize=20,
        arrowstyle='-',
        alpha=alphas,
        ax=ax
    )
    blank_canvas(ax)

    # scaling process plot
    row += 1
    ax = axes[row, col]
    ax.set_title(f"Component {k+1}\n"
                 f"Loading norm: {Lnormk:.2f}\n"
                 f"BCE Change: {importance['drop_bce'][k]:.2f}")
    t = np.arange(25) * 31.25
    nontarget_scaling, nontarget_scaling_min, nontarget_scaling_max = confidance_band(scaling0[:, :, k, :])
    target_scaling, target_scaling_min, target_scaling_max = confidance_band(scaling1[:, :, k, :])

    # highlight where differential for scaling
    _, scaling_diff_min, scaling_diff_max = confidance_band(scaling_diff[:, :, k, :], 0.95)
    scaling_differential = scaling_diff_min.gt(0.) | scaling_diff_max.lt(0.)
    for x, d in zip(t, scaling_differential):
        if d:
            ax.axvline(x, color="black", linestyle="-", alpha=0.1, linewidth=6.25)

    ax.axhline(0, color="k", linestyle="--")
    ax.fill_between(
        t,
        nontarget_scaling_min.cpu().numpy(),
        nontarget_scaling_max.cpu().numpy(),
        alpha=0.5,
        color=NONTARGET
    )
    ax.plot(t, nontarget_scaling.cpu().numpy(), color=NONTARGET)
    ax.fill_between(
        t,
        target_scaling_min.cpu().numpy(),
        target_scaling_max.cpu().numpy(),
        alpha=0.5,
        color=TARGET
    )
    ax.plot(t, target_scaling.cpu().numpy(), color=TARGET)
    ax.set_xlim(0, 24*31.25)
    ax.set_ylim(scaling_ymin, scaling_ymax)
    ax.set_xticks([0, 150, 300, 450, 600, 750])
    ax.set_xlabel("Time (ms)")

# ADD LEGEND
axes[1, 0].set_ylabel("Scaling process")
axes[3, 0].set_ylabel("Scaling process")
gs = axes[4, 0].get_gridspec()
for ax in axes[4, :]:
    ax.remove()
axlegend = fig.add_subplot(gs[4, :], frameon=False)
axlegend.axis("off")

# rank-one term legend
edge_scale = 30.
edge_alpha_scale = 3.
values = torch.Tensor([-0.20, -0.10, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20]).cpu()
xs = torch.arange(-5, 6, 1).cpu()
edge_colors = [POSITIVE if c > 0 else NEGATIVE for c in values]
edge_widths = values.abs().cpu().numpy() * edge_scale
edge_alphas = values.abs().cpu().numpy().clip(0.0, 0.2) * edge_alpha_scale
for x, color, width, alpha, value in zip(xs, edge_colors, edge_widths, edge_alphas, values):
    axlegend.plot([x - 0.2, x + 0.2], [0.0, 0.0], c=color, lw=width, alpha=alpha)
for i, v in enumerate(values):
    axlegend.text(xs[i].item(), -0.8, f"{v.item():.2f}", ha="center", va="center")
axlegend.text(-6., 0., "Rank-one term", ha="right", va="center")

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

plt.tight_layout(h_pad=0.0)
plt.savefig(dir_figures + file + ".pdf")

