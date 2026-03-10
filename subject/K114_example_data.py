import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(ROOT))

from source.data.k_protocol import KProtocol


# Data selection
TYPE = "TRN"
SUBJECT = "114"
SESSION = "001"
NAME = f"K{SUBJECT}_{SESSION}_BCI_{TYPE}"
MAT_FILE = ROOT / "K_Protocol" / f"{NAME}.mat"
REPETITION_INDEX = 2
ELECTRODE_COUNT = 3

# Keep preprocessing aligned with existing subject scripts.
WINDOW = 800.0
BANDPASS_WINDOW = (0.1, 15.0)
BANDPASS_ORDER = 2
DOWNSAMPLE = 8

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _source_label(source_idx_zero_based: int) -> str:
    source = source_idx_zero_based + 1
    if source <= 6:
        return f"R{source}"
    return f"C{source - 6}"


def main(repetition_index: int = 1, n_electrodes: int = 5) -> None:
    eeg = KProtocol(
        filename=str(MAT_FILE),
        type=TYPE,
        subject=SUBJECT,
        session=SESSION,
        window=WINDOW,
        bandpass_window=BANDPASS_WINDOW,
        bandpass_order=BANDPASS_ORDER,
        downsample=DOWNSAMPLE,
    )

    available_reps = sorted(eeg.stimulus_data["repetition"].unique().tolist())
    if repetition_index not in available_reps:
        raise ValueError(
            f"Invalid repetition_index={repetition_index}. "
            f"Available repetitions: {available_reps}"
        )

    max_electrodes = len(eeg.channel_names)
    if not (1 <= n_electrodes <= max_electrodes):
        raise ValueError(
            f"Invalid n_electrodes={n_electrodes}. "
            f"Must be between 1 and {max_electrodes}."
        )

    # Restrict to the requested repetition and take the first sequence in it.
    eeg = eeg.repetitions([repetition_index])
    trace = eeg.sequence[0].cpu().numpy()[:n_electrodes, :]  # shape: (n_electrodes, time)

    n_channels, n_time = trace.shape
    sampling_rate = eeg.sampling_rate / eeg.downsample
    time_ms = np.arange(n_time) * 1000.0 / sampling_rate

    # Build the per-stimulus timing and labels in presentation order for this sequence.
    stimulus_order = eeg.stimulus_order[0].cpu().numpy().astype(int)
    is_target_by_source = eeg.target[0].cpu().numpy().astype(bool)
    source_by_time = np.argsort(stimulus_order)
    stim_times_ms = (
        np.arange(source_by_time.shape[0])
        * eeg.stimulus_to_stimulus_interval
        * 1000.0
        / sampling_rate
    )

    fig, axes = plt.subplots(
        n_channels,
        1,
        sharex=True,
        figsize=(8, 3),
        gridspec_kw={"hspace": 0.02},
    )

    for i, ax in enumerate(axes):
        ax.plot(time_ms, trace[i], color="black", linewidth=0.7)
        for t, source_idx in zip(stim_times_ms, source_by_time):
            is_target = bool(is_target_by_source[source_idx])
            ax.axvline(
                t,
                color="crimson" if is_target else "0.6",
                linewidth=1.0 if is_target else 0.5,
                linestyle="-" if is_target else "--",
                alpha=0.85 if is_target else 0.6,
                zorder=0,
            )
        ax.set_ylabel(eeg.channel_names[i], rotation=0, labelpad=18, fontsize=10)
        ax.tick_params(axis="y", labelsize=6, length=2)
        ax.tick_params(axis="x", labelsize=7, length=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    top_ax = axes[0]
    bottom_ax = axes[-1]
    for t, source_idx in zip(stim_times_ms, source_by_time):
        is_target = bool(is_target_by_source[source_idx])
        top_ax.text(
            t,
            1.02,
            _source_label(int(source_idx)),
            transform=top_ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if is_target else "normal",
            color="crimson" if is_target else "0.35",
            clip_on=False,
        )

    if len(stim_times_ms) > 1:
        y_annot = 0.7
        bottom_ax.annotate(
            "",
            xy=(stim_times_ms[1], y_annot),
            xytext=(stim_times_ms[0], y_annot),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "<->", "color": "navy", "lw": 1.6},
        )
        bottom_ax.text(
            0.5 * (stim_times_ms[0] + stim_times_ms[1]),
            y_annot + 0.1,
            r"$\Delta=156.25\,\mathrm{ms}$",
            transform=bottom_ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="navy",
            bbox={"facecolor": "white", "edgecolor": "navy", "alpha": 0.85, "pad": 1.5},
        )

    # Response window from the last stimulus onset.
    if len(stim_times_ms) > 0:
        last_onset_ms = float(stim_times_ms[-1])
        response_end_ms = last_onset_ms + WINDOW

        xmin, xmax = bottom_ax.get_xlim()
        if response_end_ms > xmax:
            bottom_ax.set_xlim(xmin, response_end_ms * 1.01)

        y_resp = 0.7
        bottom_ax.annotate(
            "",
            xy=(response_end_ms, y_resp),
            xytext=(last_onset_ms, y_resp),
            xycoords=("data", "axes fraction"),
            textcoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "<->", "color": "navy", "lw": 1.8},
        )
        bottom_ax.text(
            0.5 * (last_onset_ms + response_end_ms),
            y_resp + 0.1,
            r"$T_o=800\,\mathrm{ms}$",
            transform=bottom_ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="navy",
            bbox={"facecolor": "white", "edgecolor": "navy", "alpha": 0.9, "pad": 1.5},
        )

    axes[-1].set_xlabel("Time (ms)")
    # fig.suptitle(
    #     f"Subject 114, repetition {repetition_index}, first sequence",
    #     fontsize=10,
    #     y=0.995,
    # )
    fig.subplots_adjust(left=0.1, right=0.995, bottom=0.13, top=0.94, hspace=0.03)
    out_file = OUT_DIR / f"K114_repetition_{repetition_index:02d}_example.pdf"
    fig.savefig(out_file)
    plt.close(fig)

    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    main(repetition_index=REPETITION_INDEX, n_electrodes=ELECTRODE_COUNT)
