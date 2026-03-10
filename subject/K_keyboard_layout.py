import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(1, str(ROOT))


# Keep this layout aligned with source/data/k_protocol.py.
KEYBOARD = np.array(
    [
        ["A", "B", "C", "D", "E", "F"],
        ["G", "H", "I", "J", "K", "L"],
        ["M", "N", "O", "P", "Q", "R"],
        ["S", "T", "U", "V", "W", "X"],
        ["Y", "Z", "1", "2", "3", "4"],
        ["5", "SPK", ".", "BS", "!", "_"],
    ]
)

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_keyboard_panel(
    ax,
    active_col: int | None = None,
    active_row: int | None = None,
    red_col_label: int | None = None,
) -> None:
    n_rows, n_cols = KEYBOARD.shape

    screen = Rectangle(
        (0, 0),
        n_cols,
        n_rows,
        facecolor="black",
        edgecolor="0.55",
        linewidth=5,
    )
    ax.add_patch(screen)

    for r in range(n_rows):
        for c in range(n_cols):
            y = n_rows - 1 - r
            is_active = (active_col is not None and c == active_col) or (
                active_row is not None and r == active_row
            )
            key_color = "white" if is_active else "0.55"
            ax.text(
                c + 0.5,
                y + 0.5,
                KEYBOARD[r, c],
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color=key_color,
            )

    for r in range(n_rows):
        y = n_rows - 1 - r + 0.5
        row_color = "black" if active_row is not None and r == active_row else "0.4"
        ax.text(-0.15, y, f"R{r + 1}", ha="right", va="center", fontsize=11, color=row_color)

    for c in range(n_cols):
        x = c + 0.5
        if red_col_label is not None and c == red_col_label:
            col_color = "red"
        elif active_col is not None and c == active_col:
            col_color = "black"
        else:
            col_color = "0.4"
        ax.text(x, n_rows + 0.12, f"C{c + 1}", ha="center", va="bottom", fontsize=11, color=col_color)

    # Mark T as target in every panel.
    target_rc = np.argwhere(KEYBOARD == "T")[0]
    tr, tc = int(target_rc[0]), int(target_rc[1])
    ty = n_rows - 1 - tr + 0.5
    tx = tc + 0.5
    ax.add_patch(Circle((tx, ty), radius=0.43, fill=False, edgecolor="red", linewidth=2.4))

    ax.set_xlim(-0.85, n_cols + 0.05)
    ax.set_ylim(-0.05, n_rows + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def main(filename: str = "K_keyboard_layout.pdf") -> None:
    fig, axes = plt.subplots(2, 1, figsize=(4,8))

    # Top panel: C2 highlighted, C2 label in red.
    draw_keyboard_panel(axes[0], active_col=1, active_row=None, red_col_label=1)
    axes[0].set_title("Target stimulus", fontsize=14, pad=8)

    # Bottom panel: R3 highlighted.
    draw_keyboard_panel(axes[1], active_col=None, active_row=2, red_col_label=None)
    axes[1].set_title("Nontarget stimulus", fontsize=14, pad=8)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.03, hspace=0.12)

    out_file = OUT_DIR / filename
    fig.savefig(out_file)
    plt.close(fig)

    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    main()

