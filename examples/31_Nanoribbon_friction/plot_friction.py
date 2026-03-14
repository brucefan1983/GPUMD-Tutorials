#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def set_academic_style(fontsize=16):
    """Set a clean publication-style Matplotlib theme."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": fontsize,
        "axes.labelsize": fontsize + 1,
        "axes.titlesize": fontsize + 1,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "axes.linewidth": 1.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0,
        "ytick.minor.width": 1.0,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "legend.frameon": False,
        "axes.grid": False,
    })



def moving_average(y, win):
    y = np.asarray(y, dtype=float)
    if win is None or win <= 1:
        return y
    win = int(win)
    if win % 2 == 0:
        win += 1
    if win >= len(y):
        win = max(1, len(y) // 2 * 2 - 1)
    if win <= 1:
        return y
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(y, kernel, mode="same")



def load_spring_force(path: str):
    """
    Auto-detect two supported formats:

    New format:
        step mode Fx Fy Fz Ftotal energy

    Old format:
        step call mode Fx Fy Fz energy fric
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            rows.append(parts)

    if not rows:
        raise ValueError(f"No valid data found in {path}")

    ncol = len(rows[0])
    if ncol == 7:
        step, mode, Fx, Fy, Fz, Ftotal, energy = zip(*rows)
        data = {
            "format": "new",
            "step": np.asarray(step, dtype=int),
            "mode": np.asarray(mode, dtype=object),
            "Fx": np.asarray(Fx, dtype=float),
            "Fy": np.asarray(Fy, dtype=float),
            "Fz": np.asarray(Fz, dtype=float),
            "Ftotal": np.asarray(Ftotal, dtype=float),
            "energy": np.asarray(energy, dtype=float),
        }
    elif ncol >= 8:
        step, call, mode, Fx, Fy, Fz, energy, fric = zip(*[r[:8] for r in rows])
        Fx = np.asarray(Fx, dtype=float)
        Fy = np.asarray(Fy, dtype=float)
        Fz = np.asarray(Fz, dtype=float)
        data = {
            "format": "old",
            "step": np.asarray(step, dtype=int),
            "call": np.asarray(call, dtype=int),
            "mode": np.asarray(mode, dtype=object),
            "Fx": Fx,
            "Fy": Fy,
            "Fz": Fz,
            "Ftotal": np.sqrt(Fx**2 + Fy**2 + Fz**2),
            "energy": np.asarray(energy, dtype=float),
            "fric": np.asarray(fric, dtype=float),
        }
    else:
        raise ValueError(
            f"Unsupported column count ({ncol}) in {path}. "
            "Expected 7 columns (new format) or >=8 columns (old format)."
        )

    return data



def select_dataset(data, call_id=None):
    """Return a plotting subset. For old format, optionally select one call."""
    if data["format"] == "new":
        return data, None

    calls = np.unique(data["call"])
    if call_id is None:
        call_id = int(calls[0])
    mask = (data["call"] == call_id)
    if not np.any(mask):
        raise ValueError(f"call={call_id} not found. Available calls: {calls.tolist()}")

    subset = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)
              for k, v in data.items()}
    return subset, call_id



def plot_force(data, smooth=1, show_total=True, outname=None):
    step = data["step"]
    Fx = moving_average(data["Fx"], smooth)
    Fy = moving_average(data["Fy"], smooth)
    Fz = moving_average(data["Fz"], smooth)
    Ftotal = moving_average(data["Ftotal"], smooth)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.plot(step, Fx, lw=2.0, label=r"$F_x$")
    ax.plot(step, Fy, lw=2.0, label=r"$F_y$")
    ax.plot(step, Fz, lw=2.0, label=r"$F_z$")
    if show_total:
        ax.plot(step, Ftotal, lw=2.4, ls="--", label=r"$F_{\mathrm{total}}$")

    ax.set_xlabel("Step")
    ax.set_ylabel(r"Spring force (eV/$\mathrm{\AA}$)")
    ax.minorticks_on()
    ax.tick_params(top=True, right=True)
    ax.legend(loc="best", handlelength=2.0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    fig.tight_layout()

    if outname:
        outpath = Path(outname)
        fig.savefig(outpath)
        print(f"[saved] {outpath.resolve()}")

    return fig, ax



def main():
    parser = argparse.ArgumentParser(
        description="Plot spring-force curves with a publication-style figure."
    )
    parser.add_argument(
        "-i", "--input", default="spring_force_0.out",
        help="Input file path (default: spring_force_0.out)"
    )
    parser.add_argument(
        "--call", type=int, default=None,
        help="For old-format files only: choose one call to plot"
    )
    parser.add_argument(
        "--smooth", type=int, default=1,
        help="Moving-average window size; odd integer recommended (default: 1)"
    )
    parser.add_argument(
        "--no-total", action="store_true",
        help="Do not plot the total force"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save the figure instead of only showing it"
    )
    parser.add_argument(
        "-o", "--output", default="spring_force.pdf",
        help="Output filename when --save is enabled (default: spring_force.pdf)"
    )
    parser.add_argument(
        "--fontsize", type=int, default=16,
        help="Base font size for the figure (default: 16)"
    )
    args = parser.parse_args()

    set_academic_style(args.fontsize)

    data = load_spring_force(args.input)
    subset, used_call = select_dataset(data, args.call)

    if data["format"] == "new":
        print(f"[info] Detected new format: step mode Fx Fy Fz Ftotal energy")
    else:
        print(f"[info] Detected old format: step call mode Fx Fy Fz energy fric")
        print(f"[info] Plotting call = {used_call}")

    plot_force(
        subset,
        smooth=args.smooth,
        show_total=not args.no_total,
        outname=args.output if args.save else None,
    )
    # plt.show()


if __name__ == "__main__":
    main()
