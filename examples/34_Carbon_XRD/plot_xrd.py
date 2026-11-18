#!/usr/bin/env python3
"""Plot the tutorial's tab-separated XRD table as a two-page PDF."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from run_xrd import CASES


TRACES = {
    "paper_fraction": ("Published Fig. S6", "#31343a", "-", 2.0),
    "reference_fraction": ("Independent calculation", "#7e37a5", (0, (4, 2)), 1.2),
    "gpumd_fraction": ("GPUMD", "#006fb0", (0, (4, 2)), 1.35),
    "lammps_fraction": ("LAMMPS", "#d04e20", (0, (1, 2)), 1.1),
}


def title(case):
    if case.startswith("rho-"):
        return f"{case[4:]} g/cm³ amorphous carbon"
    return {"diamond": "Cubic diamond", "graphite": "AB graphite"}[case]


def draw(data, x_field, x_label, limits):
    ordered = [case for case in CASES[2:] + CASES[:2] if case in data]
    ncols = min(len(ordered), 3)
    nrows = (len(ordered) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.05 * ncols, 2.45 * nrows + 0.35),
                             squeeze=False)
    handles = {}
    for case, ax in zip(ordered, axes.flat):
        rows = data[case]
        x = [float(row[x_field]) for row in rows]
        for key, (label, color, linestyle, linewidth) in TRACES.items():
            if all(row[key] != "" for row in rows):
                y = [100 * float(row[key]) for row in rows]
                line, = ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth,
                                label=label)
                handles[label] = line
        ax.set(xlim=limits, xlabel=x_label, title=title(case))
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e3e6e9", linewidth=0.6)
        ax.set_axisbelow(True)
    for ax in axes.flat[len(ordered):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of total intensity (%)")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.legend(handles.values(), handles.keys(), loc="upper center",
               ncol=len(handles) if ncols > 1 else min(len(handles), 2), frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Tab-separated XRD .txt file")
    parser.add_argument("output", type=Path, help="Output .pdf path")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".pdf":
        parser.error("Use a .pdf output file.")
    data = {}
    with args.input.open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            data.setdefault(row["structure"], []).append(row)
    if not data or not set(data).issubset(CASES):
        raise ValueError("Expected tutorial XRD case names in the input text")
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.output) as pdf:
        for field, label, limits in (
            ("d_angstrom", "d (Å)", (5, 1)),
            ("two_theta_deg", "2θ (°)", (10, 100)),
        ):
            fig = draw(data, field, label, limits)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
