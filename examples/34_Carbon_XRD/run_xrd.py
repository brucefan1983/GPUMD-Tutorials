#!/usr/bin/env python3
"""Run GPUMD XRD for the five supplied carbon structures."""

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
WAVELENGTH = 1.541838
NUMBER_OF_BINS = 150
DENSE_GRID_NODES = 8232264
CASES = ("diamond", "graphite", "rho-1.11", "rho-2.83", "rho-3.35")
COLUMNS = ("structure", "density_g_cm3", "two_theta_deg", "d_angstrom",
           "paper_fraction", "reference_fraction", "gpumd_fraction", "lammps_fraction")


def prepare_structure(source, work):
    lines = source.read_text(encoding="ascii").splitlines()
    count = int(lines[0])
    cell = re.search(r'Lattice="([^"]+)"', lines[1]).group(1)
    matrix = list(map(float, cell.split()))
    if len(matrix) != 9 or any(matrix[i] != 0 for i in (1, 2, 3, 5, 6, 7)):
        raise ValueError(f"{source.name}: expected an axis-aligned cell")
    lengths = matrix[::4]
    positions = [line.split()[:4] for line in lines[2:]]
    if len(positions) != count or any(len(row) != 4 or row[0] != "C" for row in positions):
        raise ValueError(f"{source.name}: expected {count} carbon coordinates")
    with (work / "model.xyz").open("w", encoding="ascii") as gpumd:
        gpumd.write(f'{count}\nLattice="{cell}" pbc="T T T" Properties=species:S:1:pos:R:3:vel:R:3\n')
        for atom in positions:
            gpumd.write(" ".join(atom) + " 0 0 0\n")
    return lengths


def prepare_case(case, work, form_factor):
    lengths = prepare_structure(HERE / "structures" / f"{case}.xyz", work)
    spacing = ((0.01,) * 3 if case.startswith("rho-")
               else tuple(1 / length for length in lengths))
    mesh = " ".join(f"{value:.17g}" for value in spacing)
    template = (HERE / "run.in").read_text(encoding="ascii")
    factor_option = ""
    if form_factor is not None:
        shutil.copyfile(form_factor, work / "xrd_factors.txt")
        factor_option = "form_factor xrd_factors.txt "
    command = template.replace("@SPACING@", mesh).replace("@FORM_FACTOR@", factor_option)
    (work / "run.in").write_text(command, encoding="ascii")
    shutil.copyfile(HERE / "zero.lj", work / "zero.lj")


def gpumd_histogram(path, expected_nodes=None):
    counts = [0.0] * NUMBER_OF_BINS
    rows = 0
    with path.open(encoding="ascii") as stream:
        header = stream.readline().split()
        if header[:2] != ["#", "step"] or header[3:4] != ["nodes"]:
            raise ValueError("Missing GPUMD XRD block header")
        declared_nodes = int(header[4])
        for line in stream:
            if line.startswith("#"):
                continue
            rows += 1
            _, _, _, angle_text, intensity = line.split()
            angle, value = float(angle_text), float(intensity)
            if not math.isfinite(angle) or not math.isfinite(value) or value < 0:
                raise ValueError("Invalid GPUMD reciprocal-point data")
            if 10 <= angle <= 100:
                index = min(int((angle - 10) * NUMBER_OF_BINS / 90), NUMBER_OF_BINS - 1)
                counts[index] += value
    if rows != declared_nodes or (expected_nodes is not None and rows != expected_nodes):
        raise ValueError(f"Unexpected number of GPUMD rows: {rows:,}")
    total = math.fsum(counts)
    if total <= 0 or any(not math.isfinite(value) or value < 0 for value in counts):
        raise ValueError("Invalid GPUMD intensity bins")
    return [value / total for value in counts]


def write_table(stream, rows):
    table = csv.writer(stream, delimiter="\t")
    table.writerow(COLUMNS)
    table.writerows(rows)


def calculate(args, cases):
    gpumd = str(args.gpumd.resolve(strict=True))
    env = dict(os.environ, OMP_NUM_THREADS="1")
    results = []
    for case in cases:
        density = case[4:] if case.startswith("rho-") else ""
        with tempfile.TemporaryDirectory(prefix=f"xrd-{case}-", dir=args.output.parent) as temp:
            work = Path(temp)
            try:
                prepare_case(case, work, args.form_factor)
                print(f"Running {case}", flush=True)
                subprocess.run([gpumd], cwd=work, env=env, check=True)
                gpu_bins = gpumd_histogram(work / "xrd.txt", DENSE_GRID_NODES if density else None)
                current_rows = []
                for index, intensity in enumerate(gpu_bins):
                    two_theta = 10 + (index + 0.5) * 90 / NUMBER_OF_BINS
                    spacing = WAVELENGTH / (2 * math.sin(math.radians(two_theta / 2)))
                    current_rows.append((case, density, two_theta, spacing, "", "", intensity, ""))
                peak = max(current_rows, key=lambda row: row[6])
                print(f"{case}: {NUMBER_OF_BINS} bins, strongest bin at "
                      f"{peak[2]:.1f} degrees and {peak[3]:.3f} angstrom", flush=True)
            except Exception:
                retained = work.rename(work.with_name(f"{work.name}-failed"))
                print(f"Kept failed inputs and outputs in {retained.resolve()}", file=sys.stderr)
                if results:
                    with tempfile.NamedTemporaryFile(
                            mode="w", encoding="ascii", newline="", delete=False,
                            dir=args.output.parent, prefix=f"{args.output.stem}-",
                            suffix=".partial.txt") as partial:
                        write_table(partial, results)
                        print(f"Saved completed cases to {partial.name}", file=sys.stderr)
                raise
            results.extend(current_rows)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpumd", type=Path, required=True)
    parser.add_argument(
        "--form-factor", type=Path, metavar="FILE",
        help="Custom four-Gaussian coefficient file. Default: GPUMD's built-in neutral-atom fit.")
    parser.add_argument("--case", choices=CASES, help="Run one structure. Default: run all five.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.form_factor is not None and not args.form_factor.is_file():
        parser.error(f"Scattering coefficient file does not exist: {args.form_factor}")
    if args.output.exists():
        parser.error("Use a new output file.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = (args.case,) if args.case else CASES
    values = calculate(args, cases)
    with args.output.open("x", newline="", encoding="ascii") as stream:
        write_table(stream, values)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
