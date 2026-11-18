#!/usr/bin/env python3
"""Calculate XRD for the three Fig. S6 snapshots and two crystal benchmarks."""

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
INPUTS = ("run.in", "in.xrd", "zero.lj")
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
    with (work / "atoms.data").open("w", encoding="ascii") as lammps:
        lammps.write(f"{source.stem}\n\n{count} atoms\n1 atom types\n")
        for axis, length in zip("xyz", lengths):
            lammps.write(f"0 {length:.17g} {axis}lo {axis}hi\n")
        lammps.write("\nMasses\n\n1 12.011\n\nAtoms # atomic\n\n")
        for index, atom in enumerate(positions, 1):
            lammps.write(f"{index} 1 {' '.join(atom[1:])}\n")
    return lengths


def prepare_case(case, work):
    lengths = prepare_structure(HERE / "structures" / f"{case}.xyz", work)
    spacing = ((0.01,) * 3 if case.startswith("rho-")
               else tuple(1 / length for length in lengths))
    mesh = " ".join(f"{value:.17g}" for value in spacing)
    for name in INPUTS:
        if name == "zero.lj":
            shutil.copyfile(HERE / name, work / name)
        else:
            template = (HERE / name).read_text(encoding="ascii")
            (work / name).write_text(template.replace("@SPACING@", mesh), encoding="ascii")


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
            angle = float(angle_text)
            if 10 <= angle <= 100:
                index = min(int((angle - 10) * NUMBER_OF_BINS / 90), NUMBER_OF_BINS - 1)
                counts[index] += float(intensity)
    if rows != declared_nodes or (expected_nodes is not None and rows != expected_nodes):
        raise ValueError(f"Unexpected number of GPUMD rows: {rows:,}")
    total = math.fsum(counts)
    if total <= 0 or any(not math.isfinite(value) or value < 0 for value in counts):
        raise ValueError("Invalid GPUMD intensity bins")
    return [value / total for value in counts]


def lammps_histogram(path):
    lines = path.read_text(encoding="ascii").splitlines()
    rows = [line.split() for line in lines[4:]]
    if len(rows) != NUMBER_OF_BINS:
        raise ValueError(f"Expected {NUMBER_OF_BINS} LAMMPS bins, found {len(rows)}")
    values = [(float(row[1]), float(row[3])) for row in rows]
    if any(not math.isfinite(value) or value < 0 for _, value in values):
        raise ValueError("Invalid LAMMPS intensity bins")
    return values


def write_table(stream, rows):
    table = csv.writer(stream, delimiter="\t")
    table.writerow(COLUMNS)
    table.writerows(rows)


def reproduce(args, cases, reference):
    gpumd = str(args.gpumd.resolve(strict=True))
    lammps = [str(args.lammps.resolve(strict=True)), "-log", "none", "-in", "in.xrd"]
    if args.ranks > 1:
        if args.mpiexec is None:
            raise SystemExit("Supply --mpiexec when using more than one MPI rank.")
        launcher = str(args.mpiexec.resolve(strict=True))
        lammps = [launcher, "-launcher", "fork", "-bind-to", "core", "-n", str(args.ranks), *lammps]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    results = []
    for case in cases:
        density = case[4:] if case.startswith("rho-") else ""
        expected = reference[case]
        with tempfile.TemporaryDirectory(prefix=f"xrd-{case}-", dir=args.output.parent) as temp:
            work = Path(temp)
            try:
                prepare_case(case, work)
                print(f"Running {case}", flush=True)
                subprocess.run([gpumd], cwd=work, env=env, check=True)
                gpu_bins = gpumd_histogram(work / "xrd.txt", DENSE_GRID_NODES if density else None)
                subprocess.run(lammps, cwd=work, env=env, check=True)
                cpu_bins = lammps_histogram(work / "lammps-histogram.data")
                if any(float(row["two_theta_deg"]) != angle for row, (angle, _) in zip(expected, cpu_bins)):
                    raise ValueError(f"{case}: unexpected LAMMPS bin centers")
                ref_peak = max(float(row["paper_fraction"] or row["reference_fraction"]) for row in expected)
                differences = []
                current_rows = []
                for row, gpu, (_, cpu) in zip(expected, gpu_bins, cpu_bins):
                    ref = float(row["paper_fraction"] or row["reference_fraction"])
                    differences.append(max(abs(gpu - ref), abs(cpu - ref), abs(gpu - cpu)))
                    current_rows.append((case, density, row["two_theta_deg"], row["d_angstrom"],
                                         row["paper_fraction"], row["reference_fraction"], gpu, cpu))
                error = max(differences) / ref_peak
                print(f"{case}: {NUMBER_OF_BINS} bins, max difference / reference peak = {error:.3g}",
                      flush=True)
                if error > 5e-6 or not math.isfinite(error):
                    raise ValueError(f"{case}: the normalized spectra did not match")
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
    parser.add_argument("--lammps", type=Path, required=True)
    parser.add_argument("--mpiexec", type=Path)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--case", choices=CASES, help="Run one structure. Default: run all five.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.ranks < 1 or args.output.exists():
        parser.error("Use positive ranks and a new output file.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reference = {}
    with (HERE / "results/xrd-reference.txt").open(newline="", encoding="ascii") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            reference.setdefault(row["structure"], []).append(row)
    if set(reference) != set(CASES) or any(len(rows) != NUMBER_OF_BINS for rows in reference.values()):
        raise ValueError("Expected five reference curves with 150 bins each")
    cases = (args.case,) if args.case else CASES
    values = reproduce(args, cases, reference)
    with args.output.open("x", newline="", encoding="ascii") as stream:
        write_table(stream, values)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
