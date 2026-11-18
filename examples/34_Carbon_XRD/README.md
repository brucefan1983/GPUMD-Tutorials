# Powder X-ray diffraction of carbon structures

Use GPUMD to calculate a powder XRD pattern for a 128-atom diamond cell. Compare the pattern with LAMMPS, then run the same workflow for AB graphite and three amorphous carbon snapshots.

## Requirements

- An NVIDIA GPU and a GPUMD build that supports `compute_xrd`. While [GPUMD PR #1910](https://github.com/brucefan1983/GPUMD/pull/1910) is open, clone the [feature branch](https://github.com/LYKD/GPUMD/tree/codex/native-powder-xrd), check out commit `ad258ee083adc6e6b262afd0b66c0c45517a487e`, and follow the [GPUMD installation guide](https://gpumd.org/dev/installation.html).
- LAMMPS with the `DIFFRACTION` package. This example uses [22 Jul 2025 Update 4](https://github.com/lammps/lammps/releases/tag/stable_22Jul2025_update4). If building LAMMPS, enable the package with `-D PKG_DIFFRACTION=on`.
- Python 3. Install Matplotlib if you want to plot the results.

## Run diamond first

Open a terminal in this example directory. Set the paths to your executables, then run:

```bash
export GPUMD=/path/to/gpumd
export LAMMPS=/path/to/lmp
test -x "$GPUMD"
"$LAMMPS" -h | grep DIFFRACTION
python3 run_xrd.py --gpumd "$GPUMD" --lammps "$LAMMPS" \
  --case diamond --output diamond.txt
```

A successful run reports **150 bins** and a maximum difference relative to the reference peak of at most **5 × 10⁻⁶**. The strongest diamond bin is near **44°** in 2θ and **2.06 Å** in d-spacing. Use `--case graphite --output graphite.txt` to run graphite. Its strongest peak is near **26.5°** and **3.36 Å**.

The shared [GPUMD input](run.in) uses a zero time step to sample the supplied structure and writes individual reciprocal points to `xrd.txt`. Those rows contain three reciprocal-point indices, 2θ in degrees and intensity. The driver reads that output from a temporary run directory and writes the binned comparison to the filename you supply with `--output`. On failure it keeps the failing run directory for inspection.

## Run all five structures

| Case | Atoms | Input |
| --- | ---: | --- |
| Diamond | 128 | [diamond.xyz](structures/diamond.xyz) |
| AB graphite | 6,144 | [graphite.xyz](structures/graphite.xyz) |
| Amorphous carbon, 1.11 g/cm³ | 55,296 | [rho-1.11.xyz](structures/rho-1.11.xyz) |
| Amorphous carbon, 2.83 g/cm³ | 55,296 | [rho-2.83.xyz](structures/rho-2.83.xyz) |
| Amorphous carbon, 3.35 g/cm³ | 55,296 | [rho-3.35.xyz](structures/rho-3.35.xyz) |

Omit `--case` to run all five. The amorphous cases use a dense mesh and take longer. Allow about 400 MB of temporary disk space per case. The optional MPI flags below use MPICH. Match the rank count to your allocated CPU cores. Omit `--mpiexec` and `--ranks` for a single-process LAMMPS run.

```bash
export MPIEXEC=/path/to/mpich/bin/mpiexec
python3 run_xrd.py --gpumd "$GPUMD" --lammps "$LAMMPS" \
  --mpiexec "$MPIEXEC" --ranks 24 --output all-five.txt
```

The driver converts the same extended XYZ coordinates into inputs for both programs. It uses a wavelength of **1.541838 Å**, Lorentz–polarization correction and a **10–150°** calculation window. The crystal mesh spacing is the inverse cell length along each axis. The amorphous snapshots use **0.01 Å⁻¹** along each axis. For each structure the driver sums intensities from **10–100°** in 150 bins of **0.6°** and normalizes them by the total intensity within that range.

The three amorphous configurations are final frames after 100 ps at 300 K and fixed volume. The snapshots and three reference spectra come from the authors' archive for Fig. S6. [Li et al., ACS Nano 20, 11284–11293 (2026)](https://doi.org/10.1021/acsnano.6c00656)

## Read and plot the results

The full output has 750 data rows. The tab-separated columns `two_theta_deg` and `d_angstrom` contain the bin center and its corresponding d-spacing. The intensity columns record the published spectrum where available, the independent reference calculation, and the GPUMD and LAMMPS results. A blank field indicates unavailable data. The archived [reference table](results/xrd-reference.txt) includes both native results for the two crystals and the archived and independently recomputed spectra for the three amorphous snapshots.

```bash
python3 -m pip install matplotlib
python3 plot_xrd.py all-five.txt all-five.pdf
```

The first page uses d-spacing and the second uses 2θ. To view the expected curves before running either program, open the [reference PDF](results/xrd-reference.pdf). The independently computed amorphous spectra agree with the Fig. S6 reference table in all 450 bins at six significant figures. In the 1–5 Å window, the highest bins for 2.83 and 3.35 g/cm³ are at 2.969 and 2.176 Å.

## Reference timings

These per-frame XRD timings used one RTX 4090 for GPUMD and eight single-threaded MPI ranks on an AMD EPYC 7543 for LAMMPS. Both programs used a wavelength of 1.5406 Å and a 10–100° window and sent formatted output to `/dev/null`. The amorphous timing runs used pre-equilibration frames and a coarser mesh than the Fig. S6 calculation above.

| Structure | Reciprocal points per frame | GPUMD GPU (s) | LAMMPS CPU (s) |
| --- | ---: | ---: | ---: |
| Diamond | 3,010 | 0.00287 | 0.00588 |
| AB graphite | 222,882 | 0.282 | 6.13 |
| Amorphous carbon, 1.114 g/cm³ | 150,856 | 0.539 | 41.14 |
| Amorphous carbon, 2.832 g/cm³ | 59,508 | 0.209 | 16.21 |
| Amorphous carbon, 3.350 g/cm³ | 50,100 | 0.176 | 13.35 |
