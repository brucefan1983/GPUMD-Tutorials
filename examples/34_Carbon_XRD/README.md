# Powder X-ray diffraction of carbon structures

Calculate XRD with GPUMD for a 128-atom diamond cell, then run the same workflow for AB graphite and three amorphous carbon snapshots.

## Run diamond first

Use an NVIDIA GPU and a GPUMD build whose `compute_xrd` accepts explicit `form_factor` coefficients. While [GPUMD PR #1910](https://github.com/brucefan1983/GPUMD/pull/1910) is open, use [this GPUMD commit](https://github.com/LYKD/GPUMD/tree/40b896c360f8aac18ebce638936470881e2149b4) and follow the [build instructions](https://gpumd.org/dev/installation.html). The driver uses Python 3. Plotting also needs Matplotlib.

Open a terminal in this example directory.

```bash
export GPUMD=/path/to/gpumd
python3 run_xrd.py --gpumd "$GPUMD" --case diamond --output diamond.txt
```

The default example assigns each atom a constant unit scattering amplitude. This is an analytic model for verifying the workflow. Diamond should show the allowed (111) peak near 44° (2.06 Å). Use `--case graphite --output graphite.txt` for graphite. A strong (002) reflection appears near 26.5° (3.36 Å).

To use a physical carbon scattering model, supply its nine coefficients to `--form-factor` in the order `a1 b1 a2 b2 a3 b3 a4 b4 c`. GPUMD evaluates `c + sum(a_i exp(-b_i s^2))` with `s = sin(theta)/wavelength`. Passing `0 0 0 0 0 0 0 0 1` explicitly reproduces the unit-scatterer example. For a comparison with published intensities, use the same scattering model and calculation settings as the reference.

## Run all five structures

| Case | Atoms | Input |
| --- | ---: | --- |
| Diamond | 128 | [diamond.xyz](structures/diamond.xyz) |
| AB graphite | 6,144 | [graphite.xyz](structures/graphite.xyz) |
| Amorphous carbon, 1.11 g/cm³ | 55,296 | [rho-1.11.xyz](structures/rho-1.11.xyz) |
| Amorphous carbon, 2.83 g/cm³ | 55,296 | [rho-2.83.xyz](structures/rho-2.83.xyz) |
| Amorphous carbon, 3.35 g/cm³ | 55,296 | [rho-3.35.xyz](structures/rho-3.35.xyz) |

```bash
python3 run_xrd.py --gpumd "$GPUMD" --output all-five.txt
python3 plot_xrd.py all-five.txt all-five.pdf
```

The shared [GPUMD input](run.in) uses a zero time step to sample the supplied structure. It sets the wavelength to 1.541838 Å, enables Lorentz–polarization correction, and samples 10–150°. The crystal mesh uses the inverse box lengths. The amorphous mesh uses 0.01 Å⁻¹ along each axis. Allow about 400 MB of temporary disk space per amorphous case.

GPUMD writes raw reciprocal points to `xrd.txt` in a temporary directory. The columns are three mesh indices, 2θ in degrees, and intensity. The driver sums intensities over 150 bins from 10–100° and divides by the total intensity within that range. Its tab-separated output contains both the bin center in 2θ and the corresponding d-spacing. The full output has 750 data rows. On failure, the driver keeps the failing case directory and saves the completed cases separately.

## Compare with the published spectra

The three amorphous structures are the final frames after 100 ps at 300 K and fixed volume. The [archived reference table](results/xrd-reference.txt) and [reference plots](results/xrd-reference.pdf) retain published Fig. S6 values and earlier calculations at the study's scattering parameters. They document the historical comparison. The analytic run above writes only its own GPUMD intensities.

[Li et al., ACS Nano 20, 11284–11293 (2026)](https://doi.org/10.1021/acsnano.6c00656)
