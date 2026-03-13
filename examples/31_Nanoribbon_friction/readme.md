# Nanoribbon friction simulation with GPUMD (H-BNNR on h-BN substrate)

## Reference

This example is based on the following paper:

**Modular Hybrid Machine Learning and Physics-based Potentials for Scalable Modeling of van der Waals Heterostructures**, 
Hekai Bu, Wenwu Jiang, Penghua Ying, Ting Liang, Zheyong Fan, Wengen Ouyang, 
*Journal of the Mechanics and Physics of Solids* **210**, 106540 (2026).  
DOI: <https://doi.org/10.1016/j.jmps.2026.106540>

---

## 1. Model

This tutorial performs friction simulation for a hydrogen-passivated $h$-BN nanoribbon (H-BNNR) sliding on an $h$-BN substrate.
The structure corresponds to panel c in Fig. 7 of the reference paper.


<img src="https://raw.githubusercontent.com/BBBuZHIDAO/figures_gpumd_tutorials/master/31_nanoribbon_friction/model.jpg" width="200">

Color legend:

- mauve: boron in substrate
- blue: nitrogen in substrate
- pink: boron in slider
- light blue: nitrogen in slider
- white: hydrogen atoms

### 1.1 `model.xyz` group settings

There are two integer group columns.

- **Group method 1** (the 5th column) defines layer membership (from bottom to top):
	- `group_id=0`: substrate layer 1, 9360 atoms (B 4680 + N 4680)
	- `group_id=1`: substrate layer 2, 9360 atoms (B 4680 + N 4680)
	- `group_id=2`: substrate layer 3, 9360 atoms (B 4680 + N 4680)
	- `group_id=3`: slider layer (H-BNNR), 1798 atoms (B 757 + N 757 + H 284)

- **Group method 2** (the 6th column) is only for selecting atoms pulled by spring force:
	- `group_id=0`: normal atoms (not directly spring-pulled), 29874 atoms
	- `group_id=1`: spring-driven atoms, 4 atoms in slider

In `run.in`, `add_spring    ghost_atom 1 1 ...` uses this second grouping method to apply pulling only to the atoms in group 1 of group method 2.

Main input files in this folder:

- `model.xyz`: atomic structure and group labels
- `run.in`: GPUMD control script
- `plot_friction.py`: post-processing for spring force
- `map.in`: atom-type mapping for hybrid potential
- `gr_bn_mos2.ilp`: ILP parameter file
- `nep.txt`: NEP model file

---

## 2. Simulation settings

### 2.1 Hybrid potential (NEP + ILP)

This example uses GPUMD hybrid potential style with:

```txt
potential gr_bn_mos2.ilp map.in
```

and the mapping file:

```txt
0 1 nep.txt
4
0
0
0
0
```

Please check official documentation for the `nep_ilp` setup details:  
<https://gpumd.org/dev/potentials/nep_ilp.html>

### 2.2 MD control in `run.in`

Current `run.in` is:

```txt
potential gr_bn_mos2.ilp map.in
velocity 0.00001

ensemble      heat_lan 0.0000000001 1000 0 0 1
time_step     1
add_spring    ghost_atom 1 1 0.00001 0 0 decouple 0.625 0 0 0 0 0
fix           0
dump_thermo   5000
dump_exyz     2000 1 1 1
run           2000000
```

The simulation is conducted at a nominal temperature of 0 K. Because the spring performs work on the system, a Langevin thermostat (`heat_lan`) is applied to the atoms in groups 0 and 1 of group method 0 to dissipate the injected energy and suppress any appreciable temperature rise. Actually, the energy will only dissipate through the second substrate layer (group 1 of group method 0) because the first substrate layer (group 0 of group method 0) is fixed by `fix 0`.

The spring is applied to the 4 atoms in group 1 of group method 2, which are the 4 atoms in the head of the H-BNNR slider. The spring velocity is 0.00001 Å/step, and the spring constant is 0.625 eV/Å^2. The `decouple` option means the other degrees of freedom will not be affected by the spring force.

Parameter meaning can be found in official docs:

- `ensemble heat_lan`: <https://gpumd.org/dev/gpumd/input_parameters/ensemble_heat.html>
- `add_spring`: <https://gpumd.org/dev/gpumd/input_parameters/add_spring.html>
- `fix`: <https://gpumd.org/dev/gpumd/input_parameters/fix.html>

---

## 3. How to run

Run GPUMD in this directory:

```bash
gpumd
```

Typical outputs include:

- `thermo.out`
- `dump.xyz`
- `spring_force_0.out`


`spring_force_0.out` is used in the following 7-column format:

```txt
step  mode  Fx  Fy  Fz  Ftotal  energy
```

- `step`: MD step index
- `mode`: spring mode tag written by GPUMD
- `Fx, Fy, Fz`: spring-force components on selected pulled atoms (unit: eV/Å)
- `Ftotal`: total spring force magnitude (unit: eV/Å)
- `energy`: spring-related energy term (unit: eV)

---

## 4. Post-processing

Plot spring-force curve:

```bash
python plot_friction.py --save -o spring_force_force.png
```


Example figure:

<img src="https://raw.githubusercontent.com/BBBuZHIDAO/figures_gpumd_tutorials/master/31_nanoribbon_friction/spring_force.png" width="440">

We can see the stick-slip pattern in the spring force curve, which is a hallmark of atomic-scale friction. The force curve can be further analyzed to extract frictional properties such as average friction force, slip length, etc.
