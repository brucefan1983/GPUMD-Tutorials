# Fine-Tuning the NEP89 Model for Calculating the Thermal Conductivity of MoS₂

In this tutorial, we provide a step-by-step guide on using the [NEP89 foundation model](https://github.com/brucefan1983/GPUMD/blob/master/potentials/nep/nep89_20250409/nep89_20250409.txt) for property calculations and fine-tuning. 

Here, we focus on calculating the thermal conductivity of monolayer MoS₂ as an example, demonstrating how to fine-tune NEP89 to achieve accurate physical properties when out-of-the-box predictions are insufficient. 
The fine-tuning results for the thermal conductivity of monolayer MoS₂ are presented below.

We also strongly encourage readers to reproduce the examples from the [NEP89 manuscript](https://arxiv.org/pdf/2504.21286), both in its out-of-the-box and fine-tuned applications.

<img src="https://github.com/Tingliangstu/GPUMD-Tutorials/blob/main/examples/26_fine_tune_NEP89/Figures/TC_MoS2.png" alt="Thermal Conductivity of MoS₂" width="800">

### If anyone has any questions or suggestions regarding this tutorial, please feel free to email me at: [liangting.zj@gmail.com](liangting.zj@gmail.com))

## 1. Introduction

Machine learning potentials (MLPs) covering the entire periodic table, often referred to as *foundation models* or *universal potentials*, have gained significant attention in recent years (see <a href="#references">References [1–6]</a>).

To stay updated on the latest developments, explore the [Matbench Discovery leaderboard](https://matbench-discovery.materialsproject.org/contribute).

While the NEP89 model may not achieve very high training accuracy, its training dataset encompasses both organic and inorganic materials, enabling molecular dynamics (MD) simulations across 89 elements. 

**NEP89's key strengths include:**

✅ **Comprehensive Coverage**: Supports 89 elements, covering both inorganic and organic materials.  

✅ **Exceptional Speed**: Over 1000x faster than comparable models, capable of simulating 15 million atoms on a single GPU.
	
✅ **Out-of-the-box functionality**: Instantly supports large-scale molecular dynamics simulations

**See the speed comparison below:**  

<img src="https://github.com/Tingliangstu/GPUMD-Tutorials/blob/main/examples/26_fine_tune_NEP89/Figures/speed.png" alt="Speed Comparison" width="800">

For more details, refer to the [NEP89 manuscript](https://arxiv.org/pdf/2504.21286) and the associated [WeChat article](https://mp.weixin.qq.com/s/D8j73BOke8o63BSnukebgg).
	
NEP89 is increasingly adopted in various studies and may soon be included in the [Matbench Discovery leaderboard](https://matbench-discovery.materialsproject.org/contribute).

**Potential applications for NEP89 include:**

✅ **Out-of-the-Box Simulations**: Directly applicable to diverse systems and MD scenarios.  

✅ **Fine-Tuning**: Enhanced accuracy with minimal DFT calculations for targeted MD simulations.  

✅ **Replacing AIMD**: Generating realistic MD configurations to build training datasets.
	

## 2. Accessing the NEP89 Model

The NEP89 model is included in the [GPUMD package](https://github.com/brucefan1983/GPUMD/blob/master/potentials/nep/nep89_20250409/nep89_20250409.txt). The [`nep89_20250409` folder](https://github.com/brucefan1983/GPUMD/tree/master/potentials/nep/nep89_20250409) contains three files:  
- `nep89_20250409.txt`: The NEP89 model file.  
- `nep.in` and `nep89_20250409.restart`: Used for fine-tuning (discussed later).

## 3. Out-of-the-Box Application: Thermal Conductivity of MoS₂

We begin by using NEP89 to calculate the thermal conductivity of monolayer MoS₂. The [`model.xyz`](Out-of-the-box/model.xyz) file for MoS₂ is available in the working directory. 
Since MoS₂ is a 2D material, the periodic boundary conditions are set as `pbc="T T F"`.

Below is an example [`run.in`](Out-of-the-box/run.in) file for computing thermal conductivity using the Homogeneous Non-Equilibrium Molecular Dynamics (HNEMD) method:

```plaintext
potential      nep89_20250409.txt
velocity       300

ensemble       npt_scr 300 300 100 0 0 0 20 20 100 1000
time_step      1
dump_thermo    10000
dump_position  200000
run            2000000 

ensemble       nvt_nhc 300 300 100
compute_hnemd  1000 0 0.00001 0
compute_shc    2 500 1 1000 400

dump_position  1000000
run            10000000          ## 10 ns
```

For detailed instructions on calculating thermal conductivity using HNEMD, refer to the [GPUMD tutorial on thermal transport](https://github.com/brucefan1983/GPUMD-Tutorials/blob/main/examples/04_Carbon_thermal_transport_nemd_and_hnemd/diffusive/tutorial.ipynb). 
Note that performing multiple independent HNEMD simulations to obtain standard errors is recommended.

In our case, 10 HNEMD simulations yielded a thermal conductivity of **`64.5163 ± 3.6378 W/m/K`** (see below figure). 
This result significantly deviates from both the specialized NEP model by [Jiang et al.](https://arxiv.org/abs/2505.00376) and [DFT-BTE calculations](https://pubs.aip.org/aip/jap/article/119/8/085106/143937), 
indicating that NEP89's out-of-the-box performance is suboptimal for MoS₂ thermal conductivity.

<img src="https://github.com/Tingliangstu/GPUMD-Tutorials/blob/main/examples/26_fine_tune_NEP89/Figures/out-of-the-box-TC.png" alt="out-of-the-box-TC" width="600">

## 4. Utilize NEP89 to generate fine-tuned configurations

To improve accuracy, NEP89 can be fine-tuned using a small set of DFT calculations to generate configurations tailored for MoS₂.

We recommend that readers use the NEP89 model for MD simulations in their own target applications and output atomic trajectories for fine-tuning.

In our case, we are interested in the thermal conductivity of MoS₂ at 300 K. Therefore, we performed an NPT simulation at 300 K.

It is important to note that the sampled configurations must later be used for single-point DFT calculations to obtain reference energies, forces, and stresses. 
Therefore, the number of atoms in the MD sampling should not be too large, ensuring that DFT calculations remain computationally feasible.

An example [run.in](run-MD-for-fine-tuning/run.in) is shown below:

```plaintext
potential      nep89_20250409.txt
velocity       300

ensemble       npt_scr 300 300 100 0 0 0 20 20 100 1000
time_step      1
dump_thermo    1000
dump_position  3000                                     # dump movie.xyz
run            3000000
```

Here, we performed a **3 ns NPT simulation** and sampled configurations every **3 ps**, giving a total of **1000 frames**.  

To achieve better sample phase space, we carried out **two independent random MD simulations** and then applied **farthest point sampling (FPS)** to obtain **104 representative frames**.

The FPS procedure was performed using the script [`nep-select-fps.py`](run-MD-for-fine-tuning/nep-select-fps.py).  
Of course, many other tools can achieve the same purpose.

In this script, **lines 85–87** need to be adjusted as appropriate:
	
```python
calc = NEP("nep_0409_virial.txt")
min_distance = 0.00082
get_selected_frames("movie.xyz", calc, min_distance)
```

The parameter min_distance defines the distance threshold for selecting configurations, and tuning this value allows users to control the number of sampled frames.

## 5. Single-point calculation of DFT

After obtaining the configurations, **single-point DFT calculations** can be performed.  
In our case, we used [**VASP**](https://www.vasp.at). An example [`INCAR`](SCF-calculations/INCAR) file is shown below:

```plaintext
ISTART =  0            (Not read existing wavefunction)
ISPIN  =  1            (Non-spin-polarized DFT)

ICHARG =  2            (Initial guess from superposition of atomic charge density)
LREAL  =  Auto         (Projection operators: automatic)
ENCUT  =  520          (Plane-wave cutoff energy in eV)

IVDW   =  12           (Many-body dispersion correction enabled)

LWAVE  = .FALSE.       (Do not write WAVECAR)
LCHARG = .FALSE.       (Do not write CHGCAR)

PREC     =  Accurate
KGAMMA   = .TRUE.      # Gamma point only
KSPACING = 0.15        # Automatic k-point generation
ALGO     = Normal

# Static calculation
NSW    =   1
IBRION =  -1
ISMEAR =  -5           (Gaussian smearing method)

EDIFF  =  1E-06        (Energy convergence, eV)
NELM   =  150          (Maximum SCF steps)
```

The choice of **INCAR parameters** requires users to carefully study the [VASP manual](https://www.vasp.at).  
The quality of the single-point calculations is **crucial** for dataset preparation.

In our fine-tuning dataset, we included **[`IVDW = 12`](https://www.vasp.at/wiki/DFT-D3)**.  
For systems requiring long-range dispersion corrections, users may alternatively **[`IVDW = 12`](https://www.vasp.at/wiki/DFT-D3)** during the single-point DFT step and instead adopt the [NEP+D3 strategy](https://iopscience.iop.org/article/10.1088/1361-648X/ad1278) during MD simulations.

---

After completing the single-point calculations, we used [`vasp2nep.py`](SCF-calculations/vasp2nep.py) to **extract energy/force/virial data** and output them into [`train.xyz`](https://gpumd.org/nep/input_files/train_test_xyz.html).  
Similar tools can also be found in the [GPUMD repository](https://github.com/brucefan1983/GPUMD/tree/master/tools/Format_Conversion/vasp2xyz).

In [`vasp2nep.py`](SCF-calculations/vasp2nep.py), users only need to modify the following lines:

```python
path = '26_fine_tune_NEP89/SCF-calculations'
include_virial = True
include_VDW = True
```

The script will then traverse all `OUTCAR` files in the given folder and output a combined `train.xyz`. 

Users may also further customize the script as needed.

## 6. Direct prediction of the configuration of MoS<sub>2</sub> by NEP89

Once [`train.xyz`](https://github.com/Tingliangstu/GPUMD-Tutorials/blob/main/examples/26_fine_tune_NEP89/prediction/train.xyz) has been prepared, it is useful to first test **NEP89's predictive performance** before fine-tuning.  

An example [`nep.in`](prediction/nep.in) file is given below:

```plaintext
type       89 H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Ac Th Pa U Np Pu 
version    4
zbl        2
cutoff     6 5
n_max      4 4
basis_size 8 8
l_max      4 2 1
neuron     80
lambda_1   0
lambda_e   1
lambda_f   1
lambda_v   1
batch      5000
prediction 1
population 10
generation 1000
```

Compared to the standard NEP input, the **key difference** is the addition of:

```plaintext
prediction 1
```

This enables **[prediction mode](https://gpumd.org/nep/input_parameters/prediction.html#prediction)**.

The script [`Plot_prediction.py`](prediction/Plot_prediction.py) can then be used to **visualize the prediction results**. This plotting script is adapted from [Zihan Yan](https://github.com/zhyan0603/GPUMDkit/blob/main/Scripts/plt_scripts/plt_nep_prediction_results.py).

The prediction results for **MoS₂** are shown below:

<img src="https://github.com/Tingliangstu/GPUMD-Tutorials/blob/main/examples/26_fine_tune_NEP89/prediction/prediction.png" alt="prediction" width="900">

It can be seen that NEP89, when directly predicting the fine-tuning training set ([`train.xyz`](https://github.com/Tingliangstu/GPUMD-Tutorials/blob/main/examples/26_fine_tune_NEP89/run-fine-tuning/train.xyz)), shows a certain degree of deviation.  
This issue has also been noted in a recent [preprint](https://arxiv.org/abs/2509.13798).  
In practice, however, this can be effectively resolved through **fine-tuning**.

## 7. Procedure of fine-tuning NEP89 

Next, we demonstrate how to perform fine-tuning using the prepared [`train.xyz`](https://github.com/Tingliangstu/GPUMD-Tutorials/blob/main/examples/26_fine_tune_NEP89/run-fine-tuning/train.xyz).

We recommend that readers first review the GPUMD manual section on the [`fine_tune`](https://gpumd.org/nep/input_parameters/fine_tune.html#fine-tune) command.  
	
For clarity, we also provide here a simplified schematic illustration of the potential energy surface before and after fine-tuning.

<img src="https://github.com/Tingliangstu/GPUMD-Tutorials/blob/main/examples/26_fine_tune_NEP89/Figures/fine-tuned-PES.png" alt="prediction" width="800">


## 8. Re-calculation of the thermal conductivity of MoS<sub>2</sub> using the fine-tuned model




## References

[1] Liang T, Xu K, Lindgren E, et al. [NEP89: Universal neuroevolution potential for inorganic and organic materials across 89 elements](https://arxiv.org/abs/2504.21286). arXiv preprint arXiv:2504.21286, 2025.

[2] Xia J, Zhang Y, Jiang B. [The evolution of machine learning potentials for molecules, reactions and materials](https://pubs.rsc.org/en/content/articlehtml/2025/cs/d5cs00104h). Chemical Society Reviews, 2025.

[3] Riebesell J, Goodall R E A, Benner P, et al. [A framework to evaluate machine learning crystal stability predictions](). Nature Machine Intelligence, 2025, 7(6): 836-847.

[4] Wood B M, Dzamba M, Fu X, et al. [UMA: A Family of Universal Models for Atoms](https://arxiv.org/abs/2506.23971). arXiv preprint arXiv:2506.23971, 2025.

[5]	Deng B, Zhong P, Jun K J, et al. [CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling](https://www.nature.com/articles/s42256-023-00716-3). Nature Machine Intelligence, 2023, 5(9): 1031-1041.

[6] Batatia I, Benner P, Chiang Y, et al. [A foundation model for atomistic materials chemistry](https://arxiv.org/abs/2401.00096). arXiv preprint arXiv:2401.00096, 2023.

[7] Jiang W, Bu H, Liang T, et al. [Accurate Modeling of Interfacial Thermal Transport in van der Waals Heterostructures via Hybrid Machine Learning and Registry-Dependent Potentials](https://arxiv.org/abs/2505.00376). arXiv preprint arXiv:2505.00376, 2025.

[8] Gu X, Li B, Yang R. [Layer thickness-dependent phonon properties and thermal conductivity of MoS₂](https://pubs.aip.org/aip/jap/article/119/8/085106/143937). Journal of Applied Physics, 2016, 119(8).
