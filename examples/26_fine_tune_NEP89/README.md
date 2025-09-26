# Fine-Tuning the NEP89 Model for Calculating the Thermal Conductivity of MoS₂

In this tutorial, we provide a step-by-step guide on using the [NEP89 foundation model](https://github.com/brucefan1983/GPUMD/blob/master/potentials/nep/nep89_20250409/nep89_20250409.txt) for property calculations and fine-tuning. 

Here, we focus on calculating the thermal conductivity of monolayer MoS₂ as an example, demonstrating how to fine-tune NEP89 to achieve accurate physical properties when out-of-the-box predictions are insufficient. 
The fine-tuning results for the thermal conductivity of monolayer MoS₂ are presented below.

We also strongly encourage readers to reproduce the examples from the [NEP89 manuscript](https://arxiv.org/pdf/2504.21286), both in its out-of-the-box and fine-tuned applications.

<img src="https://github.com/Tingliangstu/GPUMD-Tutorials/blob/main/examples/26_fine_tune_NEP89/Figures/TC_MoS2.png" alt="Thermal Conductivity of MoS₂" width="800">



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

Below is an example `run.in`]() file for computing thermal conductivity using the Homogeneous Non-Equilibrium Molecular Dynamics (HNEMD) method:

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

In our case, 10 HNEMD simulations yielded a thermal conductivity of **`64.5163 ± 3.6378 W/m/K`**. 
This result significantly deviates from both the specialized NEP model by [Jiang et al.](https://arxiv.org/abs/2505.00376) and [DFT-BTE calculations](https://pubs.aip.org/aip/jap/article/119/8/085106/143937), 
indicating that NEP89's out-of-the-box performance is suboptimal for MoS₂ thermal conductivity.


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
dump_position  3000
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



## 6. Direct prediction of the configuration of MoS<sub>2</sub> by NEP89


## 7. Procedure of fine-tuning NEP89 


## 8. Re-calculation of the thermal conductivity of MoS<sub>2</sub> using the fine-tuned model




### References

[1] Liang T, Xu K, Lindgren E, et al. [NEP89: Universal neuroevolution potential for inorganic and organic materials across 89 elements](https://arxiv.org/abs/2504.21286). arXiv preprint arXiv:2504.21286, 2025.

[2] Xia J, Zhang Y, Jiang B. [The evolution of machine learning potentials for molecules, reactions and materials](https://pubs.rsc.org/en/content/articlehtml/2025/cs/d5cs00104h). Chemical Society Reviews, 2025.

[3] Riebesell J, Goodall R E A, Benner P, et al. [A framework to evaluate machine learning crystal stability predictions](). Nature Machine Intelligence, 2025, 7(6): 836-847.

[4] Wood B M, Dzamba M, Fu X, et al. [UMA: A Family of Universal Models for Atoms](https://arxiv.org/abs/2506.23971). arXiv preprint arXiv:2506.23971, 2025.

[5]	Deng B, Zhong P, Jun K J, et al. [CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling](https://www.nature.com/articles/s42256-023-00716-3). Nature Machine Intelligence, 2023, 5(9): 1031-1041.

[6] Batatia I, Benner P, Chiang Y, et al. [A foundation model for atomistic materials chemistry](https://arxiv.org/abs/2401.00096). arXiv preprint arXiv:2401.00096, 2023.

[7] Jiang W, Bu H, Liang T, et al. [Accurate Modeling of Interfacial Thermal Transport in van der Waals Heterostructures via Hybrid Machine Learning and Registry-Dependent Potentials](https://arxiv.org/abs/2505.00376). arXiv preprint arXiv:2505.00376, 2025.

[8] Gu X, Li B, Yang R. [Layer thickness-dependent phonon properties and thermal conductivity of MoS₂](https://pubs.aip.org/aip/jap/article/119/8/085106/143937). Journal of Applied Physics, 2016, 119(8).

[9]