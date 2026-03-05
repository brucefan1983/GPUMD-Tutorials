# Heat conduction in superionic phases

**Author:** Wenjiang Zhou  
**Email:** wjzhou@stu.pku.edu.cn

## Check codes
1. Run GPUMD
2. python GK.py
3. python GK_comparison.py

## Perform simulation
4. mkdir 650K; cd 650K
5. Run GPUMD
6. cd ..
7. python GK.py

## Main results
Temperature_kappa_650K.txt

## Post processing
1. cd 650K
2. python GK_ave_plot.py

## Note
1. The superionic ion needs to be grouped into 1 set for the momentum calculation.
2. In practice, for three or more multi-component systems, additional considerations are required [1].
3. The correlation time needs to be tested.  

## Reference
[1] W. Zhou, B. Tang, Z. Fan, F. Grasselli, S. Baroni, and B. Song, Heat transport in superionic materials via machine-learned molecular dynamics. 10.48550/arXiv.2512.04718.

