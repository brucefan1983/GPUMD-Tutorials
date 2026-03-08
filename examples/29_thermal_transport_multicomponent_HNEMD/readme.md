# thermal conductivity of Ar–Kr binary mixture

## Files
 - */inputfile: structure, force field, task job.
 - */write_input*.py: batch generate and execute EMD or HNEMD tasks.
 - */postprocess/*.py: script for postprocess.

## How to run
### EMD method
Enter the `emd` folder:
```shell
1. cd emd
```
Execute write_input.py to generate a series of EMD task (`./md/md-0`, `./md/md-1`, ...) and  submit them automatically, the number of task is controled by `md_number` in `write_input.py`:
```shell
2. python write_input.py
```
When the tasks finishing, enter `./postprocess` and then execute `Correlation.py` to compute the correlation and interal. We recommend using GPU to accelerate the computation. The parameters need to be modified in `Correlation.py` includes `device`,`temperature`,`time_step`,`volume`,`group_num`,`sampling_interval`,`correlation_steps`,`output_interval`,`runs`.
```shell
3. cd postprocess
4. python Correlation.py
```
Finally, execute `kappa.py` to compute thermal conductivity:
```shell
5. python kappa.py
```
### HNEMD method
Enter the `hnemd` folder:
```shell
1. cd hnemd
```
Exectute `write_input0.py` and `write_input1.py` to generate HNEMD tasks (`./md0/md-*` and `./md1/md-*`). One needs to modify `md_number` and `fe` in `write_input*.py` to control task number and driving force. In `write_input0.py`, the driving force is thermal driving force, dissipative flux is equilvalent to the heat flux, so we can compute the couple between heat flux and any other flux. In `write_input1.py`, the driving force is diffusive driving force, the dissipative flux is equilvalent to 1st element's momentum (Ar's momentum in this example). Similarly, we can compute the couple between 1st element's momentum and any other flux.
```shell
2. python write_input0.py
3. python write_input1.py
```
Finally, enter `postprocess` and execute `plot-hnemd.py` to plot the onsager coefficients and calculate thermal conductivity.
```shell
4. python plot-hnemd.py
```

## Reference
Under revirew.