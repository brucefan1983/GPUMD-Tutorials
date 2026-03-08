import os
from ase.io import read,write
import numpy as np

temperature = 115
time_step = 1
text = f"""
potential   ./lj.txt
velocity    {temperature}

ensemble nvt_nhc {temperature} {temperature} 100
dump_thermo 1000
time_step {time_step}
run 10000

ensemble nvt_nhc {temperature} {temperature} 100
dump_thermo 1000
time_step {time_step}
compute_hnemdec 0 1000 0.1 0 0
run 50000
"""

frame = read("./inputfile/model.xyz")
md_number = 1
fe = 0.002
for i in range(md_number):
    # create work dir
    work_dir = os.path.join("md1", f"md-{i}")
    if os.path.exists(work_dir):
        continue
    os.makedirs(work_dir)
    
    # write run.in
    text = f"""
potential   ./lj.txt
velocity    {temperature}

ensemble nvt_nhc {temperature} {temperature} 100
dump_thermo 1000
time_step {time_step}
run 10000

ensemble nvt_nhc {temperature} {temperature} 100
dump_thermo 1000
time_step {time_step}
compute_hnemdec 1 10 {fe} 0 0
run 500000
"""
    run_in = os.path.join(work_dir,"run.in")
    with open(run_in,"w") as f:
        f.write(text)

    # write model.xyz
    model = os.path.join(work_dir,"model.xyz")
    write(model,frame,format="extxyz")

    # copy jobfile
    assert 0==os.system(f"cp ./inputfile/jobfile {work_dir}")

    # link force-field file
    potential_file = os.path.abspath("./inputfile/lj.txt")
    assert 0==os.system(f"ln -s {potential_file} {os.path.join(work_dir,'lj.txt')}")

    # sub jobfile
    assert 0==os.system(f"cd {work_dir};bsub<jobfile")
