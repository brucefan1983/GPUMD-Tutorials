from deposit import read_xyz, gpumd
import numpy as np

run_in =['potential nep.txt', 
         'velocity 300', 
         'time_step 0.5', 
         'ensemble nve',
         'fix 0',
         'dump_exyz 1000',
         'run 10000',
         'time_step 1', 
         'ensemble nvt_nhc 300 300 100', 
         'dump_thermo 1000', 
         'dump_restart 10000', 
         'run 50000']

for i in range(20):
    if i == 0:
        atoms = read_xyz('relax/restart.xyz')
    else:
        atoms = read_xyz('deposit/{}/restart.xyz'.format(i-1))
    
    group = []
    thickness = 5 #A 
    pos = atoms.positions
    for p in pos:
        if p[2] < thickness:
            group.append(0)  # fixed group
        else:
            group.append(1) 
    atoms.groups = [group]

    cell = atoms.get_cell()
    for _ in range(20):
        a = cell[0]
        b = cell[1]
        u = np.random.rand()
        v = np.random.rand()
        r = u * a + v * b
        
        x = r[0]
        y = r[1]
        z = 60    
        
        atoms.append(
            'C',
            position=(x, y, z),
            velocity=(0, 0, -0.01),  
            group_idxs=[1]          
        )

    atoms.zero_momentum()
    gpumd('deposit/{}'.format(i), atoms, run_in, 'relax/nep.txt')