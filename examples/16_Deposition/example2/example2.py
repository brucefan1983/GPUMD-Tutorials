from ase.build import bulk
from ase import Atoms
import deposition
import os

if os.access('./cluster', os.F_OK):
	print(' ')
else:
	os.mkdir('./cluster')

# Generating deposition atoms/clusters
Si_fcc = bulk('Si', 'diamond', a=5.43, cubic=True)
dep_atom1 = {
	'a_type': Si_fcc.symbols, 'position': Si_fcc.positions,
	'mass': 28.0855, 'name': './cluster/0.txt', 'exyz': 1}
deposition.build_dep_atom(dep_atom1)
dep_atom2 = {
	'a_type': Si_fcc.symbols[:4], 'position': Si_fcc.positions[:4],
	'mass': 28.0855, 'name': './cluster/1.txt', 'exyz': 1}
deposition.build_dep_atom(dep_atom2)
dep_atom3 = {
	'a_type': ['Si'], 'position': [[0,0,0]],
	'mass': 28.0855, 'name': './cluster/2.txt', 'exyz': 1}
deposition.build_dep_atom(dep_atom3)

# Building the deposition substrate
dep_sub = {'name': 'model.xyz', 'im_file': 'model0.xyz', 'vacuum': 10 , 'defect_p': 0.1}
deposition.build_dep_sub(dep_sub)

# Main cycle
params = {
    'path': '/home/tjs/new/k/GPUMD-3.9.4/GPUMD-3.9.4/src/gpumd',
    'cycle': 2,
    'species': 3,
    'number': 20,
    'vz': [0.005, 0.01, 1],
    'h_range': [50,60],
    'h_cutoff': 50,
    'group': 3,
    'number': 25,
    'prob': 0
}
deposition.para(params)
deposition.main_dep()