import deposition
import os

if os.access('./cluster', os.F_OK):
    print(' ')
else:
    os.mkdir('./cluster')

# Generating deposition atoms/clusters
dep_atom = {
    'a_type': ['Ga','Ga','O','O','O'], 'position': [[11.0403,0,0],[10.209,1.54595,3.03374],[12.0823,1.54595,0.04904],[10.0253,0,1.60612],[10.1454,0,4.25799]],
    'mass': ['Ga 69.723','O 16.00'], 'name': './cluster/0.txt', 'exyz': 1}
deposition.build_dep_atom(dep_atom)
dep_atom = {
    'a_type': ['Ga','O'], 'position': [[0,0,0],[1,1,1]],
    'mass': ['Ga 69.723','O 16.00'], 'name': './cluster/1.txt'}
deposition.build_dep_atom(dep_atom)
dep_atom = {
    'a_type': ['O'], 'position': [[0,0,0]],
    'mass': 16.00, 'name': './cluster/2.txt'}
deposition.build_dep_atom(dep_atom)
dep_atom = {
    'a_type': ['Ga'], 'position': [[0,0,0]],
    'mass': 69.723, 'name': './cluster/3.txt'}
deposition.build_dep_atom(dep_atom)

# Building the deposition substrate
dep_sub = {'name': 'model.xyz', 'im_file': 'GaO.xyz', 'vacuum': 10 , 'defect_l': 3}
deposition.build_dep_sub(dep_sub)

# Main cycle
params = {
    'path': '/home/tjs/new/k/GPUMD-3.9.4/GPUMD-3.9.4/src/gpumd',
    'cycle': 2,
    'species': 4,
    'number': 20,
    'vz': [0.005, 0.001, 0],
    'h_c': [20,25],
    'h_cutoff': 100,
    'group': 3,
    'number': 25,
    'prob': 2,
    'sto_ratio': [['Ga','O'],[2, 3]]
}

deposition.para(params)
deposition.main_dep()