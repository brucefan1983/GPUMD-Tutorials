from ase.build import bulk,diamond100
from ase import Atoms
import deposition
import os 

if os.access('./cluster', os.F_OK):
    print(' ')
else:
    os.mkdir('./cluster')

# Example of generating deposition atoms/clusters
# Building diamond-structured silicon with ASE
Si_fcc = bulk('Si', 'diamond', a=5.43, cubic=True)
dep_atom1 = {
	'a_type': Si_fcc.symbols, 'position': Si_fcc.positions,
	'mass': 28.0855, 'name': './cluster/0.txt', 'exyz': 1}
deposition.build_dep_atom(dep_atom1)

# Monatomic silicon
dep_atom2 = {
	'a_type': ['Si'], 'position': [[0,0,0]],
	'mass': 28.0855, 'name': './cluster/1.txt'}
deposition.build_dep_atom(dep_atom2)

# Polyatomic clusters
dep_atom3 = {
	'a_type': ['Si','O','O'], 'position': [[0,0,0],[0,1.62,0],[0,0,1.62]],
	'mass': ['Si 28.0855','O 16.00'], 'name': './cluster/2.txt', 'exyz': 1}
deposition.build_dep_atom(dep_atom3)

# Building silicon clusters with Python
cluster = []
for i in range(6):
	cluster.append('Si')
dep_atom4 = {
	'a_type': cluster, 
	'position': [[0,0,0],[1.3575,1.3575,1.3575],[0,2.715,2.715],
	[2.715,0,2.715],[2.715,2.715,0],[4.0725,4.0725,1.375]],
	'mass': 28.0855, 'name': './cluster/3.txt'}
deposition.build_dep_atom(dep_atom4)


# Building the deposition substrate with ASE
Si_100 = diamond100('Si', size=(16, 16, 20), a = 5.43, orthogonal = True)
dep_sub1 = {
	'a_type': Si_100.symbols, 'position': Si_100.positions,
	'mass': 28.0855, 'name': 'model.xyz',
	'Cell': Si_100.cell, 'z': 150}
deposition.build_dep_sub(dep_sub1)
# Refactor an existing EXYZ file into a deposition-ready substrate
dep_sub2 = {'name': 'model0.xyz', 'im_file': 'model.xyz', 'vacuum': 0}
deposition.build_dep_sub(dep_sub2)
dep_sub3 = {'name': 'GaO0.xyz', 'im_file': 'GaO.xyz', 'defect_l': 2, 'defect_p': 0.5}
deposition.build_dep_sub(dep_sub3)
