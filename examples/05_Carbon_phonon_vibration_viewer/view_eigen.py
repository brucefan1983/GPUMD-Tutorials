#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: LiangTing
23/9/2025 23:26:57
"""

import numpy as np
import os
from ase.io import read, write

def get_frequency_eigen_info(q_vaule, frequency, eig_file='band.yaml'):
    import yaml
    with open(eig_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)             # Load the yaml file and convert it to a dictionary
        qpoints_number = data['nqpoint']     # get number of qpoints used to calculate
        atoms_number = data['natom']         # get number of atoms in poscar
        phonons = data.get('phonon')         # get phonon dictionary
        
        print("\n****** Now we have {} q-points and {} branches for dispersion ******\n".format(qpoints_number, atoms_number*3))

    # for extract eigenvector
    print("***** Dealing {0} qpoint and {1} THz frequency for the structure ! *****\n".format(q_vaule, frequency))
    
    eigenvectors = []
    for phonon in phonons:
        if q_vaule == phonon['distance']:
            for freq_eig in phonon['band']:
                if frequency == freq_eig['frequency']:
                    eigenvectors = freq_eig['eigenvector']
                    break
            break

    return eigenvectors, atoms_number

def position_plus_eigen(structure, eigenvectors, atoms_number):

    if atoms_number != np.size(eigenvectors, 0):
        raise ValueError("The data dimension of the eigenvector is inconsistent with atomic number")
    # deal with eigenvector
    from tqdm import tqdm
    import copy
    positions = structure.get_positions()
    positions_second_frame = copy.deepcopy(positions)
    for i, atom in tqdm(enumerate(eigenvectors), desc="Processing eigenvectors", unit=" atoms"):
        positions_second_frame[i][0] = positions[i][0] + atom[0][0]  # x
        positions_second_frame[i][1] = positions[i][1] + atom[1][0]  # y
        positions_second_frame[i][2] = positions[i][2] + atom[2][0]  # z

    return positions_second_frame

if __name__ == "__main__":
    
    frequency = 5.3329508737
    q_vaule = 0.00   # Get from phonon dispersion
    
    file = 'POSCAR_DNWs'
    
    structure = read(file)
    
    eigenvectors, atoms_number = get_frequency_eigen_info(q_vaule, frequency)
    
    #print(eigenvectors)
    
    positions_second_frame = position_plus_eigen(structure, eigenvectors, atoms_number)
    # write first frame
    write("eigen_DNWs_Gamma_qpoint-{:0.6}.xyz".format(q_vaule), structure, append=False)
    	
    # write second frame
    structure.set_positions(positions_second_frame)
    write("eigen_DNWs_Gamma_qpoint-{:0.6}.xyz".format(q_vaule), structure, append=True)

    print('\n******************** All Done !!! *************************')