# ============================================================
# Author: Ting Liang at 22/9/2025 20:21:17
# Email: liangting.zj@gmail.com
# Description: Calculating the phonon dispersion using NEP model
# ============================================================

# ASE structure

from ase.io import read, write
from ase.units import GPa

# Calorine
from calorine.tools import get_force_constants, relax_structure
from calorine.calculators import CPUNEP

import matplotlib.pyplot as plt
import numpy as np

def relax(structure, eps, steps=100000):
    
    pressure = -np.sum(structure.get_stress()[:3]) / 3 / GPa
    print(f'pressure before: {pressure: 2f} GPa')

    print("******** Start relax structure, wait !!!!! *********")
    
    relax_structure(structure, fmax=eps, steps=steps, minimizer='fire', constant_cell=False,
                    constant_volume=False)                  # minimizer to use; possible values: 'bfgs', 'fire', 'gpmin', 'bfgs-scipy'
    print("******** Relax ALL Done !!!!! *********")
    
    pressure = -np.sum(structure.get_stress()[:3]) / 3 / GPa
    print(f'pressure after relax: {pressure: 2f} GPa')
    
    # write('POSCAR', structure)       # If you want to output the optimized structure, please remove the comments.
    
    return structure

def get_band_path(band_path, band_label, structure, use_seek_path=False, band_resolution=50):

    if use_seek_path:
        from seekpath import get_path
        path_data = get_path((structure.cell, structure.get_scaled_positions(), structure.numbers))
        
        labels = path_data['point_coords']
        bands_ranges = []

        for set in path_data['path']:
            bands_ranges.append([labels[set[0]], labels[set[1]]])
            
        bands_and_labels = {
            'ranges': bands_ranges,
            'labels': band_label
        }
        
        bands_and_labels = {'ranges': [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]], [[0.5, 0.0, 0.5], [0.625, 0.25, 0.625]], 
        	[[0.375, 0.375, 0.75], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]], 
        	'labels': [('Gamma', 'X'), ('X', 'U'), ('K', 'Gamma'), ('Gamma', 'L')]}

    elif band_path is not None:
        bands_ranges = []
        for i in range(len(band_path)):
            if i < len(band_path) - 1:
                bands_ranges.append(np.array([band_path[i], band_path[i + 1]]))
            else:
                break
        bands_and_labels = {
            'ranges': bands_ranges,
            'labels': band_label
        }
        
    else:
        print("Please provide high symmetry points paths")

    bands = []
    for q_start, q_end in bands_and_labels['ranges']:
        band = []
        for i in range(band_resolution + 1):
            band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
        bands.append(band)

    return bands, bands_and_labels

def phonon_calculator(relaxed_atoms, calculator, repeat_cell,
                      band_path, band_label, write_band_yaml=True,
                      force_constants=False, save_distance=False):
    """
    phonopy interface, get_force_constants function from calorine
    """
    
    phonon = get_force_constants(relaxed_atoms, calculator, repeat_cell)
    
    filename = "FORCE_CONSTANTS"
    
    if force_constants:
        from phonopy.file_IO import write_FORCE_CONSTANTS
        write_FORCE_CONSTANTS(phonon.get_force_constants(), filename=filename)
        print('\n******** {} file is written successfully **********'.format(filename))
        print('**** One can use it to generate Phonon dispersion with PHONOPY command line **** \n')

    print("******** Processing phonon information, wait !!!!! *********")
    # Get band path
    bands, bands_and_labels = get_band_path(band_path, band_label, relaxed_atoms)

    phonon.run_band_structure(bands,
                              with_group_velocities=True,
                              with_eigenvectors=True)

    bands_dict = phonon.get_band_structure_dict()
    bands_phonopy = (bands_dict['qpoints'],
                     bands_dict['distances'],
                     bands_dict['frequencies'],
                     bands_dict['eigenvectors'],
                     bands_dict['group_velocities'])

    if save_distance:
        output_file="distances.txt"
        with open(output_file, 'w') as f:
            for i, distance_array in enumerate(bands_dict['distances']):
                f.write(f"# Band {i+1} distances\n")
                np.savetxt(f, distance_array, fmt='%.8e')
                f.write("\n")
        print(f"Saved all distances to {output_file}")

    if write_band_yaml:
        phonon.write_yaml_band_structure()
        print('******** band.yaml file is written successfully **********')

    return bands_phonopy, bands_and_labels

def plot_band_structure(band_structure, bands_and_labels,
                        save_band_structure_file=True,
                        save_gv_file=False,
                        save_ave_gv_file=False,
                        show_dispersion=False):

    band_distances = []
    frequency = []
    group_velocities = []
    eigenvectors = []

    for i, freq in enumerate(band_structure[1]):
        plt.plot(band_structure[1][i], band_structure[2][i], color='r')

        # means: bands[1] = q_distance, _band[2] = frequency, _bands[3] = eigenvectors, _band[4] = group_velocities
        
        band_distances.append(band_structure[1][i])
        frequency.append(band_structure[2][i])
        eigenvectors.append(band_structure[3][i])
        group_velocities.append(band_structure[4][i])

    if save_band_structure_file:
        band_distances = np.reshape(band_distances, (np.size(band_distances, 0) * np.size(band_distances, 1), -1))
        frequency_1 = np.reshape(frequency, (np.size(frequency, 0) * np.size(frequency, 1), -1))

        np.savetxt('band_structure.txt', np.column_stack((band_distances, frequency_1)), delimiter='   ')
        print("********* Save band-structure file successfully *********")

    # ************* if save_ave_gv_file = True, save_gv_file must be True **************
    if save_ave_gv_file:
        save_gv_file = True

    if save_gv_file:
        frequency_2 = np.reshape(frequency, (-1, 1))
        group_velocities = np.reshape(group_velocities, (-1, 3))

        total_group_velocities = []
        for i in range(np.size(group_velocities, 0)):
            group_velocity = np.sqrt((((group_velocities[i][0]) * 100) ** 2 + ((group_velocities[i][1]) * 100) ** 2 + (
                        (group_velocities[i][2]) * 100) ** 2))
            total_group_velocities.append(group_velocity)

        # Group velocity is in the unit of m/s
        np.savetxt('group_velocities.txt', np.column_stack((frequency_2, total_group_velocities)), delimiter='   ')
        print("********* Save group-velocities file successfully *********")

        if save_ave_gv_file:
            # ************** Use the API of more_itertools **************
            from more_itertools import chunked
            sort_freq = np.sort(frequency_2, axis=0)
            Gv_sort_index = np.argsort(frequency_2, axis=0)
            sort_group_velocities = np.array(total_group_velocities)[Gv_sort_index]

            ave_interval = np.size(frequency, 0) * np.size(frequency, 2)

            # Take an average every 'ave_interval' frequencies
            ave_group_velocities = [sum(x) / len(x) for x in chunked(sort_group_velocities, ave_interval)]
            ave_frequency = [sum(x) / len(x) for x in chunked(sort_freq, ave_interval)]

            # Group velocity is in the unit of m/s
            np.savetxt('group_ave_velocities.txt', np.column_stack((ave_frequency, ave_group_velocities)),
                       delimiter='   ')

            print("********* Save average group-velocities(frequency-dependent) file successfully *********")

    # plt.axes().get_xaxis().set_ticks([])
    plt.ylabel('Frequency (THz)', fontsize='x-large')
    plt.xlabel('Wave vector', fontsize='x-large')
    plt.xlim([0, band_structure[1][-1][-1]])
    plt.ylim([0, 20])

    plt.axhline(y=0, color='k', ls='dashed')
    plt.suptitle('Phonon dispersion', fontsize='x-large')

    def replace_list(text_string):
        substitutions = {'Gamma': r'$\Gamma$'}

        for item in substitutions.items():
            text_string = text_string.replace(item[0], item[1])

        return text_string

    if 'labels' in bands_and_labels and bands_and_labels['labels'] is not None:

        # plt.rcParams.update({'mathtext.default': 'regular'})
        labels = bands_and_labels['labels']

        labels_e = []
        x_labels = []

        for i, freq in enumerate(band_structure[1]):
            if labels[i][0] == labels[i - 1][1]:
                labels_e.append(replace_list(labels[i][0]))
            else:
                labels_e.append(
                    replace_list(labels[i - 1][1]) + '/' + replace_list(labels[i][0]))
            x_labels.append(band_structure[1][i][0])

        x_labels.append(band_structure[1][-1][-1])
        labels_e.append(replace_list(labels[-1][1]))
        labels_e[0] = replace_list(labels[0][0])

        ## ********** Output qx_axis_data for plot figures ***********
        np.savetxt('qx_axis_data.txt', np.column_stack(x_labels), delimiter='\n')

        plt.xticks(x_labels, labels_e, rotation='horizontal', fontsize='x-large')

    plt.savefig('Phonon_dispersion.png', format='png', dpi=500, bbox_inches='tight')

    if show_dispersion:
        plt.show()

def get_calculator(structure_file, NEP_file, band_path, band_label, repeat_cell=[1, 1, 1]):

    # Get model from ACE
    structure = read(structure_file)
    
    calculator = CPUNEP(NEP_file)
    print("\n********* Using NEP potential **********")

    structure.calc = calculator
 
    f_max = 0.00001
    
    # First relax
    relaxed_structure = relax(structure=structure, eps=f_max)

    # Calculate phonon information using Phonopy interface
    band_structure, bands_and_labels = phonon_calculator(relaxed_structure,
                                                         calculator, repeat_cell, band_path, band_label)

    plot_band_structure(band_structure, bands_and_labels)
    

if __name__ == "__main__":

    # Construct model
    
    structure_file = 'POSCAR_DNWs'
    
    repeat_supercell = [1, 4, 1]

    band_path = [[0.0, 0.0, 0.0],
                 [0.0, 0.5, 0.0]]
                 
    band_label = [('Gamma', 'Y')]

    # Nep file
    nep_potential_file = 'C_2024_NEP4.txt'
    
    get_calculator(structure_file, nep_potential_file, band_path, band_label, repeat_supercell)
    
    print('***************** Calculation ALL Done !!! *****************')
