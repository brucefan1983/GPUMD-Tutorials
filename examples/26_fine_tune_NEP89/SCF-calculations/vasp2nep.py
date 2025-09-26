# -*- coding: utf-8 -*-
"""
Author: Liang Ting, email: liangting.zj@gamil.com
2022/12/4 17:10:05
Usage:
  python vasp2nep.py

  From ugly VASP OUTCAR format to NEP dataset for train GPUMD dataset (including train.in and test.in-- if one dont need, just set it equal 0)
  Requirement: Numpy, random
  Ref (get data from OUTCAR):
  dpdata: https://github.com/deepmodeling/dpdata/blob/master/dpdata/vasp/outcar.py
  
"""

import os
import warnings
import numpy as np
import random

class vasp2nep(object):

    def __init__(self, criterion, test_data_frames=None,
                       path=None, config_type=True, include_VDW=True, if_virial=True, if_weight=True, weight_value=1.0):

        # File Properties
        self.filename_flag = criterion
        self.path = path
        self.dataset_config_type = config_type
        self.include_virials = if_virial
        self.include_weight = if_weight
        self.weight_value = weight_value

        # Outcar properties
        self.outcar_name = None
        self.test_data_frames = test_data_frames
        self.include_VDW = include_VDW

    def get_outcar_file_name(self):
        outcar_file_name = []
        num_outcar_file = 0
        dirs_file_name = []

        for root, dirs, file_lists in os.walk(self.path):
            for file_list in file_lists:
                if self.filename_flag in file_list:
                    num_outcar_file += 1
                    dirs_file_name.append(os.path.basename(root))
                    outcar_file_name.append(os.path.join(root, file_list))

        print('\n************* Existing TOTAL {0} {1} files to construct NEP-4 dataset *************\n'.format(num_outcar_file, self.filename_flag))
        self.total_frame = num_outcar_file
        self.outcar_name = np.array(outcar_file_name)
        self.dirs_file_name = np.array(dirs_file_name)    # For output some information to screen, maybe don't need

    def system_info(self, lines, type_idx_zero=True):  # Get total atom numbers and atom types (names)
        self.atom_nums = None
        self.atom_names = []
        self.atom_types = []

        for ii in lines:
            if 'TITEL' in ii:
                # get atom names from POTCAR info, tested only for PAW_PBE ...
                _ii = ii.split()[3]
                if '_' in _ii:
                    # for case like : TITEL  = PAW_PBE Sn_d 06Sep2000
                    self.atom_names.append(_ii.split('_')[0])
                else:
                    self.atom_names.append(_ii)
            if 'ions per type' in ii:
                atom_nums = [int(s) for s in ii.split()[4:]]
                if self.atom_nums is None:
                    self.atom_nums = atom_nums
                else:
                    assert (self.atom_nums == atom_nums), "in consistent numb atoms in OUTCAR"

        self.atom_names = self.atom_names[:len(self.atom_nums)]
        for idx, ii in enumerate(self.atom_nums):
            for jj in range(ii):
                if type_idx_zero:
                    self.atom_types.append(idx)         # Index from zero
                else:
                    self.atom_types.append(idx + 1)

        self.total_atom_nums = sum(self.atom_nums)
        # Change data type to ndarray for indexing
        self.atom_names = np.array(self.atom_names)
        self.atom_types = np.array(self.atom_types)
        self.atom_type_lists = self.atom_names[self.atom_types]

    def write_2nep_every_frame(self, file_name, data_train):

        fd = open(file_name, mode='w')
        fd.write(data_train)                       # comment line and others type x y z fx fy fz
        fd.close()

    def output2nep(self, train_filename='train.xyz', test_filename='test.xyz'):
        # Get data from OUTCAR
        # select different searching tokens
        converge_flag = 'The electronic self-consistency was not achieved in'
        force_token = 'TOTAL-FORCE (eV/Angst)'
        energy_token = 'free  energy   TOTEN'
        energy_index = 4

        virial_token = 'FORCE on cell =-STRESS'          # virial flag
        virial_index = 13
        if self.include_VDW:
            virial_index = virial_index + 1              # For OUTCAR include VDW (MBD)

        cell_token = 'VOLUME and BASIS-vectors are now'
        cell_index = 5

        ## Generate random numbers (seed)
        if self.test_data_frames:
            print('*********************** {} data will be output ********************\n'.format(test_filename))
            random.seed(1000)        # For check the history
            # one can start from 2, since I don't want the first two frame including into the test data, one can change it
            index = np.array(random.sample(range(1, self.total_frame), self.test_data_frames))
            # If not test_data_frames, there is no need to create these variable
            data_test = ""

        # Get data from every frames
        data_train = ""
        num_not_achieved = 0

        for i, name in enumerate(self.outcar_name):
            # Got files to loop
            fd = open(name, "r")
            lines = fd.readlines()
            fd.close()
            print('***** You are Processing {} files *****'.format(self.outcar_name[i]))
            
            # Call function and get the number of atoms (property)
            self.system_info(lines)
            
            # New parameters
            coord_and_force = []
            cell = []
            outcar_data = ""
            
            self.cell = []
            self.virial = []
            self.coord_and_force = []
            self.energy = []
            
            # Check for convergence flag
            not_converge_found = False

            # Get some important parameters from outcar
            for idx, context in enumerate(lines):
                # SCF not achieved
                if converge_flag in context:
                   warnings.warn('**** SCF was not achieved in the given number of steps (NELM) in {} file, check it !!! ****\n'.format(self.outcar_name[i]))
                   not_converge_found = True
                   break
                   
                # Get cell
                elif cell_token in context:
                    for dd in range(3):
                        tmp_l = lines[idx + cell_index + dd]
                        try:
                           cell.append([np.float64(ss) for ss in tmp_l.replace('-', ' -').split()[0:3]])
                        except:
                        	 cell.append([0.000000000,  0.000000000, 100.000000000])
                        	 print("cell of Z direction is convert to 100, due to the VASP output bug")
                        	                         	 
                    self.cell = np.array(cell)
                    #print(self.cell)
                    if not len(self.atom_nums):
                        assert(self.atom_nums == []), "**** Connot extract cell in this OUTCAR, impossible !! ****"

                # Get virial
                elif virial_token in context:
                    self.virial = np.array(np.float64(lines[idx + virial_index].split()[1:8]))
                    #print(self.virial)
                    if not len(self.virial):
                        assert(self.virial == []), "**** Connot found virial in this OUTCAR, maybe it contains VDW ****"

                # Get coord and forces
                elif force_token in context:
                    for jj in range(idx + 2, idx + 2 + self.total_atom_nums):
                        tmp_l = lines[jj]
                        info = [float(ss) for ss in tmp_l.split()]
                        coord_and_force.append(info[0:6])
                    self.coord_and_force = np.array(coord_and_force)
                    #print(self.coord_and_force)
                # Get energy
                elif energy_token in context:
                    self.energy = np.float64(context.strip().split()[energy_index])
            
            if not_converge_found:
            	 num_not_achieved += 1
            	 continue  # Skip to the next iteration of the loop
                    
            # Nep version-4 and here also can be change according the atom type weight
            outcar_data += "{0}\n".format(self.total_atom_nums)

            outcar_data += "Lattice=\""
            # output cell
            for aa in range(3):
                if aa == 2:
                    outcar_data += '{0:10.9f} {1:10.9f} {2:10.9f}\" '.format(*self.cell[aa])  # cut the space
                else:
                    outcar_data += '{0:10.9f} {1:10.9f} {2:10.9f} '.format(*self.cell[aa])

            outcar_data += "Energy={0:10.9f} ".format(self.energy)
            outcar_data += "Properties=species:S:1:pos:R:3:force:R:3 "

            if self.include_virials:
                outcar_data += 'Virial=\"{0:10.8f} {1:10.8f} {2:10.8f} {3:10.8f} {4:10.8f} {5:10.8f} {6:10.8f} {7:10.8f} {8:10.8f}\" '.format(
                                                                                                self.virial[0], self.virial[3], self.virial[5],
                                                                                                self.virial[3], self.virial[1], self.virial[4],
                                                                                                self.virial[5], self.virial[4], self.virial[2])
            outcar_data += "pbc=\"T T T\""
            
            if self.dataset_config_type:
                outcar_data += " Config_type={}".format(self.dirs_file_name[i])
                
            if self.include_weight:
                outcar_data += " weight={}".format(self.weight_value)
            else:
                outcar_data += " weight={}".format(1.0)
                
            outcar_data += "\n"
             
            # output coord and forces
            for kk in range(self.total_atom_nums):
                outcar_data += '{0} {1:15.8f} {2:15.8f} {3:15.8f} {4:15.8f} {5:15.8f} {6:15.8f}\n'.\
                                                     format(self.atom_type_lists[kk], *self.coord_and_force[kk])

            ## Determined the train data or test data (random)
            if test_data_frames and (i in index):
                print('******************* Frame {} are write to {} *****************'.format(self.outcar_name[i], test_filename))
                data_test += outcar_data
            else:
                data_train += outcar_data

        # Output to test.xyz if need
        if test_data_frames:
            first_line_test = len(index)
            self.write_2nep_every_frame(test_filename, data_test)
            print('\n************************* Finally got {} frame in {} *********************'.format(first_line_test, test_filename))

        ## Output Train.xyz
        frames = self.total_frame - self.test_data_frames - num_not_achieved
        self.write_2nep_every_frame(train_filename, data_train)
        print('************************ Finally got {} frame in {} **********************\n'.format(frames, train_filename))

if __name__ == "__main__":

    criterion = 'OUTCAR'
    test_data_frames = 0
    
    out_file_train = 'train.xyz'
    out_file_test = 'test.xyz'

    if os.path.exists(out_file_train):
        os.remove(out_file_train)
        print("*************************** Remove the Old train.xyz ***************************")
    if os.path.exists(out_file_test):
        os.remove(out_file_test)
        print("**************************** Remove the Old test.xyz ***************************")

    path = '26_fine_tune_NEP89/SCF-calculations'
    include_virial = True
    include_VDW = True
    
    dataset = vasp2nep(criterion,
                       test_data_frames=test_data_frames,
                       path=path,
                       if_virial=include_virial,
                       include_VDW=include_VDW)

    dataset.get_outcar_file_name()
    dataset.output2nep()

    print('*************************************************************************************')
    print('*********************************** JOB ALL DONE !!! ***********************************')
    print('*************************************************************************************')
