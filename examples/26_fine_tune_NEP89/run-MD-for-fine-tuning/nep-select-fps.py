# ============================================================
# Author: Ting Liang at 22/9/2025 20:21:17
# Email: liangting.zj@gmail.com
# Description: For farthest point sampling
# ============================================================

from pynep.calculate import NEP
from pynep.select import FarthestPointSample
from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def get_selected_frames(fxyz, calc, min_distance=0.05, mean=True, normalized=True, selected=True):

    dataset=read(fxyz, index=":", format='extxyz')

    total_num_of_atoms = 0
    descriptors = []
    
    for i, structrue in enumerate(dataset):

        if i % 100 == 0 and i > 0:
           print("****** Processing {0} frames. ******".format(i))
        if mean:	  
           descriptors.append(np.mean(calc.get_property('descriptor', structrue), axis=0))
           total_num_of_atoms += structrue.get_global_number_of_atoms()
        else:
           descriptors.append(calc.get_property('descriptor', structrue))
           
    print(f"Finished processing {len(dataset)} frames in total.\n")

    if not mean:
        # Concatenate all descriptors
        all_descriptors = np.concatenate(descriptors, axis=0)
        print("The total number of atoms in original dataset are {0}.".format(all_descriptors.shape[0]))
        print("The total number of descriptor components are {0}.".format(all_descriptors.shape[1]))
        if normalized:
            # see https://calorine.materialsmodeling.org/get_started/visualize_descriptor_space_with_pca.html
            pass
    else:
        all_descriptors = np.array(descriptors)
        
        print("The total number of atoms in original dataset are {0}.".format(total_num_of_atoms))
        print("The total number of descriptor components are {0}.".format(all_descriptors.shape[1]))

    pca = PCA(n_components=2)
    pc = pca.fit_transform(all_descriptors)

    p0 = pca.explained_variance_ratio_[0]
    p1 = pca.explained_variance_ratio_[1]
    
    print(f'Explained variance for component 0: {p0:.2f}')
    print(f'Explained variance for component 1: {p1:.2f}')

    if selected:
       sampler = FarthestPointSample(min_distance=min_distance)
       selected_i = sampler.select(all_descriptors, [])
       print(f"\nTotal selected frames: {len(selected_i)}\n") 
     
       if mean:
       	   
           write('selected.xyz', [dataset[i] for i in selected_i], format='extxyz')

       np.savetxt("selected_index.data", selected_i, fmt='%d')

       pca.fit(all_descriptors)
       selected_pc = pca.transform(np.array([all_descriptors[i] for i in selected_i]))

    plt.scatter(pc[:, 0], pc[:, 1], alpha=0.4, label="All data", color="orange")
    #print("The Pc shape is {}.".format(np.shape(pc)))

    if selected:
        plt.scatter(selected_pc[:, 0], selected_pc[:, 1], alpha=0.4, label="Selected data", color="green")

    plt.xlabel('PCA dimension 0, Var={0:2f}.'.format(p0))
    plt.ylabel('PCA dimension 1, Var={0:2f}.'.format(p1))

    plt.legend()
    plt.savefig('select.png')
    plt.show()

if __name__=='__main__':

    calc = NEP("nep_0409_virial.txt")
    min_distance = 0.00082
    get_selected_frames("movie.xyz", calc, min_distance)
