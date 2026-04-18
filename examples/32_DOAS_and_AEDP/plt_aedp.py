import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import os

def plot_energy_scatter_subplots():
    folders = ['600K', '1000K']
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.67), dpi=100)

    all_energies = []
    for folder in folders:
        file_path = os.path.join(folder, 'position_energy.out')
        data = np.loadtxt(file_path)
        all_energies.extend(data[:, 3])

    vmin, vmax = min(all_energies), max(all_energies)

    norm = Normalize(vmin=vmin, vmax=vmax)

    for i, folder in enumerate(folders):
        file_path = os.path.join(folder, 'position_energy.out')
        data = np.loadtxt(file_path)
        energy = data[:, 3]
        
        lower_threshold, upper_threshold = np.percentile(energy, [8, 92])
        
        middle_mask = (energy >= lower_threshold) & (energy <= upper_threshold)
        high_mask = energy > upper_threshold
        low_mask = energy < lower_threshold
        
        sorted_indices = np.concatenate([
            np.where(middle_mask)[0],
            np.where(high_mask)[0],
            np.where(low_mask)[0]
        ])
        
        x = data[sorted_indices, 0]
        y = data[sorted_indices, 1]
        z = data[sorted_indices, 2]
        energy = data[sorted_indices, 3]

        scatter = axes[i, 0].scatter(z, x, c=energy, cmap='viridis', norm=norm, s=50, alpha=0.4, edgecolors='none')
        axes[i, 0].set_xlabel(r'Lattice c ($\AA$)')
        axes[i, 0].set_ylabel(r'Lattice a ($\AA$)')

        scatter = axes[i, 1].scatter(x, y, c=energy, cmap='viridis', norm=norm, s=50, alpha=0.4, edgecolors='none')
        axes[i, 1].set_xlabel(r'Lattice a ($\AA$)')
        axes[i, 1].set_ylabel(r'Lattice b ($\AA$)')

        scatter = axes[i, 2].scatter(y, z, c=energy, cmap='viridis', norm=norm, s=50, alpha=0.4, edgecolors='none')
        axes[i, 2].set_xlabel(r'Lattice b ($\AA$)')
        axes[i, 2].set_ylabel(r'Lattice c ($\AA$)')

        if i == 0 or i == 1:  
            divider = make_axes_locatable(axes[i, 2])
            cax = divider.append_axes("right", size="3%", pad=0.1)
            sm = ScalarMappable(cmap='viridis', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label('Atomistic Energy (eV)')

    plt.tight_layout()
    #plt.show()
    plt.savefig('AEDP.png', dpi=400)

plot_energy_scatter_subplots()