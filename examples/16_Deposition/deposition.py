# Build the to-be-deposited atoms (clusters)
def build_dep_atom(dep_atom):
    # Initialization calculation parameters
    a_type = dep_atom.get('a_type')      # Atomic species
    position = dep_atom.get('position')  # XYZ coordinates
    m = dep_atom.get('mass')             # Atomic masses
    name = dep_atom.get('name')          # Output file name
    exyz = dep_atom.get('exyz',0)        # Create xyz file for visualization
    # Load parameters and write files
    with open('%s' % name,'w') as file:
        if exyz == 1:
            list_tem = ['0\n','triclinic=F pbc="F F F" Lattice="10 0 0 0 10 0 0 0 10" Properties=species:S:1:pos:R:3:mass:R:1\n']
        if type(m) != float: # Polyatomic type
            listn = [] ; listm = []
            for i in m:
                tem = i.split() ; listn.append(tem[0]) ; listm.append(tem[1])
        for i in range(len(position)): # Atomic positions
            if type(m) != float:
                for j in range(len(listn)):
                    if a_type[i] in listn[j]:
                        m = listm[j] 
            tem = '%s %s %s %s %s\n' % (a_type[i],position[i][0],position[i][1],position[i][2],m)
            if exyz == 1:
                list_tem.append(tem)
            file.writelines(tem)
    # Create exyz file
    if exyz == 1:
        list_tem[0] = '%i\n' % (len(list_tem)-2)
        with open('%s.xyz' % name,'w') as file:
            for line in list_tem:
                file.writelines(line)

# Build the deposition substrate (exyz/ASE formats supported)
def build_dep_sub(dep_sub):
    import random
    # Initialization calculation parameters
    a_type = dep_sub.get('a_type',None)                      # Atomic species
    position = dep_sub.get('position',None)                  # XYZ coordinates
    m = dep_sub.get('mass',None)                             # Atomic masses
    name = dep_sub.get('name', 'model.xyz')                  # Output file name
    Cell = dep_sub.get('Cell', [[1,0,0],[0,1,0],[0,0,1]])    # Simulation-cell lattice 
    z = dep_sub.get('z', Cell[2][2])                         # Cell length along z 
    vacuum = dep_sub.get('vacuum', 5)                        # Vacuum thickness added at the bottom of z 
    fix = dep_sub.get('fix', 10)                             # Thickness of the fixed layer  
    heat_g = dep_sub.get('heat_g', 0.5)                      # Fraction of atoms in the hot-bath group 
    im_file = dep_sub.get('im_file',None)                    # Input file name 
    defect_p = dep_sub.get('defect_p', 0.2)                  # Surface-layer defect ratio
    defect_l = dep_sub.get('defect_l', 5)                    # Thickness of the surface layer  
    # If no input file is provided, build the substrate from the supplied list.
    if im_file == None:
        list_tem = ['0\n']
        if type(m) != float:
            listn = [] ; listm = []
            for i in m:
                tem = i.split() ; listn.append(tem[0]) ; listm.append(tem[1])
        # Set the x, y, z bounds
        z_max = max(map(lambda x: x[2], position)) ; z_min = min(map(lambda x: x[2], position))
        y_max = max(map(lambda x: x[1], position)) ; y_min = min(map(lambda x: x[1], position))
        x_max = max(map(lambda x: x[0], position)) ; x_min = min(map(lambda x: x[0], position))
        tem = 'triclinic=F pbc="T T F" Lattice="%.6f 0 0 0 %.6f 0 0 0 %.6f" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:1\n' % (Cell[0][0],Cell[1][1],z)
        list_tem.append(tem)
        # Group select
        for i in range(len(position)):
            if type(m) != float:
                for j in range(len(listn)):
                    if a_type[i] in listn[j]:
                        m = listm[j]
            if position[i][2]-z_min <= fix:
                group = 0
            elif (position[i][1]-y_min)/y_max <= heat_g:
                group = 1
            else:
                group = 2
            if position[i][2] < z_max - defect_l:
                position[i][2] = position[i][2] + vacuum
                tem = '%s %.6f %.6f %.6f %s 0 0 0 %s\n' % (a_type[i],position[i][0],position[i][1],position[i][2],m,group) ; list_tem.append(tem)
            else:
                # Set defects
                choice = random.choices([0, 1],weights = [defect_p,1 - defect_p])[0]
                if choice == 1:
                    position[i][2] = position[i][2] + vacuum
                    tem = '%s %.6f %.6f %.6f %s 0 0 0 %s\n' % (a_type[i],position[i][0],position[i][1],position[i][2],m,group) ; list_tem.append(tem)
            list_tem[0] = '%i\n' % (len(list_tem)-2)
    else:
        with open('%s' % im_file,'r') as file:
            lines = file.readlines()
        a = lines[1].split("Lattice=") ; b = a[1].split("\"") ; c = b[1].split(" ")
        x = float(c[0]) ; y = float(c[4]) ; z = float(c[8])
        list_tem = ['0\n'] ; listx = [] ; listy = [] ; listz = []
        list_tem.append('triclinic=F pbc="T T F" Lattice="%.6f 0 0 0 %.6f 0 0 0 %.6f" Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:1\n' % (x,y,z))
        # Read files and set x, y, z bounds
        for line in lines[2:]:
            tem = line.split() ; listx.append(float(tem[1])) ; listy.append(float(tem[2])) ; listz.append(float(tem[3]))
        z_max = max(listz) ; z_min = min(listz) ; y_max = max(listy) ; y_min = min(listy) ; x_max = max(listx) ; x_min = min(listx)
        # Group select
        for i in range(len(listx)):
            if listz[i]-z_min <= fix:
                group = 0
            elif (listy[i]-y_min)/y_max <= heat_g:
                group = 1
            else:
                group = 2
            if listz[i] < z_max - defect_l:
                line = lines[i+2].split() ; line[3] = float(line[3]) + vacuum
                tem = '%s %s %s %s %s 0 0 0 %s\n' % (line[0],line[1],line[2],line[3],line[4],group) ; list_tem.append(tem)
            else:
                # Set defects
                choice = random.choices([0, 1],weights = [defect_p,1 - defect_p])[0]
                if choice == 1:
                    line = lines[i+2].split() ; line[3] = float(line[3]) + vacuum
                    tem = '%s %s %s %s %s 0 0 0 %s\n' % (line[0],line[1],line[2],line[3],line[4],group) ; list_tem.append(tem)
        list_tem[0] = '%i\n' % (len(list_tem)-2)
    # write file
    with open('%s' % name,'w') as file:
        for line in list_tem:
            file.writelines(line)

# Initialization calculation parameters
param = {} # Set global dictionary
def para(params):
    # Read simulation box info and back up the model.xyz as original.xyz
    with open('./model.xyz','r') as file:
        lines = file.readlines()
    with open('./restart.xyz','w') as file:
        for line in lines:
            file.writelines(line)
    with open('./original.xyz','w') as file:
        for line in lines:
            file.writelines(line)
    param['atom_num'] = int(lines[0])                                           # The number of atoms (Auto)
    a = lines[1].split("Lattice=") ; b = a[1].split("\"") ; c = b[1].split(" ")
    param['x'] = float(c[0]) ; param['y'] = float(c[4]) ; param['z'] = float(c[8])
    print('Cell size of x-axis: %f' % param['x'])                               # Cell size of x-axis (Auto)
    print('Cell size of y-axis: %f' % param['y'])                               # Cell size of y-axis (Auto)
    # Extract parameters from the parameter dictionary, using default values where not specified
    param['path'] = params.get('path')                                          # Path for GPUMD
    param['cycle'] = params.get('cycle', 5)                                     # Number of deposition cycles
    param['species'] = params.get('species', 1)                                 # Number of deposition atoms (clusters) species
    param['number'] = params.get('number', 15)                                  # Total number ofatoms (clusters) selected per deposition
    param['vxy'] = params.get('vxy', [0, 0])                                    # Initial velocity of deposition atoms in the x, y-axis
    param['vz'] = params.get('vz', 0.005)                                       # Initial velocity of deposition atoms in the z-axis 
                                                                                # 1. Fixed velocity 2. Random velocity range [0.005, 0.01, 1] (vz[2]=1) 3. Gauss distribution [0.005, 0.001, 0]
    param['cutoff'] = params.get('cutoff', 7)                                   # Minimum distance between two atoms (clusters)
    param['h_range'] = params.get('h_range', [param['z']-10, param['z']])       # z-bounds of the insertion slab
    param['h_c'] = params.get('h_c')                                            # The value of the deposition region in the z-axis (auto-updated each cycle)
    param['x_range'] = params.get('x_range', [0, param['x']])                   # x-bounds of the insertion region
    param['y_range'] = params.get('y_range', [0, param['y']])                   # y-bounds of the insertion region
    param['h_cutoff'] = params.get('h_cutoff',param['h_range'][1])              # z-axis threshold for removing undeposited atoms (atoms with z > h_cutoff will be removed)
    param['group'] = params.get('group', 3)                                     # Group of deposition atoms (corresponding to group in GPUMD)
    param['sto_ratio'] = params.get('sto_ratio')                                # Atomic ratio for multi-species deposition [[atom1, atom2], [2, 3]]
    param['sto_standard'] = params.get('sto_standard', 0.5)
    param['prob'] = params.get('prob', 0)                                       # Atom-addition mode:
                                                                                # 0: uniform deposition  
                                                                                # 1: probability-weighted deposition with fixed surface-layer thickness  
                                                                                # 2: probability-weighted deposition with dynamically calculated surface-layer thickness
    param['mash_xy'] = params.get('mash_xy', [10, 10])                          # Number of x, y-axis grids for probabilistic deposition (prob = 1, 2)
    param['delta_z'] = params.get('delta_z', 5)                                 # z-axis grid thickness for probabilistic deposition (prob = 1, 2)
    param['prob_h'] = params.get('prob_h', param['delta_z'])                    # Surface layer thickness set when adding atoms with prob = 1
    param['delta_z_cut'] = params.get('delta_z_cut', 0.1)                       # Defect ratio used to define the surface layer for probabilistic deposition (prob = 1, 2)

    # Output configuration parameters
    print('Total cycle : %s' % param['cycle'])
    print('Total kinds of clusters: %s' % param['species'])  
    print('Total clusters of every step: %s' % param['number'])  
    if type(param['vz']) == int:
        print('Deposition velocity: %s' % param['vz'])
    elif type(param['vz']) == list:
        if param['vz'][2] == 0:
            print('Deposition velocity follows Gaussian distribution N(%s, %s)' % (param['vz'][0],param['vz'][1]))
        elif param['vz'][2] == 1:
            print('Deposition velocity from %s to %s' % (param['vz'][0],param['vz'][1]))
    print('Minimum distance between two clusters: %s' % param['cutoff'])
    if param['h_c'] == None:
        print('Deposition region along z-axis: %s to %s' % (param['h_range'][0],param['h_range'][1]))
    else:
        print('Atoms will be placed %s - %s above the current top of the settled film' % (param['h_c'][0],param['h_c'][1]))
    print('Deposition region along x-axis: %s to %s' % (param['x_range'][0],param['x_range'][1]))
    print('Deposition region along y-axis: %s to %s\n' % (param['y_range'][0],param['y_range'][1]))
    if param['prob'] == 0:
        print('Atom clusters will be uniformly deposited\n')
    else:
       print('Atomic clusters tend to preferentially deposited into vacancies')
       print('Threshold of surface layer vacancy defect: %s' % (1-param['delta_z_cut']))
       if param['prob'] == 1:
        print('Thickness of surface layer: %s\n' % param['prob_h'])
    # For multi-species deposition
    if param['sto_ratio'] != None:
        param['list_c'] = multi_species()

# For multi-species deposition, count chemical stoichiometry ratio in clusters
def multi_species():
    # Parameters used by this function
    species = param['species'] ; sto_ratio = param['sto_ratio'] ; list_c = []
    for i in range(species):
        with open('./cluster/%s.txt' % i) as file:
            lines = file.readlines()
        list_tem = [sto_ratio[0],[0]*len(sto_ratio[0])]
        for line in lines:
            tem = line.split()
            for j in range(len(sto_ratio[0])):
                if tem[0] in sto_ratio[0][j]:
                    list_tem[1][j] = list_tem[1][j] + 1
        total_atom = sum(list_tem[1])
        for i in range(len(list_tem[1])):
            list_tem[1][i] = list_tem[1][i]/total_atom #[[atom1,...],[chemical stoichiometry]]
        list_c.append(list_tem) 
    return(list_c)

# Main loop procedure - core processing logic
def main_dep():
    import subprocess # 
    # Parameters used by this function
    cycle = param['cycle'] ; number = param['number'] ; path = param['path']
    # Initialize count.txt for storing per-cycle atom deposition counts
    with open('count.txt','w') as file:
        file.writelines('#Atoms saved in every cycle:\n')
    #cycle
    for i in range(cycle):
        print('begin %s cycle' % (i+1))
        param['tem_cycle'] = i+1
        with open('./restart.xyz','r') as file:
            lines1 = file.readlines()
        add_atom()  # Prepare atoms (clusters) required for this deposition step
        with open('./cluster/add.txt','r') as file:
            lines2 = file.readlines()
        lines1[0] = '%i\n' % (len(lines2) + int(lines1[0])) # Total atoms: add.txt count + restart.xyz count
        with open('./model.xyz','w') as file:  # Merge restart.xyz and add.txt into a single structure
            for line in lines1:
                file.writelines(line)
            for line in lines2:
                file.writelines(line)
        with open('./output', 'w') as file:
            subprocess.run([path], stdout=file, stderr=subprocess.STDOUT)
        zdir()     # Remove atoms that have not yet deposited
    print('Calculation finished')

# Initialize positions and velocities of atoms introduced in this deposition cycle
def add_atom():
    import random
    # Parameters used by this function
    x_max = param['x_range'][1] ; y_max = param['y_range'][1] ; x_min = param['x_range'][0] ; y_min = param['y_range'][0]
    vz = param['vz'] ; vx = param['vxy'][0] ; vy = param['vxy'][1] ; h_min = param['h_range'][0] ; h_max = param['h_range'][1] ; h_c = param['h_c']
    cutoff = param['cutoff'] ; num = param['number'] ; group = param['group']
    x0 = param['x'] ; y0 = param['y'] ; species = param['species'] ; prob = param['prob'] ; sto_ratio = param['sto_ratio'] ; sto_standard = param['sto_standard']
    list1 = [0] ; list2 = [0] ; list3 = [0] ; listadd = []
    # Set random-number generation strategy of deposition
    if prob != 0:
        list_m = z_pos_uni()
    # Set random-number generation strategy of chemical stoichiometry ratio
    if sto_ratio != None:
        import numpy as np
        list_s = [] # Set parameters
        for i in range(species):
            list_s.append(i)
        list_c = param['list_c'] ; atom_species = len(sto_ratio[1]) ; sto = [0]*atom_species
        with open('restart.xyz', 'r') as file: # Calculate chemical stoichiometry ratio of original file
            lines = file.readlines()
            for line in lines[2:]:
                tem = line.split()
                for num0 in range(atom_species):
                    if tem[0] in sto_ratio[0][num0]:
                        sto[num0] = sto[num0] + 1
        prob_a = 0 ; tot_atom = sum(sto)*(1+sto_standard) # Assuming the total number of atoms (defult 1.5 times)
        while prob_a == 0: # Compute and print probabilities, ensuring each value stays within [0, 1].
            list_de_atom = []
            for i in range(atom_species):
                sto_ratio_1 = sto_ratio[1][i]/sum(sto_ratio[1]) # Normalized stoichiometric ratio
                tem = (tot_atom*sto_ratio_1 - sto[i])/(sum(sto)*sto_standard) # (%totoal_target_atom-%existing_target_atom)/%added_total_atom
                list_de_atom.append(tem) # The final normalized stoichiometric ratio
            list_atom_prob = [] ; total_prob = 1 # Probability of the last N functional groups; Total unallocated probability
            for i in range(species-atom_species):
                atom_prob = random.uniform(0, total_prob) # Set the probability for the first N-%elemental clusters
                total_prob = total_prob - atom_prob
                # Adjust the appearance probability of the remaining required elements
                for j in range(atom_species): 
                    list_de_atom[j] = list_de_atom[j] - list_c[i][1][j]*atom_prob
                list_atom_prob.append(atom_prob)
            A = [] ; B = []
#解方程
#           A = [[]]; B = [total_prob]
#           for i in range(atom_species): # 团簇选取概率和为1
#               A[0].append(1) ; A.append([]) ; B.append(list_de_atom[i])
# 这三行代码为测试时使用，由于原子基团选取的概率和为一，此处已知数比未知量大1，即可以列出n-1个方程组。若每组方程组的解均相同，则代码正确，经此方式验证二元结构无勿
            for i in range(atom_species):
                B.append(list_de_atom[i]) ; A.append([])
                for j in range(atom_species):
                    A[i].append(list_c[species-atom_species+j][1][i])
            x = np.linalg.solve(A,B)
            if min(x) >= 0 and max(x) <= 1:
                prob_a = 1
                for i in x:
                    list_atom_prob.append(i)
        sele = random.choices(list_s ,weights = list_atom_prob)[0]
    # Set position and velocity for each functional clusters
    for i in range(num):
        a = 0 ; b = 0
        while a == 0:
            a = 1
            if param['h_c'] == None:
                s3 = random.uniform(h_min,h_max)
            else:
                z_pos_uni() # Dynamically update the deposition-zone height
                s3 = random.uniform(z_pos_uni()[4] + h_c[0],z_pos_uni()[4] + h_c[1])
            if prob == 0:
                s1 = random.uniform(x_min,x_max) ; s2 = random.uniform(y_min,y_max)
            else:
                tem = random.choices(list_m[0] ,weights = list_m[1])[0] # Weighted selection of the xy deposition zone
                s1 = random.uniform(tem[0],tem[0]+list_m[2]) ; s2 = random.uniform(tem[1],tem[1]+list_m[3])
            # Periodicity requires additional processing of boundary atoms
            if s1 < cutoff :
                s11 = x0 + s1 # s11,s12: impact of periodic
            elif s1 > x0 - cutoff:
                s11 = s1 - x0
            else:
                s11 = s1
            if s2 < cutoff :
                s21 = y0 + s2
            elif s2 > y0 - cutoff:
                s21 = s2 - y0
            else:
                s21 = s2
            # Compute inter-cluster distance
            for j in range(len(list1)):
                x1 = (s1 - list1[j])**2 ; y1 = (s2 - list2[j])**2 ; z1 = (s3 - list3[j])**2
                x2 = (s11 - list1[j])**2 ; y2 = (s21 - list2[j])**2                
                if x1 + y1 + z1 <= cutoff**2 or x2 + y2 + z1 <= cutoff**2:
                    a = 0 ; b = b + 1
                    if b >= 1000:
                        print('Too many clusters were added in single cycle, please reduce the \'%s\' value !' % num)
                        exit()
        if s1 > x0:
            s1 = s1 - x0
        if s2 > y0:
            s2 = s2 - y0 # Constrain all atoms within the simulation box
        if sto_ratio == None:
            sele = random.randint(0,species-1)
        else:
            sele = random.choices(list_s ,weights = list_atom_prob)[0]                
        list1.append(s1) ; list2.append(s2) ; list3.append(s3)
        # Set velocity
        if type(vz) == list:
            if vz[2] == 1:
                vz0 = random.uniform(vz[0],vz[1])
            elif vz[2] == 0:
                vz0 = random.gauss(vz[0],vz[1])
        else:
            vz0 = vz
        with open('./cluster/%s.txt' % sele,'r') as file:
            lines = file.readlines()
        for line in lines:
            a = line.split()
            x = float(a[1]) + s1 ; y = float(a[2]) + s2 ; z = float(a[3]) + s3
            if x >= x0:
                x = x - x0
            if y >= y0:
                y = y -y0
            listadd.append('%s %.4f %.4f %.4f %s %.4f %.4f %.4f %s\n' % (a[0],x,y,z,a[4],vx,vy,-vz0,group))
    with open('./cluster/add.txt','w') as file:
        for line in listadd:
            file.writelines(line)

# For non-uniform deposition, identify the target layer position
def z_pos_uni():
    # Parameters used by this function
    param['tem_cycle'] = 0
    listmesh = [] ; listmesh_d = [] ; mash_x = param['mash_xy'][0] ; mash_y = param['mash_xy'][1] ; delta_z = param['delta_z'] ; delta_z_cut = param['delta_z_cut'] ; h_max = param['h_range'][1] 
    x_max = param['x_range'][1] ; y_max = param['y_range'][1] ; x_min = param['x_range'][0] ; y_min = param['y_range'][0] ; h_cutoff = param['h_cutoff'] ; prob = param['prob'] ; prob_h = param['prob_h']
    dx = (x_max - x_min)/mash_x ; dy = (y_max - y_min)/mash_y # Define minimum grid spacing in x and y directions
    for i in range(mash_x):
        for j in range(mash_y):
            listmesh.append(0) ; listmesh_d.append([dx*i+x_min,dy*j+y_min])
    with open('model.xyz','r') as file:
        lines = file.readlines() ; listz = [] ; listxy = [[],[]] # listz: stores atomic z-coordinates; listxy: stores atomic (x,y) coordinates
        for line in lines[2:]:
            b0 = line.split()
            listz.append(float(b0[3])) ; listxy[0].append(float(b0[1])) ; listxy[1].append(float(b0[2]))
    min_z = min(listz) # min_z: minimum z-coordinate in the structure
    numz_t = int((h_cutoff-min_z)/delta_z) ; listz_num = [0]*(numz_t+1)
    for z_coord in listz:
        if z_coord <= h_cutoff:
            tem = int((z_coord-min_z)/delta_z) ; listz_num[tem] = listz_num[tem] + 1
    norma = listz_num[0]
    for i in range(len(listz_num)):
        listz_num[i] = listz_num[i]/norma
    uni = 0
    for i in range(len(listz_num)):
        if listz_num[i] < delta_z_cut:
            max_z = min_z + delta_z*(i+1) ; uni = 1
            break
    if uni == 0: # Donot hits the uniform-deposition threshold
        tem_cycle = param['tem_cycle'] ; prob = 0
    for i in range(len(listz)):
        if prob == 1:
            if listz[i] >= max_z - prob_h and listz[i] <= max_z:
                numx = int((listxy[0][i] - x_min)/dx) ; numy = int((listxy[1][i] - y_min)/dy)
                temxy = numx*mash_y + numy ; listmesh[temxy] = listmesh[temxy] + 1
        elif prob == 2 or prob == 0:
            if listz[i] >= max_z - delta_z and listz[i] <= h_max:
                numx = int((listxy[0][i] - x_min)/dx) ; numy = int((listxy[1][i] - y_min)/dy)
                temxy = numx*mash_y + numy ; listmesh[temxy] = listmesh[temxy] + 1
    max_atom = max(listmesh) ; add_atom = max_atom*len(listmesh) - sum(listmesh)
    if add_atom > 0:
        for i in range(len(listmesh)):
            listmesh[i] = (max_atom - listmesh[i])/add_atom
    else:
        for i in range(len(listmesh)):
            listmesh[i] = 1/len(listmesh)
            tem_cycle = param['tem_cycle']
    return [listmesh_d, listmesh, dx, dy, max_z]

# Remove flying atoms, update ‘restart.xyz’, and log the deposition statistics.
def zdir():
    # Parameters used by this function
    number = param['atom_num'] ; h_c = param['h_cutoff']
    with open('./restart.xyz','r') as file:
        lines = file.readlines()
        list1 = ['0\n'] ; list1.append(lines[1])
        for line in lines[2:]:
            a = line.split() ; z = float(a[3])
            if z < h_c:
                list1.append(line)
    list1[0] = '%i\n' % (len(list1)-2)
    with open('./restart.xyz','w') as file:
        for line in list1:
            file.writelines(line)
    new = int(list1[0]) ; raw = int(lines[0]) ; dele = raw - new
    list2 = ['Existing atomic number = %d\nDeleted atomic number = %d\n' % (new,dele)]
    new_num = new - number
    with open('./cluster/add.txt','r') as file:
        lines = file.readlines() ; num = len(lines)
    added = num - dele
    print('Number of new atoms added in this cycle : %s' % num)
    print('Number of new atoms saved in this cycle : %s' % added)
    print('Number of new atoms saved in total : %s' % new_num)
    with open('count.txt','a') as file:
        file.writelines(list2)

# Compute layer-resolved density along z-axis for the deposited structure
def density(dz, z_min=None, z_max=None, name_i = 'restart.xyz', name_o='density.txt'):
    with open('%s' % name_i,'r') as file:
        lines = file.readlines()
    a = lines[1].split("Lattice=") ; b = a[1].split("\"") ; c = b[1].split(" ")
    x_max = float(c[0]) ; y_max = float(c[4]) ; x_min = 0 ; y_min = 0
    if z_min == None:
        z_min = param['h_range'][0]
    if z_max == None:
        z_max = param['h_range'][1]
        x_max = param['x_range'][1] ; y_max = param['y_range'][1] ; x_min = param['x_range'][0] ; y_min = param['y_range'][0]
    listz = [] ; listm = [] ; Na = 6.02*10**23 # 1 Å³ = 10⁻²⁴ cm³; Na: Avogadro's number
    z0 = (z_max-z_min)/dz ; V = (x_max-x_min)*(y_max-y_min)*z0
    for i in range(dz):
        listz.append(0) ; listm.append([])
    for line in lines[2:]:
        b = line.split() ; z = float(b[3])
        if z >= z_min and z <= z_max:
            m = float(b[4]) ; tem = int((z-z_min)/z0)
            listz[tem] = listz[tem] + 1 ; listm[tem].append(m)
    with open('%s' % name_o,'w') as file:
        for i in range(dz):
            tem_num = listz[i]/V ; tem_mass = sum(listm[i])*10**24/V/Na # density1: atoms/Å³; density2: g/cm³
            tem = '%s    %.4f    %.4f\n' % (z_min+z0*(i+1),tem_num,tem_mass)
            file.writelines(tem)
def atom_num():
    with open('count.txt','r') as file:
        lines = file.readlines() ; a = 0 ; listtem = [[],[],[]]
        for line in lines[1:]:
            a = a + 1
            b = line.split('=')
            listtem[a%2+1].append(b[1])
            if a%2 == 0:
                listtem[0].append(int(a/2))
    with open('del_atom.txt','w') as file:
        for i in range(len(listtem[0])):
            file.writelines('%s %s' % (listtem[0][i],listtem[1][i]))
    with open('tot_atom.txt','w') as file:
        for i in range(len(listtem[0])):
            file.writelines('%s %s' % (listtem[0][i],listtem[2][i]))