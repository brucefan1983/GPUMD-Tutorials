# -*- coding: utf-8 -*-
import numpy as np
import math

def calc_angle(v1, v2):
    alpha = math.acos(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
    return alpha

def calc_elstic(strain_ij, strain_kl):
    s_ijkl = np.mean(strain_ij*strain_kl)-np.mean(strain_ij)*np.mean(strain_kl)
    return s_ijkl

def parse_thermo_out(file_path):
    # Define the headers based on your column mapping
    headers = [
        "T", "K", "U", "Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy", 
        "ax", "ay", "az", "bx", "by", "bz", "cx", "cy", "cz"
    ]
    
    # Initialize the dictionary with empty lists
    data_dict = {header: [] for header in headers}
    
    with open(file_path, 'r') as f:
        for line in f:
            # Clean the line and split by whitespace
            parts = line.split()
            
            # Skip empty lines or lines that don't match our column count
            if len(parts) != len(headers):
                continue
                
            try:
                # Convert values to float and append to lists
                for i, val in enumerate(parts):
                    data_dict[headers[i]].append(float(val))
            except ValueError:
                # This skips header rows if they contain text instead of numbers
                continue
                
    return data_dict

def output_elastic_gpumd(path, slice_num):
    #Obtain the strain tensor     
    thermo = parse_thermo_out("thermo.out")
    
    for keys in thermo:
        thermo[keys]=thermo[keys][1000:] #Discard the first 100 ps data
        
    Cij = np.zeros((slice_num,3))
    for i in range(slice_num): #Split to 10 slices
        # print(i)
        thermo_slice = dict()
        for keys in thermo:
            thermo_slice[keys]=thermo[keys][1000*i:1000*(i+1)]
        alpha = np.zeros(len(thermo_slice["ax"]))
        beta  = np.zeros_like(alpha)
        gamma = np.zeros_like(alpha)
        for j in range(len(thermo_slice["ax"])):
            va = [thermo_slice["ax"][j], thermo_slice["ay"][j], thermo_slice["az"][j]]
            vb = [thermo_slice["bx"][j], thermo_slice["by"][j], thermo_slice["bz"][j]]
            vc = [thermo_slice["cx"][j], thermo_slice["cy"][j], thermo_slice["cz"][j]]
            alpha[j] = calc_angle(vb, vc)
            beta[j]  = calc_angle(va, vc)
            gamma[j] = calc_angle(va, vb)
           
        strain = dict()
        strain["11"] = thermo_slice["ax"]/np.mean(thermo_slice["ax"])-1 #xx 
        strain["22"] = thermo_slice["by"]/np.mean(thermo_slice["by"])-1 #yy
        strain["33"] = thermo_slice["cz"]/np.mean(thermo_slice["cz"])-1 #zz
        strain["23"] = (alpha-math.pi/2)/2 #yz
        strain["13"] = (beta-math.pi/2)/2 #xz
        strain["12"] = (gamma-math.pi/2)/2 #xy
        
        #Calculate the cubic compliance tensor
        #S11=S1111=S2222=S3333,S12=S1122=S1133=S2233; S44=S2323=S1313=S1212
        V = np.mean(np.array(thermo["ax"])*np.array(thermo["by"])*np.array(thermo["cz"])) #volue, unit in angstrom^{3}
        T = 300 #temperature, unit in K
        kB = 1.38064852 #unit in e-23 J/K
        scale = 100/(T*kB)*V #unit in GPa^{-1}, scale=V/(KB*T)
        S1111 = scale*calc_elstic(strain["11"], strain["11"])
        S2222 = scale*calc_elstic(strain["22"], strain["22"])
        S3333 = scale*calc_elstic(strain["33"], strain["33"])
        S1122 = scale*calc_elstic(strain["11"], strain["22"])
        S1133 = scale*calc_elstic(strain["11"], strain["33"])
        S2233 = scale*calc_elstic(strain["22"], strain["33"])
        S2323 = scale*calc_elstic(strain["23"], strain["23"])
        S1313 = scale*calc_elstic(strain["13"], strain["13"])
        S1212  = scale*calc_elstic(strain["12"], strain["12"])
        
        # All the below value should be very close to zero
        S1123 = scale*calc_elstic(strain["11"], strain["23"])
        S1113 = scale*calc_elstic(strain["11"], strain["13"])
        S1112 = scale*calc_elstic(strain["11"], strain["12"])
        
        # print(S1123, S1113, S1112)
        # Convert Sijkl to  Spq 
        # NOTED that the Spq = Sijkl when p q = 1, 2, 3 BUT Spq = 4*Sijkl when p q = 4, 5, 6.
        S11 = (S1111+S2222+S3333)/3
        S12 = (S1122+S1133+S2233)/3
        S44 = 4*(S2323+S1313+S1212)/3
        Spq = np.array([[S11, S12, S12,   0,   0,   0], 
                        [S12, S11, S12,   0,   0,   0],
                        [S12, S12, S11,   0,   0,   0],
                        [0,     0,   0, S44,   0,   0],
                        [0,     0,   0,   0, S44,   0],
                        [0,     0,   0,   0,   0, S44]])
        
        # Convert Spq to Cpq and 
        Cpq = np.linalg.inv(Spq)
        C11 = Cpq[1,1]
        C12 = Cpq[1,2]
        C44 = Cpq[4,4]
        # print(np.array([C11, C12, C44]))
        Cij[i] = np.array([C11, C12, C44])
    return Cij

def calc_ste(array): # calculate the standard error
    ste = np.zeros(3)
    for i in range(array.shape[1]):
        ste[i] = math.sqrt(sum(abs(array[:,i] - array[:,i].mean())**2))/len(array[:,i])
    return ste
        

Cij_Si_gpumd_1ns = output_elastic_gpumd("./", 10)
print("C11, C12, C44 (unit in GPa) of Si estimated by gpumd with npt_scr for 1ns:")
avg = np.average(Cij_Si_gpumd_1ns, axis=0)
print(" ".join(f"{x:.1f}" for x in avg))
print("with standard error of: ")
ste = calc_ste(Cij_Si_gpumd_1ns)
print(" ".join(f"{x:.1f}" for x in ste))
