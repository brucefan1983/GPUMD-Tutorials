import numpy as np

time_interval = 1 #unit:fs

mu = np.loadtxt("dipole.out")[:,1:]
    
# use only the first 10% of the data for the correlation function, as the rest is not statistically meaningful 
Nmax=len(mu)//10 

def calc_corr_fft(mu, Nmax):
    n = len(mu)
    # Fill in data to make the length satisfy FFT properties
    padded_length = n + Nmax - 1
    padded_mu = np.zeros((padded_length, 3), dtype=mu.dtype)
    padded_mu[:n] = mu

    fft_result = np.fft.fft(padded_mu, axis=0)
    power_spectrum = np.abs(fft_result) ** 2
    autocorr = np.fft.ifft(power_spectrum, axis=0)

    # Take the real autocorrelation part
    autocorr = autocorr[:Nmax, :].real
    norm_factors = np.array([n - lag for lag in range(Nmax)]).reshape(-1, 1)
    autocorr /= norm_factors

    # Merge the results of three dimensions
    autocorr_flat = np.sum(autocorr, axis=1)
    return autocorr_flat / autocorr_flat[0]

def compute_power_spectrum(acf,Nmax):
    ir = np.zeros(Nmax)
    for k in range(Nmax):
        ir[k] = (acf*np.cos(2*np.pi*k*np.arange(0,Nmax,1)/(2*Nmax-1))).sum()    
    return ir
    
mean_value = mu.mean(axis=0)
correlation = calc_corr_fft(mu-mean_value,Nmax)

file_handle = open("dip_dip_correlation.time.txt",'w')
file_handle.write("# correlation in the time domain, first column time in fs\n")

for i in range(-Nmax+1,Nmax):
    file_handle.write("%f %f\n" %(i*time_interval,correlation[abs(i)]))
file_handle.close()

Kronecker_function = np.ones(Nmax) *2
Kronecker_function[0] = 1
Hann_window = (np.cos(np.pi*np.array(range(Nmax))/Nmax)+1)*0.5
correlation = correlation * Hann_window * Kronecker_function

cm_array = np.arange(0,Nmax,1)/((2*Nmax-1)*time_interval*10**(-15)*2.99792458*10**(10)) 
ir_intensity = compute_power_spectrum(correlation,Nmax)
data = np.array([cm_array,ir_intensity*cm_array**2]).T

def smooth_curve(x, k):
    window = np.ones(k) / k
    smoothed = np.convolve(x, window, mode='valid')
    return smoothed

np.savetxt("ir_original.txt",data)
# Smooth the curve
smoothed = smooth_curve(data[:,1], 10)
data = np.array([cm_array[:len(smoothed)],smoothed]).T
np.savetxt("ir_smooth.txt",data)