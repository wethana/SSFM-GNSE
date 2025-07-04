# -*- coding: utf-8 -*-
"""
Minimal SSFM Code for pulse propagation in a fiber
@author: Ethan Welch

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift

# Simulation parameters
t_on     = 0.150 # ps On time (FWHM)
T0       = t_on * np.log(2)**(-1/2)/2 # Converts FWHM to T0
T_window = 200   # ps
nt       = 2**14 # Number of time points
dt       = T_window / nt # [ps]
dtau     = dt/T0
t        = np.array(range( int(-nt/2), int(nt/2))) * dt # [ps]
tau      = t/T0
omega    = fftshift(range(int(-nt/2),
                          int(nt/2)))*(2*np.pi/T_window*T0)
freq = fftshift(omega)/(2*np.pi)

# Pulse parameters
Energy = 4 # nJ
U0  = np.exp(-1 / 2 * (t/T0) ** 2)
P0 = Energy / (sum(abs(U0)**2) * dt) *1000 # nJ/ps p/n = W
c = 2.99792458e-4  # m/ps
Lambda = 1053e-9 # 1053 nm
omega0 = 2 * np.pi * c / Lambda

# Fiber parameters Calculated for silicone dioxide at 1053 nm
fiber_length = 0.1                  # [m]
beta2        = 0.0172798            # ps^2/m
beta3        = 4.3315350e-5         # ps^3/m
alpha        = 5.8e-4               # 1/m
gamma        = 4.2483e-3            # [1/(W m)]
LD2          = T0 ** 2 / abs(beta2) # [m]
LD3          = T0 ** 3 / abs(beta3) # [m]
LNL          = 1 / (gamma * P0)     # [m]
s            = 1 / (omega0 * T0)    # Steepening parameter
Tr           = 0.003                # [ps]
tauR         = Tr / T0              # Normalized 
N            = np.sqrt(LD2 / LNL)   # soliton number

# Other calculated parameters
step_num = round(20 * fiber_length / LD2 * N**2)
steps = range(0,step_num)
deltaz = fiber_length / step_num
deltaZ = deltaz / LD2
hhz = 1j*N**2*deltaZ
tau = t/T0

# Initialize U
U = np.zeros([step_num,nt],dtype=np.complex_) 
U[0,:] = U0

# Precalculate dispersion
dispersion = np.exp((0.5j * np.sign(beta2) * omega**2 
                     + 1j/6 * np.sign(beta3) * LD2/LD3 * omega**3 
                     - alpha * LD2 / 2) * deltaZ)

# Function to Calculate nonlinear dispersion
# See Deoterdomg and pool 2016 Ch 18 (18.14)
def NonLinear(U_temp, s, tauR, tau):
    NL = abs(U_temp) ** 2 
    + 1j *s*np.conj(U_temp)*np.gradient(U_temp, tau) 
    + (1j * s - tauR) * np.gradient(abs(U_temp)**2,tau)  
    return NL

# Take first half step
temp = U[0,:]*np.exp(NonLinear(U[0,:], s, tauR, tau) * hhz/2)

# Main loop
for n in steps[1:-1]:
    f_temp = ifft(temp)*dispersion
    U[n,:] = fft(f_temp)
    temp = U[n,:]*np.exp(NonLinear(U[n,:],s,tauR,tau)*hhz)
    
# Take final half step
U[-1,:] = temp*np.exp(-NonLinear(U[-2,:],s,tauR,tau)*hhz/2) 

# Calculate initial and final spectrums
Uf = U[-1,:]
spect0    = abs(fftshift(ifft(U0)))**2
spect0    = spect0/np.nanmax(spect0)
spectf    = abs(fftshift(ifft(Uf)))**2
spectf    = spectf/np.nanmax(spectf)

# Calculate initial widths
rms_UUf    = np.sqrt(np.sum(t**2 * abs(Uf)**2)
                     / np.sum(abs(Uf)**2))
rms_spectf = np.sqrt(np.sum(freq**2 * spectf) 
                     / np.sum(spectf))


fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0,0].plot(t, abs(U[0,:])**2)
axes[0,0].set_title("Initial intensity profile")
axes[0,0].set_xlim([-4*rms_UUf, 4*rms_UUf])
axes[0,0].set_xlabel("T [ps]")

axes[0,1].plot(t, abs(U[-1,:])**2)
axes[0,1].set_title("Final intensity profile")
axes[0,1].set_xlim([-4*rms_UUf, 4*rms_UUf])
axes[0,1].set_xlabel("T [ps]")

axes[1,0].plot(freq, spect0)
axes[1,0].set_title("Initial spectrum")
axes[1,0].set_xlim([-4*rms_spectf, 4*rms_spectf])
axes[1,0].set_xlabel(r"$\Delta \omega \,[THz]$")

axes[1,1].plot(freq, spectf)
axes[1,1].set_title("Final spectrum")
axes[1,1].set_xlim([-4*rms_spectf, 4*rms_spectf])
axes[1,1].set_xlabel(r"$\Delta \omega \,[THz]$")

plt.tight_layout()
plt.show()