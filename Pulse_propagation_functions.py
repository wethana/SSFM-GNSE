# -*- coding: utf-8 -*-

import numpy as np
from scipy.fft import fft, fftfreq, ifft, fftshift
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, CheckButtons
import pandas as pd

random.seed()
##############################################################################


def STD(x, y):
    """
    Caluculates the Root Mean Square of y(x)

    Parameters
    ----------
    x : Independent variable
        
    y : Dependent variable

    Returns
    -------
    STD : Standard deviation of continuous function

    """
    dx = x[1] - x[0]
    #rms = (np.nansum(y * x**2)  - np.nansum(y * x * dx)**2) / (np.nansum(y[~np.isnan(x)]) * dx) #(3.2.26)
    norm_factor = sum(y)*dx
    y = y/norm_factor
    Mean = np.nansum(x*y)*dx
    
    STD = np.sqrt(np.nansum((x-Mean)**2 * y)*dx)

    if np.isnan(STD):
        print("STD returns nan")
    return STD
##############################################################################

#Generate STUD pulse
def STUD_pulse(n_spikes, t, t_on, Dt, C = 0, order = 1, jitter_A = 0, jitter_t = 0, jitter_w = 0, t_center = 0, Energy = 0, coupled = False, rand_phase = False, **kwargs):
    dt = t[1] - t[0]
    T0 = t_on /(2 * np.log(2)**(1/(2* order))) #Convert FWHM power to sigma in E-field   
    
    counter = 0
    while True:
        counter = counter + 1
        if coupled:
            U0 = np.zeros([n_spikes,len(t)],dtype=np.complex64)
        elif not coupled:
            U0 = np.zeros(len(t))
        
        #Generate spike times, widths, and Amplitudes
        if n_spikes == 1:
            spike_t = np.array([0])
        else:
            #spike_t = np.linspace(-n_spikes * Dt/2, n_spikes * Dt/2, n_spikes) 
            spike_t = (np.array(range(n_spikes)) - n_spikes/2 + 0.5) * Dt
            spike_t = spike_t + (np.array([random.random() for _ in range(n_spikes)]) - 0.5) *2 * jitter_t * Dt
        spike_w  = (np.array([random.random() for _ in range(n_spikes)]) - 0.5) * 2 * jitter_w * T0 +T0
        #spike_A = np.array([np.sqrt(1 + (random.random() - 0.5) *2 * jitter_A) for _ in range(n_spikes)]) 
        spike_A = np.sqrt(T0 / spike_w) #Sqrt because we want intensity variation
        
        # Generate spikes
        for n in range(n_spikes):
            if coupled:
                U0[n] = spike_A[n] * np.exp(-(1 + 1j * C) / 2 * ((t - t_center - spike_t[n])/(spike_w[n])) ** (2 * order) + 2j * rand_phase * np.pi * random.random())
            elif not coupled:
                U0 = U0 + spike_A[n] * np.exp(-(1 + 1j * C) / 2 * ((t - t_center - spike_t[n])/(spike_w[n])) ** (2 * order) + 2j * rand_phase * np.pi * random.random())
                
        #Check to see if spikes are overlapping improperly.  This is not perfect.
        if t_on > Dt * 0.7:
            break
        t_dif = (spike_t[1:-1] - spike_t[0:-2]) / (abs(spike_w[1:-1]+spike_w[0:-2])/(2*Dt))
        #rint(t_dif)
        if all(t_dif > (1 + 1/order)): #1/order is my guesss to account for the fact that gaussians are wider than their on time.
            break
        if counter > 5:
            print("Cannot prevent spike overlap")
            break
        print("try " + str(counter))
    
    spike_A = spike_A / np.max(abs(U0)**2)
    U0 = U0 / np.max(abs(U0)) #This sets the max value of U0 to 1 so that P0 represents peak power
    if coupled:
        combined_spikes = np.sum(U0, axis = 0)
    elif not coupled:
        combined_spikes = U0
        
    P0 = Energy / (sum(abs(combined_spikes)**2) * dt) * 1000 # nJ/ps p/n = W  #Energy = P0 * integrate(U) if max(U) = 1, then P0 is peak power
    
    #plt.plot(t,abs(U0)**2)
    return U0, T0, P0, spike_A, spike_w, spike_t
        

t = np.linspace(-200,200,1000)
#U0, T0, P0, spike_A, spike_w, spike_t = STUD_pulse(16, t ,15, 30, order = 3, C = 1, jitter_A = 0.3, jitter_t = 0.3, jitter_w=0.3, Energy = 9)
#plt.figure()
#plt.plot(t, P0 * np.transpose(abs(U0)**2),'g',linewidth = 2)
#plt.savefig('demo.png', transparent=True)
#U0, T0, P0, spike_A, spike_w, spike_t = STUD_pulse(1, t ,10, 30, order = 1, C = 1, jitter_A = 0, Energy = 9)
#plt.plot(t, P0 * np.transpose(abs(U0)**2))

##############################################################################
def GenerateU(t,T0_P = 0.05, Energy = 9, C = 0, phi = 0, PowerProfile = 0, f_type = "Gaussian", order = 1, FWHM = True, rand_phase = False):
    """
    

    Parameters
    ----------
    t : time array with real units.
    
    T0_P : Characteristic time of pulse as measured with power as opposed to electric field.
        DESCRIPTION. The default is 0.05.
    Energy : float, Energy of pulse with real units.
        DESCRIPTION. The default is 9.
    C : float, Chirp parameter relavent to Gaussian and Sech pulses.
        DESCRIPTION. The default is 0.
    phi : Array, Phase information. Superseeds chirp parameter Only required for "arb"
        DESCRIPTION. The default is 0.
    PowerProfile : array, Power profile of the pulse. only works if f_type == "arb" 
        DESCRIPTION. The default is 0.
    f_type : String, "Gaussian", "sech", "arb", May put "STUD" in later
        DESCRIPTION. The default is "Gaussian".
    order : float, Determins the order of Gaussian pulse
        DESCRIPTION. The default is 1.
    FWHM : bool, set to True if T0_P is the FWHM of power
                 set to False if T0_P is already the charactaristic time scale.
        DESCRIPTION. The default is True.

    Returns
    -------
    tau : TYPE
        DESCRIPTION.
    U : TYPE
        DESCRIPTION.
    T0 : TYPE
        DESCRIPTION.
    P0 : TYPE
        DESCRIPTION.

    """
    dt = t[1] - t[0]
    if f_type == "Gaussian":
        if FWHM:
            T0_P = T0_P /(2 * np.log(2)**(1/(2* order))) #Convert FWHM to sigma
        T0 = 2 ** (1/(2*order)) * T0_P    #Convert Power to E field
        if phi == 0:
            U = np.exp(-(1 +1j*C)/2 * (t/T0)**(2 * order))  
        else:
            U = np.exp(-1/2 * (t/T0)**(2 * order)) * np.exp(1j * phi)  
        P0 = Energy / (sum(abs(U)**2) * dt) /1000 # nJ/ps p/n = W
    elif f_type == "sech":
        if FWHM:
            T0 = T0_P / (2 * np.arccosh(np.sqrt(2)))
        else:
            T0 = T0_P
        if phi == 0:
            U = 1 / np.cosh(t/T0) * np.exp( - 1j * C * t**2 /(2 * T0**2))
        else:
            U = 1 / np.cosh(t/T0) * np.exp( - 1j * C * t**2 /(2 * T0**2))
        P0 = Energy / (sum(abs(U)**2 * dt)) / 1000
    elif f_type == "arb": # This function is not yet supported.
        P0 = max(PowerProfile)
        U = np.sqrt(PowerProfile / P0) * np.exp(phi)
        T0 = STD(t,np.abs(U))
    else:
        raise ValueError('Pulse type not defined')
    tau = t/T0
    
    return tau, U, T0, P0

##############################################################################
def NonLinear(U_temp, s, tauR, tau):
    NL = abs(U_temp) ** 2 + 1j * s * np.conj(U_temp) * np.gradient(U_temp, tau) + (1j * s - tauR) * np.gradient(abs(U_temp)**2,tau) # See Deoterdomg and pool 2016 Robust split-step Fourier methods for simulating the propagation of ultra-short pulses in single-and two-mode optical communication fibers chapter 18 (18.14) 
    return NL

def coupled_NonLinear(U_temp, s, tauR, tau, σ = 2, γm = 1):
    # σ = 2 for pulses with the same polarization
    # γm is the nonlinear parameter for each pulse normalized to γ0
    # U_temp[m, T]
    UU_temp = abs(U_temp)**2
    summed_powers = np.sum(γm * UU_temp, axis = (0))
    NL = σ * summed_powers + (γm - σ) * UU_temp 
    return NL
    
###############################################################################
def run_simulation(U0, beta2, beta3, N, omega, alpha, LD, LD3, deltaz,step_num, nt, D, s, tauR, tau, steps, t):
    U = np.zeros([step_num,nt],dtype=np.complex_) #Initialize U
    U[0,:] = U0
    hhz = 1j*N**2*deltaz
    temp = U[0,:]*np.exp(NonLinear(U[0,:], s, tauR, tau) * hhz/2)
    
    if LD3 ==0 or beta3 == 0:
        dispersion = np.exp(D * (0.5j * np.sign(beta2) * omega**2 - alpha * LD / 2) * deltaz) #Preloaded phase factor
    else:
        dispersion = np.exp(D * (0.5j * np.sign(beta2) * omega**2 + 1j/6 * np.sign(beta3) * LD/LD3 * omega**3 - alpha * LD / 2) * deltaz) #Preloaded phase factor
    #Main Loop
    for n in steps[1:-1]:
        
        f_temp = ifft(temp)*dispersion
        U[n,:] = fft(f_temp)
        temp = U[n,:] * np.exp(NonLinear(U[n,:], s, tauR, tau) * hhz)

        if n%150 == 0:
            print('Step ' + str(n) +'/'+str(step_num))
            
        if np.isnan(np.nanmax(U[n,:])):
            print("U is nan")
            continue
    U[-1,:] = temp * np.exp(-NonLinear(U[-2,:],  s, tauR, tau) * hhz/2)  
    return U

def coupled_simulation(U0, beta2, beta3, N, omega, alpha, LD, LD3, deltaz,step_num, nt, D, s, tauR, tau, steps, t, coupled):
    #Initialize M different electric fields
    #U[m,z_step, T] 

    
    if LD3 ==0 or beta3 == 0:
        dispersion = np.exp(D * (0.5j * np.sign(beta2) * omega**2 - alpha * LD / 2) * deltaz) #Preloaded phase factor
    else:
        dispersion = np.exp(D * (0.5j * np.sign(beta2) * omega**2 + 1j/6 * np.sign(beta3) * LD/LD3 * omega**3 - alpha * LD / 2) * deltaz) #Preloaded phase factor

    
    hhz = 1j*N**2*deltaz
    
    if coupled:
        M = len(U0[:,0])
        U = np.zeros([M,step_num,nt],dtype=np.complex_) #Initialize U
        U[:,0,:] = U0    
        temp = U[:,0,:]*np.exp(coupled_NonLinear(U[:,0,:], s, tauR, tau) * hhz/2)
        #Main Loop
        for n in steps[1:-1]:
            f_temp = ifft(temp)*dispersion
            U[:,n,:] = fft(f_temp)
            temp = U[:,n,:] * np.exp(coupled_NonLinear(U[:,n,:], s, tauR, tau) * hhz)
            
            if n%150 == 0:
                print('Step ' + str(n) +'/'+str(step_num))
            
            if np.isnan(np.nanmax(U[:, n, :])):
                print("U is nan")
                continue
            
        U[:,-1,:] = temp * np.exp(coupled_NonLinear(U[:,-2,:],  s, tauR, tau) * hhz/2) 
    else:
        U = np.zeros([step_num,nt],dtype=np.complex_) #Initialize U
        U[0,:] = U0
        temp = U[0,:]*np.exp(NonLinear(U[0,:], s, tauR, tau) * hhz/2)
        
        #Main Loop
        for n in steps[1:-1]:
            f_temp = ifft(temp)*dispersion
            U[n,:] = fft(f_temp)
            temp = U[n,:] * np.exp(NonLinear(U[n,:], s, tauR, tau) * hhz)

            if n%150 == 0:
                print('Step ' + str(n) +'/'+str(step_num))
                
            if np.isnan(np.nanmax(U[n,:])):
                print("U is nan")
                continue
        U[-1,:] = temp * np.exp(NonLinear(U[-2,:],  s, tauR, tau) * hhz/2)  
    return U

###############################################################################
def generate_slider_graph(U, beta2, beta3, deltaz, tau, dtau, P0, dt, freq, step_num, T0, c, omega0, omega, lambda0, LD, LD3, LNL, Energy0, N, alpha, s, tauR, t):
    
    #Calculate the phase, spectrum and on_time at all points in the fiber
    UU = abs(U)**2
    phi      = np.unwrap(np.arctan2(np.imag(U),np.real(U)))                                     # Get phase
    exact_dw = [beta2 * n * deltaz *tau / ((1 + (n * deltaz)**2))  for n in range(len(U[:,0]))] # caclulated dispersion needed to compensate for fiber
    spect    = [abs(fftshift(ifft(U[n,:])))**2                     for n in range(len(U[:,0]))] # Load spectrum
    spect    = [spect[n]/np.nanmax(spect[n])                       for n in range(len(U[:,0]))] # Normalize spectrum
    dw       = [-np.gradient(phi[n,:],dtau)                        for n in range(len(U[:,0]))] # Change in frequency due to changing phase
    Energy   = [sum(UU[n,:]) * dt * P0                             for n in range(len(U[:,0]))] # E = int(P(t)) = P0 * int(UU*(t))
    STD_freq = [STD(freq, spect[n])                                for n in range(len(U[:,0]))] # Standard deviation of frequency (bandwidth)
    STD_tau  = [STD( tau, UU[n,:])                                 for n in range(len(U[:,0]))] # Standard deviation of pulse intensity (t_on)
    for n in range(len(U[:,0])):            # Remove frequency shift information for low signals where it is just noise.
        dw[n][UU[n,:] < np.max(UU[n,:]) * 0.001] = np.nan
    
    
    gif_fig = plt.figure( figsize=(6,6), dpi = 200)
    gs = gif_fig.add_gridspec(5,2, width_ratios = (4,1), height_ratios = (1, 0.8, 0.4, 1,0.3), hspace = 0,wspace = 0.5)
    font = {'family': 'serif',
            'serif': ['Palatino Linotype'],
            'weight': 'normal',
            'size': 10}
    plt.rc('font', **font)

    # Add sliders
    z_slider_ax = plt.axes([0.1, 0.02, 0.6, 0.05])
    z_slider = Slider(z_slider_ax, 'Z = ', 0, step_num-1, valinit=0, valstep=1)
    gif_fig.text(0.8,0.05,"$L_D$")

    # Add text box for manual input
    ax_z_text = plt.axes([0.7, 0.02, 0.09, 0.05])
    text_z = TextBox(ax_z_text, '', initial=str(0))

    #Get plot x limits
    freq_lim = np.array([-1,1])*STD_freq[-1] * 5
    if freq_lim[1]> freq[-1] or any(np.isnan(freq_lim)):
        freq_lim = [freq[0],freq[-1]]
    tau_lim  = np.array([-1,1])*STD_tau[-1] * 4
    if tau_lim[1]> tau[-1] or any(np.isnan(tau_lim)):
        tau_lim = [tau[0],tau[-1]]

    #Initialize Intensity profile plots
    gif_I = gif_fig.add_subplot(gs[0,0])
    y_I = UU[0,:]
    line_I =gif_I.plot(tau, y_I)[0]
    gif_I.set(xlim = tau_lim, ylim= [0,np.nanmax(UU)*1.05], xlabel = 'τ', ylabel = 'Power/P0')
    gif_I.grid()
    gif_Iy = gif_I.twiny()
    gif_Iy.set_xticks(gif_I.get_xticks())
    gif_Iy.set_xticklabels(['{:,.1f}'.format(T0 * t) for t in gif_I.get_xticks()])
    gif_Iy.set_xlim(gif_I.get_xlim())
    gif_Iy.set_xlabel("T [ps]")
    gif_I.xaxis.set_ticks_position('none')
    #gif_I2 = gif_I.twinx()
    #gif_I2.set_ylim(gif_I.get_ylim())
    #gif_I2.set_yticklabels(['{:,.2f}'.format(x*P0) for x in gif_I.get_yticks()])


    #Initialize chirp plots
    gif_dw = gif_fig.add_subplot(gs[1,0])
    y_dw = dw[0]
    line_dw =gif_dw.plot(tau, y_dw)[0]
    line_exact_dw = gif_dw.plot(tau,exact_dw[0],'r', linestyle = 'None')[0]
    gif_dw.set(xlim = tau_lim, ylim= np.array([-1,1])*np.nanmax(dw), ylabel = '$T_0$ δω')
    gif_dw.set_xlabel("τ", labelpad = -3)
    gif_dw.grid()
    gif_dw2 = gif_dw.twinx()
    gif_dw2.set_yticks(gif_dw.get_yticks())
    gif_dw2.set_yticklabels(['{:,.0f}'.format(1e9 * (2 * np.pi * c/(omega0 + x/T0) - lambda0)) for x in gif_dw2.get_yticks()])
    gif_dw2.set_ylabel("$\delta\lambda \:[nm]$")
    gif_dw2.set_ylim(gif_dw.get_ylim())

    #Initialize spectrum profile plots
    gif_f = gif_fig.add_subplot(gs[3,0])
    y_f = spect[0]
    #line_f = gif_f.plot(freq ,y_f)[0]
    #gif_f.set(xlim = freq_lim, ylim= [0,1.1], xlabel = '$T_0\: \delta\omega$', ylabel = 'Normalized Power')
    line_f = gif_f.plot(freq/T0 ,y_f)[0]
    gif_f.set(xlim = freq_lim/T0, ylim= [0,1.1], xlabel = '$\delta\omega$ [THz]', ylabel = 'Normalized Power')

    #Initialize table
    gif_table = gif_fig.add_subplot(gs[:,1])
    gif_table.set_axis_off()
    table_dict = {"z [m]":    0,
                  "z [$L_D$]":  0,
                  "$T_0$ [ps]":              '{:.1e}'.format(T0), 
                  "$T_{STD} [ps]$":          '{:.1e}'.format(STD(t,UU[0,:])),
                  'β_2':                     '{:.1e}'.format(beta2), 
                  'β_3':                     '{:.1e}'.format(beta3), 
                  "$L_D$ [m]":               '{:.1e}'.format(LD),
                  "$L_{D3}$ [m]":            '{:.1e}'.format(LD3),
                  "$L_{NL}$ [m]":            '{:.1e}'.format(LNL),
                  "Energy [nJ]":             '{:.1e}'.format(Energy0),
                  "$P_0$ [W]":               '{:.1e}'.format(P0),
                  "$N$":                     '{:.1e}'.format(N),
                  "$α_{dB} [m^{-1}]$":       '{:.1e}'.format(alpha * 4.343), #Convert alpha to alpha in decibels
                  "s":                       '{:.1e}'.format(s),
                  "$T_R/T_0$":               '{:.1e}'.format(tauR),
                  
                  #"$\lambda_{RMS} [nm]$":    '{:.1e}'.format(rms(Lambda,spect)),
                  }
    table_list = list(zip(table_dict.keys(), table_dict.values()))
    table = gif_table.table(table_list,
                            cellLoc = 'center', loc = 'upper left',colWidths = [0.75, 0.8])

    #gif_table.patch.set_visible(False)
    #gif_table.axis('off')
    #gif_table.axis('tight')

    #Make gui of pulse profile and spectrum
    def update(val):
        #if frame%25==0:
        #    print('frame ' + str(frame) + '/' +str(int(step_num/compression)))
        #update lines
        n = int(z_slider.val)
        if n > len(U[:,0]):
            n = len(U[:,0])
        y_I = abs(U[n,:])**2
        line_I.set_ydata(y_I)
        #exact_sigma = ((1 + C*n*deltaz)**2 + (n*deltaz)**2)**0.5/np.sqrt(2)  #The sqrt(2) is because this is for U**2  This is checked
        
        y_dw= dw[n]
        line_dw.set_ydata(y_dw)
        
        y_exact_dw = exact_dw[n]
        line_exact_dw.set_ydata(y_exact_dw)
        
        y_f = spect[n]
        line_f.set_ydata(y_f)
        
        #update table
        table_dict["z [m]"]                 = '{:.1e}'.format(n*deltaz * LD)
        table_dict["z [$L_D$]"]             = '{:.1e}'.format(n*deltaz)
        table_dict["Energy [nJ]"]           = '{:.2e}'.format(Energy[n])
        table_dict["$T_{STD} [ps]$"]        = '{:.1e}'.format(STD(t,UU[n,:]))
        table_list = list(zip(table_dict.keys(), table_dict.values()))
        
        for i in range(int(len(table._cells)/2)):
           table._cells[i,1]._text.set_text(table_list[i][1])

        #gif_table.table([["z [m]", z_pos_m],["z [L_D]", z_pos_LD],["T_0 [ps]",T0],["Energy [nJ]",Energy[n]]])
        
        text_z.set_val('{:.1f}'.format(n*deltaz))
        #update text
        #z_pos = 'z = ' + '{:.1e}'.format(n*deltaz) + ' * L_{D}\nz = ' + '{:.1e}'.format(n*deltaz * LD) + ' m'
        #txt_I.set_text(z_pos + '\n T/T0 = ' + '{:.1f}'.format(sigma) )#+ '\n exact = '+ '{:.1f}'.format(exact_sigma))
        #txt_dw.set_text(z_pos)
        #txt_f.set_text(z_pos)
        return (line_I, line_dw, line_exact_dw, line_f, gif_table)

    def update_text(text):
        try:
            z = float(text)
            n = int(z/deltaz)
            z_slider.set_val(n)
        except ValueError:
            pass
        
    z_slider.on_changed(update)
    text_z.on_submit(update_text)
    plt.show()

    return 0



# Function to apply dispersion
def apply_dispersion(b2, b3, f, tau, omega):
    dispersion = np.exp((0.5j * b2 * omega**2  + 1j/6 * b3 * omega**3)) #Phase factor
    f_temp = ifft(f)*dispersion
    return fft(f_temp)

def characterize_STUD(spike_t, UU, t, T0, P0):
    #Bin the spikes
    N = len(spike_t)
    dt = t[1] - t[0]  #Spacing between spikes
    
    #Set up window for each spike
    if len(spike_t) == 1:
        spike_dt_mean = (t[-1] - t[0])/2.2 #A little less then half of time window
    else:
        spike_dt_mean = (spike_t[-1] - spike_t[0]) / (N - 1)

    spike_w = np.zeros(N)
    spike_A = np.zeros(N)
    spike_t_new = np.zeros(N)
    spike_E = np.zeros(N)
    #plt.figure()
    for n in range(N):
        if len(spike_t) == 1:
            t_index = np.array(range(len(t)))
        else:
            t_index = np.where((t > spike_t[n] - spike_dt_mean/2) * (t < spike_t[n] + spike_dt_mean/2))[0]
        t_snip = t[t_index]
        spike = UU[t_index] 
        spike_A[n] = spike.max()
        spike_t_new[n] = t_snip[np.where(spike_A[n])]
        
        # This approximates FWHM, but has its flaws.
        #spike_top_index = np.where(spike>spike.max()/2)[0]
        #spike_w[n] = (spike_top_index[-1] - spike_top_index[0]) * dt # Find how many steps between first and land point above 50% max value and multiply by the distance between steps.
        spike_w[n] = STD(t_snip,spike) * 2.35482 # Get the standard deviation and multiply by 2sqrt(2ln2) to get FWGHM 
        spike_E[n] = sum(spike) * dt * P0 /1000 #[nJ]
        #plt.plot(tau_snip, spike)
    spike_dt = spike_t_new[1:] - spike_t_new[0:-1]
    
    Dt = np.mean(spike_dt)
    if N >1:
        f_DC_text = r'$f_{DC} = $' + '{:.2f}'.format(np.mean(spike_w/Dt)*100) +'%\n'
        spike_width_std  = r'$\sigma(t_{on})$'  + '{:.3f}'.format(np.std( spike_w)/np.mean(spike_w)*100) + '%\n'  
    else:
        f_DC_text = ''
        spike_width_std  = ''  
    spike_width_mean = r'$t_{on} = $'          + '{:.3f}'.format(np.mean(spike_w)) + ' [ps]\n'    
    
    spike_energy     = r'$\overline{E_{spike}}$ = ' + '{:.2f}'.format(np.mean(spike_E)) + ' [nJ]\n'  
    
    spike_energy_std = r'$\sigma_{E}$ = '  + '{:.2f}'.format(np.std(spike_E)/np.mean(spike_E)*100) + '%\n'
    
    spike_amplitude_std = r'$\sigma_A$ = ' + '{:.2f}'.format(np.std(spike_A)/np.mean(spike_A)*100) + '%\n'
    
    STUD_text = f_DC_text + spike_width_mean + spike_width_std + spike_energy + spike_energy_std + spike_amplitude_std
    
    return spike_w, spike_A, spike_t_new, spike_dt, spike_E, STUD_text
        

def reverse_dispersion_plot(U, beta2, beta3, deltaz, tau, dtau, P0, dt, freq, step_num, T0, c, omega0, omega, lambda0, LD, LD3, LNL, Energy0, N, alpha, s, tauR, t, spike_A, spike_w, spike_t ):
    UU = abs(U)**2
    
    # Initial Guess is negative of applied simulated dispersion
    b2 = -deltaz * (step_num-1) # note Lenght is already normalized to LD
    b2_lim = (np.array([-1,1])*abs(b2) + b2)*10

    if LD3 == 0:
        b3 = 0
        b3_lim = [-1, 1]
    else:
        b3 = -LD/LD3 * deltaz * (step_num - 1)
        b3_lim = (np.array([-1,1])*abs(b3) + b3)*3
    U_temp = apply_dispersion(b2, b3, U, tau, omega)

    #Set up figure
    fig = plt.figure( figsize=(6,4), dpi = 200)
    gs = fig.add_gridspec(2,2, height_ratios = (1,0.15), width_ratios= (1, 0.2), hspace = 0,wspace = 0.5)
    
    ax0 = fig.add_subplot(gs[0,0])

    
    if len(spike_t)>1:
        xlimits = np.array([spike_t[0], spike_t[-1]])*1.3
        plt.xlim(xlimits)
        ax0.set_xlim(xlimits)
    corrected, = ax0.plot(t, abs(U_temp)**2, 'g', label = 'Compensated')
#    origional, = ax0.plot(tau,UU[0],           'b', label = 'Origional')
    origional2, = ax0.plot(spike_t, spike_A, label = "Origional peak locations", marker = 'o', linestyle = 'None',)
    #after,     = ax0.plot(tau,UU[-1],          'r', label = 'After Fiber')
    
    
    #ax0.legend()


    
    spike_w_new, spike_A_new, spike_t_new, spike_dt_new, spike_E_new = characterize_STUD(spike_t, UU, t,T0)
    dt_factor = (len(spike_t) - 1)/len(spike_t) #the dt measurements come from N spikes but have N-1 differences
    
    var_list  = ["Energy [nJ]",      "$t_{on} [ps]$",        "ΔT [ps]"]
    mean_list = [np.mean(spike_E_new), np.mean(spike_w_new), np.mean(spike_dt_new) *dt_factor]
    std_list  = [np.std(spike_E_new),  np.std(spike_w_new),            np.std(spike_dt_new)]
    
    mean_list = list(map(lambda x: f'{x:.2e}', mean_list))
    std_list  = list(map(lambda x: f'{x:.2e}', std_list))
    
    table_fig = fig.add_subplot(gs[:,1])
    table_fig.set_axis_off()
    table_dict = {"Var":   var_list, 
                  "mean":  mean_list,
                  "std":   std_list
                  }
    table_df = pd.DataFrame(table_dict) 
    table_STUD = table_fig.table(cellText = table_df.values, colLabels = table_df.columns,
                            cellLoc = 'center', loc = 'upper right',colWidths = [0.75, 0.8, 0.8])
    #table.auto_font_size(False)
    table_STUD.set_fontsize(12)
    table_STUD.scale(1,1)

    # Add sliders
    ax_b2 = plt.axes([0.1, 0.1, 0.65, 0.03])
    ax_b3 = plt.axes([0.1, 0.05, 0.65, 0.03])
    slider_b2 = Slider(ax_b2, 'b2', b2_lim[0], b2_lim[1], valinit=b2)
    slider_b3 = Slider(ax_b3, 'b3', b3_lim[0], b3_lim[1], valinit=b3)

    # Add text boxes for manual input
    ax_b2_text = plt.axes([0.75, 0.1, 0.2, 0.05])
    ax_b3_text = plt.axes([0.75, 0.045, 0.2, 0.05])
    text_b2 = TextBox(ax_b2_text, '', initial=str(b2))
    text_b3 = TextBox(ax_b3_text, '', initial=str(b3))

    # Add check buttons
#    check_ax = plt.axes([0.75, 0.75, 0.2, 0.1])


#    def toggle_visibility(label):
#        if label == 'Origional':
#            print("toggled "+ str(not origional.get_visible()))
#            origional.set_visible(not origional.get_visible())       
#        elif label == 'After Fiber':
#            after.set_visible(not after.get_visible())
#            print("toggled")
#        elif label == 'Compensated':
#            corrected.set_visible(not corrected.get_visible())
#            print("toggled")
        

    def update2(val):
        b2 = slider_b2.val
        b3 = slider_b3.val
        text_b2.set_val("{:.3f}".format(b2))
        text_b3.set_val("{:.3f}".format(b3))
        U_temp = apply_dispersion(b2, b3, U, tau, omega)
        UU_temp = abs(U_temp)**2
        if ax0.get_ylim()[1] < max(UU_temp):
            ax0.set_ylim([0, max(UU_temp) ])
        corrected.set_ydata(UU_temp)
        corrected.set_label("asdf")
        
        spike_w_new, spike_A_new, spike_t_new, spike_dt_new, spike_E_new = characterize_STUD(spike_t, UU_temp, t, T0)
        dt_factor = (len(spike_t) - 1)/len(spike_t) #the dt measurements come from N spikes but have N-1 differences
        var_list  = ["Energy [nJ]",      "$t_{on} [ps]$",        "ΔT [ps]"]
        mean_list = [np.mean(spike_E_new), np.mean(spike_w_new), np.mean(spike_dt_new)*dt_factor]
        std_list  = [np.std(spike_E_new),  np.std(spike_w_new),            np.std(spike_dt_new)]
        
        mean_list = list(map(lambda x: f'{x:.2e}', mean_list))
        std_list  = list(map(lambda x: f'{x:.2e}', std_list))
        
        table_fig = fig.add_subplot(gs[:,1])
        table_fig.set_axis_off()
        table_dict = {"Var":   var_list, 
                      "mean":  mean_list,
                      "std":   std_list
                      }
        table_df = pd.DataFrame(table_dict) 
        
        for j in range(1,table_df.shape[0]):
            for i in range(0, table_df.shape[1]):
                table_STUD._cells[i+1,j]._text.set_text(table_df.iloc[i,j])
        
        plt.draw()
        #print("$t_{on} [ps]$ = " + mean_list[1])
        

    def update_text_b2(text):
        try:
            b2 = float(text)
            slider_b2.set_val(b2)
        except ValueError:
            pass

    def update_text_b3(text):
        try:
            b3 = float(text)
            slider_b3.set_val(b3)
        except ValueError:
            pass

    slider_b2.on_changed(update2)
    slider_b3.on_changed(update2)
    text_b2.on_submit(update_text_b2)
    text_b3.on_submit(update_text_b3)
    #check_button = CheckButtons( check_ax, ['Origional', 'After Fiber', "Compensated"], [True, True, True])  
    #check_button.on_clicked(toggle_visibility)
    
def Raman_gain(lambda0): #Copolarized 
    c =  2.99792458E-2 #speed of light cm/ps 
    data = pd.read_csv("C:\\Users\\Owner\\Desktop\\Fiber and nonlinear Optics\\Raman gain from wave number cm-1.csv", 
                       names = [ "wave number","gain_normalized"])
    data['f [THz]'] = data['wave number'] * c
    data = data.sort_values(by = 'wave number')
    data["gain"] = data["gain_normalized"] * 1e-13 /lambda0 # lambda0 in um
    return data

#data = Raman_gain()
#plt.plot(data['f [THz]'],data['gain'])

def Raman_response():
    data = pd.read_csv("C:\\Users\\Owner\\Desktop\\Fiber and nonlinear Optics\\Raman response from t-tp fs.csv", 
                       names = [ "fs","response"])
    data["ps"] = data["fs"]/1000
    return data
#data = Raman_response()
#plt.plot(data["fs"],data["response"])

def get_n(Lambda):
    B = np.array([0.6961663, 0.4079426, 0.8974794])# [Unitless]
    lambdai = np.array([0068.4043, 0116.2414, 9896.161]) #[nm]
    if any(np.divide(np.abs(Lambda- lambdai),lambdai)<0.4):
        print("Warning! too close to resonance")
    S = sum([B[j]/(1-(lambdai[j]/Lambda)**2) for j in range(len(lambdai))])  #Sellmeier sum
    return np.sqrt(1 + S)

def get_beta1(lamb):
    h = 10000000000 #steps in Hz used to differentiate. This is minimum value
    c =  2.99792458E14#um/s
    c2pi = 2 * c * np.pi
    omega = 2 * np.pi * c/lamb
    dndω = (get_n(c2pi/(omega + h)) - get_n(c2pi/(omega - h)))/(2*h)
    return 1/c * (get_n(lamb) + omega * dndω) * 1E18 # 1/c (n + ω dn/dω) In [ps/m] 

def get_beta2(lamb):
    h = 10000000000 #steps in Hz used to differentiate. This is minimum value
    c =  2.99792458E14#um/s
    c2pi = 2 * c * np.pi
    omega = c2pi/lamb
    dndω = (get_n(c2pi/(omega + h))- get_n(c2pi/(omega - h)))/(2*h)
    d2ndω2 = (get_n(c2pi/(omega + 2*h)) + get_n(c2pi/(omega - 2*h))- 2*get_n(c2pi/(omega)))/(4*h**2)
    return 1/c *(2*dndω + omega* d2ndω2)*1E27 # 1/c(2 dn/dω + ω d^2n/dω^2) # We want units ps^2/m

def get_beta3(lamb):
    h = 10000000000 #steps in Hz used to differentiate. This is minimum value
    c =  2.99792458E14#um/s
    c2pi = 2 * c * np.pi
    omega = c2pi/lamb
    return (get_beta2(c2pi/(omega + h)) - get_beta2(c2pi/(omega - h)))/(2*h) * 1E9 #ps^3/m
    
    
    