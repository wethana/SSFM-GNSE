# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:38:31 2024

@author: Ethan Welch
"""

import tkinter as tk
from tkinter import ttk, StringVar
from tkinter.ttk import Style
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.cm as cm
from scipy import pi
import sys
from Pulse_propagation_functions import STUD_pulse, NonLinear, coupled_NonLinear, STD, get_beta1, get_beta2, get_beta3, characterize_STUD
import pandas as pd
from scipy.fft import fft, ifft, fftshift #fftfreq,
from scipy.optimize import minimize




class STUD_GUI:
    global parameter   
    parameter = pd.read_excel("Setup_Variables2.xlsx", header = 0, index_col = 0, converters = {'col':int, 'default': float})
    parameter = parameter.fillna('')
    parameter = parameter.T.to_dict()
    global key
    key = list(parameter.keys())
    
    
    
    def __init__(self, root):
        self.root = root
        
        self.root.protocol("WM_DELETE_WINDOW", self.quit_me)
        
        self.root.title("STUD pulse profile")

        # Create the notebook (tabs container)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True)

        # Create tabs
        self.input_tab = ttk.Frame(self.notebook)
        self.output_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.input_tab, text='Input')
        self.notebook.add(self.output_tab, text='Output')        

        style = Style()
        style.configure('W.TButton', font =
               ('Palatino Linotype', 10, 'bold'),
                foreground = 'black', background = "green")

        self.c =  2.99792458E-4 #speed of light m/ps 
        self.omega0 = 2 * pi * self.c/(1.053 * 1E-6) #Should be [1/ps]    
        self.mode  = "P_edit"
        
        row = [1,1,1] #The one is to make room for the buttons
        
        for i in range(len(key)):
            col = parameter[key[i]]["col"]
            
            parameter[key[i]]["Entry_Label"] = ttk.Label(self.input_tab, text = parameter[key[i]]["Label"])
            parameter[key[i]]["Entry_Label"].grid(column = col, row = row[int(col/2)])
            
            if   parameter[key[i]]["Type"] == "Entry":
                parameter[key[i]]["StringVar"] = StringVar()
                parameter[key[i]]["Widget"] = ttk.Entry(self.input_tab, textvariable = parameter[key[i]]["StringVar"])
                parameter[key[i]]["Widget"].bind("<KeyRelease>", self.update_initial_plot)
                
                # Lock variables inconsistent with mode
                if not parameter[key[i]][self.mode]:
                    parameter[key[i]]["Widget"].config(state = 'readonly')
                
                
            elif parameter[key[i]]["Type"] == "Spinbox":
                values = tuple(eval(parameter[key[i]]["Options"]))
                #value = int(parameter[key[i]]["default"])
                parameter[key[i]]["Widget"] = ttk.Spinbox(self.input_tab, from_ = values[0], to = values[-1], values = values)
               
            elif parameter[key[i]]["Type"] == "Checkbutton":
                parameter[key[i]]["StringVar"] = StringVar()
                parameter[key[i]]["Widget"] = ttk.Checkbutton(self.input_tab, variable = parameter[key[i]]["StringVar"], onvalue = True, offvalue= False)
            
            print("Key is " + key[i])
            parameter[key[i]]["Widget"].bind("<FocusOut>", self.update_values, add = '+')
            parameter[key[i]]["Widget"].bind("<Return>",   self.update_values, add = '+')
            parameter[key[i]]["Widget"].grid(column = col+1, row = row[int(col/2)])
            row[int(col/2)] = row[int(col/2)]+1
                  

        # Plot
        self.fig1, self.ax = plt.subplots(figsize=(5, 6), nrows=2)
        self.fig1.subplots_adjust(hspace = 0.4)
        self.ax[0].set_xlabel('Time [ps]')
        self.ax[0].set_ylabel('Power [W]')
        self.ax[0].set_title('STUD Pulse')
        self.ax[1].set_xlabel('omega')
        self.ax[1].set_ylabel('Power [W]')
        self.ax[1].set_title('STUD Pulse')
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self.input_tab)
        self.canvas1.get_tk_widget().grid(row=0,column = 6, columnspan=2, rowspan = max(row))
        
        # Initialize default data
        
        #self.set_all_default()
        for i in range(len(key)):   
            self._set_value(key[i],parameter[key[i]]["default"])                                       
            
        self.update_values()
        
        # Initial plot
        #self.update_initial_plot()
        
        # Simulation button
        self.run_button = ttk.Button(self.input_tab, text = "Run Simulation")#, command = lambda:self.run_the_simulation(), style = 'W.TButton')
        self.run_button.bind('<Button>', self.run_the_simulation)
        self.run_button.bind('<Button>', self.plot_2D, add = '+')
        self.run_button.bind('<Button>', self.setup_zslider, add = '+')# Put in variable connections
        self.run_button.bind('<Button>', self.setup_compensator, add = '+')# Put in variable connections
        self.run_button.bind('<Button>', self.switch_tab, add = '+')# Switch tab
        self.run_button.grid(row = 0, column= 0, columnspan=2, padx = 10)
        self.run_the_simulation()
        self.plot_2D()
        self.setup_zslider()
        self.setup_compensator()
        self.switch_tab()
        
        # Toggle button
        self.mode_index = 1
        self.tog_button = ttk.Button(self.input_tab, text = "Switch to Numerical parameters")
        self.tog_button.bind('<Button>', self.toggle)
        self.tog_button.grid(row = 0, column= 2, columnspan=2, padx = 10)
        
            
    def _set_value(self, var, val):
        if parameter[var]["Type"] == "Entry":
            if not parameter[var][self.mode]:
                parameter[var]["Widget"].configure(state = 'normal')
            parameter[var]["Widget"].delete(0,tk.END)
            try:
                int(val)
                parameter[var]["Widget"].insert(0,val)
            except ValueError:
                parameter[var]["Widget"].insert(0,parameter[var]["default"])
            if not parameter[var][self.mode]:
                parameter[var]["Widget"].configure(state = 'readonly')
        elif parameter[var]["Type"] == "Spinbox":
            parameter[var]["Widget"].set(val)
            
        elif parameter[var]["Type"] == "Checkbutton":
            parameter[var]["StringVar"].set(val)
    
    def update_values(self, _=None):
        
        # Get values needed to update STUD Pulse
        if self.mode == "N_edit":
            self._set_value("lambda0",        self.calculate_value("lambda0", self.mode))
            self._set_value("beta1",          self.calculate_value("beta1", self.mode))
            self._set_value("beta2",          self.calculate_value("beta2", self.mode))
            self._set_value("beta3",          self.calculate_value("beta3", self.mode))
            self._set_value("T0",             self.calculate_value("T0", self.mode))
            self._set_value("LD3",            self.calculate_value("LD3", self.mode))
            self._set_value("gamma",          self.calculate_value("gamma", self.mode))
            self._set_value("Tmax_ps",        self.calculate_value("Tmax_ps", self.mode))    
            self._set_value("fiber_length_m", self.calculate_value("fiber_length_m", self.mode))  
            self._set_value("dt",             self.calculate_value("dt", self.mode)) 
            self._set_value("T_best",         self.calculate_value("T_best", self.mode))
            self._set_value("LD",             self.calculate_value("LD", self.mode))
            self._set_value("LNL",            self.calculate_value("LNL", self.mode))
            
            
            # Generate STUD
            self.U0, self.T0, self.P0, self.spike_A, self.spike_w, self.spike_t = self.get_stud()
            print( "self.T0 = " + str(self.T0))
            self.UU0 = abs(self.U0)**2
            
            self._set_value("P0",             self.calculate_value("P0", self.mode))     
            self._set_value("Energy",         self.calculate_value("Energy", self.mode))   
            self._set_value("LD",                   self.calculate_value("LD", self.mode))
            
            
        elif self.mode == "P_edit":
            self._set_value("lambda0",           self.calculate_value("lambda0", self.mode))
            self._set_value("beta1",           self.calculate_value("beta1", self.mode))
            self._set_value("beta2",           self.calculate_value("beta2", self.mode))
            self._set_value("beta3",           self.calculate_value("beta3", self.mode))
            self._set_value("Tmax",           self.calculate_value("Tmax", self.mode))   
            self._set_value("gamma",          self.calculate_value("gamma", self.mode))
            self._set_value("dt",             self.calculate_value("dt", self.mode)) 
            self._set_value("C",              self.calculate_value("C", self.mode))
            self._set_value("LNL",            self.calculate_value("LNL", self.mode))
            
            # Generate STUD
            self.U0, self.T0, self.P0, self.spike_A, self.spike_w, self.spike_t = self.get_stud()
            self.UU0 = abs(self.U0)**2
            
            self._set_value("P0",             self.P0)
            self._set_value("T0",             self.T0)
            self._set_value("LD",             self.calculate_value("LD", self.mode))
            self._set_value("LD3",            self.calculate_value("LD3", self.mode))
            self._set_value("LNL",            self.calculate_value("LNL", self.mode))
            self._set_value("N",              self.calculate_value("N", self.mode))
            self._set_value("fiber_length",   self.calculate_value("fiber_length", self.mode)) 
        
        
        self._set_value("Leff",                   self.calculate_value("Leff", self.mode))
        self._set_value("step_num",             self.calculate_value("step_num", self.mode))
        self._set_value("deltaz",               self.calculate_value("deltaz", self.mode))
        self._set_value("s",               self.calculate_value("s", self.mode))
        self._set_value("ld3",              self.calculate_value("ld3", self.mode))
        self._set_value("lnl",              self.calculate_value("lnl", self.mode))
        
        self.update_initial_plot()
            
            
        #self._set_value(var,self.calculate_value(var))
            
    def calculate_value(self, var, mode):
        try:
            if   var == "Tmax_ps":
                T0   = float(parameter["T0"]["Widget"].get())
                Tmax = float(parameter["Tmax"]["Widget"].get())
                val  = Tmax * T0 
                
            elif var == "Tmax":
                T0   = float(parameter["T0"]["Widget"].get())
                Tmax = float(parameter["Tmax_ps"]["Widget"].get())
                val  = Tmax / T0
                
            elif var == "T0":
                t_on = float(parameter["t_on"]["Widget"].get())
                order = float(parameter["order"]["Widget"].get())
                val = t_on /(2 * np.log(2)**(1/(2* order))) #Convert FWHM power to sigma in E-field  
                
            elif var == "P0":
                if mode == "N_edit":
                    LNL = float(parameter["LNL"]["Widget"].get())
                    gamma = float(parameter["gamma"]["Widget"].get())
                    val = 1 / (LNL * gamma) 
                elif mode == "P_edit":
                    val = self.P0 #This value is caluclated in the update_initial_plot function which always runs before calculate_value
                    
            elif var == "Energy":
                if mode == "N_edit":
                    P0 = float(parameter['P0']["Widget"].get())
                    dt = float(parameter['dt']["Widget"].get())
                    val = sum(self.UU0) * P0 * dt * 1E-3#Energy is integrated normalized UU *P0
                
            elif var == "N":  #I am not yet conserned with cases where LD = inf
                LD = float(parameter["LD"]["Widget"].get())
                LNL = float(parameter["LNL"]["Widget"].get())
                val = np.sqrt(LD/LNL)    # N^2 = gamma P0 T0^2 / |beta2|  Soliton Number
                
            elif var == "LD":
                T0    = float(parameter["T0"]["Widget"].get())
                beta2 = float(parameter["beta2"]["Widget"].get())
                if beta2 == 0:
                    val = 'NA'
                else:
                    val = T0**2/abs(beta2)  # m 2nd order Dispersion length
                    
            elif var == "LD3":
                T0    = float(parameter["T0"]["Widget"].get())
                beta3 = float(parameter["beta3"]["Widget"].get())
                if beta3 == 0:
                    val = 'NA'
                else:
                    val   = T0**3/abs(beta3) # m 3rd order Dispersion length
            
            elif var == "ld3":
                LD = float(parameter["LD"]["Widget"].get())
                LD3 = float(parameter["LD3"]["Widget"].get())
                val = LD3/LD
                    
            elif var == "LNL":
                if mode == "N_edit":
                    N  = float(parameter["N"]["Widget"].get())
                    LD = float(parameter["LD"]["Widget"].get())
                    val = LD / N**2
                    
                elif mode == "P_edit":
                    P0    = float(parameter["P0"]["Widget"].get())
                    gamma = float(parameter["gamma"]["Widget"].get())
                    if gamma == 0 or P0 == 0:
                        val = 'NA'
                    else:
                        val = 1/(gamma * P0)   # m Nonlinear Dispersion length
                        
            elif var == "lnl":
                LD = float(parameter["LD"]["Widget"].get())
                LNL = float(parameter["LD3"]["Widget"].get())
                val = LNL/LD

            elif var == "gamma":
                lambda0 = float(parameter["lambda0"]["Widget"].get())
                # Get n2, I don't have a good model for this, and I doubt that this will change the results much
                # If you want better n2 values, you will need to model https://opg.optica.org/ao/fulltext.cfm?uri=ao-37-3-546&id=63143
                # Should be okay for wavelengths below 600 nm
                n2 = 2.74E-20 # [m^2/W]
        
                #self.c =  2.99792458E-4 #speed of light m/ps 
                self.omega0 = 2 * pi * self.c/(lambda0 * 1E-9) #Should be [1/ps]  
                w0 = float(parameter["w0"]["Widget"].get()) * 1E-6 #[m]
                Aeff = np.pi * (w0)**2 #m^2
                val = (self.omega0*n2/(self.c*Aeff))#1/(W m)  # [1/ps][m^2/W][ps/m][1/m^2]
                
            elif var == 'dt':
                Tmax_ps = float(parameter["Tmax_ps"]["Widget"].get())
                nt      = float(parameter["nt"]["Widget"].get())
                val = (2 * Tmax_ps)/nt
                
            elif var == "deltaz":
                step_num      = self.calculate_value("step_num", mode)
                fiber_length_m  = float(parameter["fiber_length_m"]["Widget"].get())
                #LD            = float(parameter["step_num"]["Widget"].get())
                val = fiber_length_m / step_num
                
            elif var == "fiber_length":
                LD             = parameter["LD"]["Widget"].get()
                LD3            = parameter["LD3"]["Widget"].get()
                LNL            = parameter["LNL"]["Widget"].get()
                fiber_length_m = float(parameter["fiber_length_m"]["Widget"].get())
                if LD != 'NA':
                    val = fiber_length_m / float(LD)
                elif LD3 != 'NA':
                    val = fiber_length_m / float(LD3)
                elif LNL != 'NA':
                    val = fiber_length_m / float(LNL)
                else:
                    val = 'NA'
            
            elif var == "fiber_length_m":
                LD             = parameter["LD"]["Widget"].get()
                LD3            = parameter["LD3"]["Widget"].get()
                LNL            = parameter["LNL"]["Widget"].get()
                fiber_length   = float(parameter["fiber_length"]["Widget"].get())
                if LD != 'NA':
                    val = fiber_length * float(LD)
                elif LD3 != 'NA':
                    val = fiber_length * float(LD3)
                elif LNL != 'NA':
                    val = fiber_length * float(LNL)
                else:
                    val = parameter['fiber_length_m']['default']
            
            elif var == "step_num":
                fiber_length   = float(parameter["fiber_length"]["Widget"].get())
                N = float(parameter["N"]["Widget"].get())  
                if N < 1:
                    step_num = round(20 * fiber_length)
                else: 
                    step_num = round(20 * fiber_length * N ** 2) #number of z steps
                if step_num < 100:
                    step_num = 100
                val = step_num
                
            elif var == "s":
                T0  = float(parameter["T0"]["Widget"].get())
                val = 1/(self.omega0 * T0)   # Self steepening parameter
                
            elif var == "C":
                try:
                    C = float(parameter["C"]["Widget"].get()) 
                    if C < 0:
                        C_sign = -1
                    else:
                        C_sign = 1
                except:
                    C_sign = 1
                t_on = float(parameter["t_on"]["Widget"].get()) 
                T_best = float(parameter["T_best"]["Widget"].get()) 
                val = C_sign * (np.sqrt((t_on/T_best)**2-1))          # Chirp parameter #For Gaussian with FWHM = True note that T_best_power_FWHM / T0_power_FWHM = T0/T_best
            
            elif var == "T_best":
                t_on = float(parameter["t_on"]["Widget"].get()) 
                C = float(parameter["C"]["Widget"].get()) 
                val = t_on / np.sqrt(C**2 +1)         # Chirp parameter #For Gaussian with FWHM = True note that T_best_power_FWHM / T0_power_FWHM = T0/T_best
            
            elif var == "Leff":
                alpha = float(parameter["alpha"]["Widget"].get()) 
                fiber_length_m = float(parameter["fiber_length_m"]["Widget"].get()) 
                try:
                    val = (1 - np.exp(-alpha * fiber_length_m)) / alpha
                except:
                    val = fiber_length_m
                
            elif var == "lambda0":
                val = float(parameter[var]["Widget"].get())
            
            elif var == "beta1": 
                lambda0 = float(parameter["lambda0"]["Widget"].get())
                val = get_beta1(lambda0)
            
            elif var == "beta2": 
                lambda0 = float(parameter["lambda0"]["Widget"].get())
                val = get_beta2(lambda0)
                
            elif var == "beta3":
                lambda0 = float(parameter["lambda0"]["Widget"].get())
                val = get_beta3(lambda0)
                
            elif var == "alpha":
                val = float(parameter["alpha"]["Widget"].get())
                
            else:
                print(var + ' is not solved')
                
            
        except:
            return ""
        try:
            print(var + ' = ' + str(val))
            return val
        except:
            print(str(var) + ' not found')
            return ""
        

    def get_stud(self, _=None):
        STUD_pulse_parameters = {}
        for i in range(len(key)):
            if parameter[key[i]]["Type"] == "Entry":

                try:
                    STUD_pulse_parameters[key[i]] = float(parameter[key[i]]["Widget"].get())
                except ValueError:
                    print("ValueError for " + key[i] + " while getting STUD")
                    #STUD_pulse_parameters[key[i]] = float(parameter[key[i]]["default"])
                    #self._set_value(key[i],parameter[key[i]]["default"])
                    
            elif parameter[key[i]]["Type"] == "Spinbox":
                STUD_pulse_parameters[key[i]] = int(float(parameter[key[i]]["Widget"].get()))
            
            elif parameter[key[i]]["Type"] == "Checkbutton":
                STUD_pulse_parameters[key[i]] = bool(int(float(parameter[key[i]]["StringVar"].get())))
        
        nt = int(float(parameter["nt"]["Widget"].get()))
        #Tmax_ps = float(parameter["Tmax_ps"]["Widget"].get())
        
        dt = float(parameter["dt"]["Widget"].get())
        self.t = np.array(range(int(-nt/2),int(nt/2))) * dt     #Time array
        
        STUD_pulse_parameters["jitter_A"] = 0   # Bedros does not want this, so I bury the functionality for now
        STUD_pulse_parameters["t"] = self.t

        self.U0, self.T0, self.P0, self.spike_A, self.spike_w, self.spike_t = STUD_pulse(**STUD_pulse_parameters)

        #parameter["P0"]["default"] = self.P0
        #self._set_value("P0",self.P0)
        
        return self.U0, self.T0, self.P0, self.spike_A, self.spike_w, self.spike_t
    
    def update_initial_plot(self, _=None):

        Tmax_ps  = float(parameter['Tmax_ps']['Widget'].get())
        t_center = float(parameter['t_center']['Widget'].get())
        nt       = int(float(parameter['nt']['Widget'].get()))
        spect    = abs(fftshift(ifft(self.U0)))**2
        spect    = spect/np.nanmax(spect)
        Tmax = Tmax_ps / self.T0
        
        self.tlim = np.array([self.spike_t[0] - self.spike_w[0]*20, self.spike_t[-1] + self.spike_w[-1]*20]) + t_center
        
        self.omega = fftshift(range(int(-nt/2),int(nt/2)))*(pi/Tmax)
        self.freq = fftshift(self.omega)/(2*pi)
        
        self.ax[0].clear()
        self.ax[0].set_xlabel('Time [ps]')
        self.ax[0].set_ylabel('Power [W]')
        self.ax[0].set_title('STUD Pulse P[T]')
        self.ax[0].plot(self.t, self.UU0 * self.P0)
        self.ax[0].set(xlim = self.tlim)
        
        self.ax[1].clear()
        self.ax[1].set_xlabel('$T_0$ δω')
        self.ax[1].set_ylabel('Power [W]')
        self.ax[1].set_title('STUD Pulse P[ω]')
        self.ax[1].plot(self.freq, spect)
        
        self.canvas1.draw()
        
    def toggle(self,_=None):
        #Switch between Numerical entry and Physical entry modes
        self.mode_index = not self.mode_index
        self.mode_options = ["N_edit", "P_edit"]
        print("Mode is " + self.mode_options[self.mode_index] )
        self.mode = self.mode_options[self.mode_index] 
        for i in range(len(key)):
            if parameter[key[i]][self.mode]:
                parameter[key[i]]["Widget"].config(state = 'normal')
            else:
                parameter[key[i]]["Widget"].config(state = 'readonly')
        if self.mode == "N_edit":
            self.tog_button.config(text = "Switch to Physical parameters")
        elif self.mode == "P_edit":
            self.tog_button.config(text = "Switch to Numerical parameters")
            
        
    def run_the_simulation(self, _=None):#U0, beta2, beta3, N, omega, alpha, LD, LD3, deltaz,step_num, nt, D, s, tauR, tau, steps, t):
        #Unpack variables
        print("running the simulation")
        U0       = self.U0
        omega    = self.omega
        T0       = float(parameter["T0"      ]["Widget"].get())
        beta2    = float(parameter["beta2"   ]["Widget"].get())
        beta3    = float(parameter["beta3"   ]["Widget"].get())
        N        = float(parameter["N"       ]["Widget"].get())
        alpha    = float(parameter["alpha"   ]["Widget"].get())
        LD       = float(parameter["LD"      ]["Widget"].get())
        LD3      = float(parameter["LD3"     ]["Widget"].get())
        deltaz   = float(parameter["deltaz"  ]["Widget"].get())
        deltaZ  = deltaz/LD
        step_num =   int(float(parameter["step_num"]["Widget"].get()))
        nt       =   int(float(parameter["nt"      ]["Widget"].get()))
        D = 1
        s        = float(parameter["s"       ]["Widget"].get())
        tauR     = float(parameter["TR"      ]["Widget"].get())/T0
        tau      = self.t/T0
        #coupled  = float(parameter["coupled" ]["StringVar"].get())
        
        steps = range(0,step_num)
        
        if LD3 ==0 or beta3 == 0:
            dispersion = np.exp(D * (0.5j * np.sign(beta2) * omega**2 - alpha * LD / 2) * deltaZ) #Preloaded phase factor
        else:
            dispersion = np.exp(D * (0.5j * np.sign(beta2) * omega**2 + 1j/6 * np.sign(beta3) * LD/LD3 * omega**3 - alpha * LD / 2) * deltaZ) #Preloaded phase factor

        
        hhz = 1j*N**2*deltaZ
        
        if False: #coupled:
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
                    break
                
            U[:,-1,:] = temp * np.exp(coupled_NonLinear(U[:,-2,:],  s, tauR, tau) * hhz/2) 
        else:
            U = np.zeros([step_num,nt],dtype=np.complex_) #Initialize U
            U[0,:] = U0
            temp = U[0,:]*np.exp(NonLinear(U[0,:], s, tauR, tau) * hhz/2)
            
            #Main Loop
            print("made it to main loop")
            for n in steps[1:-1]:
                f_temp = ifft(temp)*dispersion
                U[n,:] = fft(f_temp)
                temp = U[n,:] * np.exp(NonLinear(U[n,:], s, tauR, tau) * hhz)

                if n%150 == 0:
                    print('Step ' + str(n) +'/'+str(step_num))
                    
                if np.isnan(np.nanmax(U[n,:])):
                    print("U is nan")
                    break
            U[-1,:] = temp * np.exp(NonLinear(U[-2,:],  s, tauR, tau) * hhz/2)  
            
        # Package all the useful variables
        T0 = float(parameter["T0"]["Widget"].get())
        dt = float(parameter["dt"]["Widget"].get())
        P0 = float(parameter["P0"]["Widget"].get())
        self.U = U
        self.UU = abs(U)**2
        #self.phi      = np.unwrap(np.arctan2(np.imag(U),np.real(U)))
        #self.exact_dw = [beta2 * n * deltaZ *tau / ((1 + (n * deltaxi)**2))  for n in range(len(U[:,0]))]
        self.spect    = [abs(fftshift(ifft(U[n,:])))**2                     for n in range(len(U[:,0]))]
        self.spect    = [self.spect[n]/np.nanmax(self.spect[n])             for n in range(len(U[:,0]))]
        #self.dw       = [-np.gradient(self.phi[n,:],dt/T0)                  for n in range(len(U[:,0]))]
        self.Energy   = [sum(self.UU[n,:]) * dt * P0                        for n in range(len(U[:,0]))]  #E = int(P(t)) = P0 * int(UU*(t))
        self.STD_freq = [STD(self.freq, self.spect[n])                      for n in range(len(U[:,0]))]
        self.STD_t    = [STD( self.t, self.UU[n,:])                              for n in range(len(U[:,0]))]
        #for n in range(len(U[:,0])):
        #    self.dw[n][self.UU[n,:] < np.max(self.UU[n,:]) * 0.001] = np.nan
    
    def plot_2D(self,_=None):
        # Initialize plots
        t_lim  = np.array([-1,1])*self.STD_t[-1] * 4
        if t_lim[1]> self.t[-1] or any(np.isnan(t_lim)):
            t_lim = [self.t[0],self.t[-1]]
        self.t_lim = t_lim
            
        
        self.fig2, self.ax2 = plt.subplots(figsize=(4, 6))
        plt.subplots_adjust(left= 0.2, right = 0.85)
        self.ax2.set_xlabel('Time [ps]')
        #self.fig2.set_dpi()
        self.ax2.set_ylabel('z [m]')
        self.ax2.set_title('STUD Pulse')
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master = self.output_tab)
        self.canvas2.get_tk_widget().grid(row=0,column = 0, columnspan = 2)
        image_2_plot = self.UU.T[np.logical_and(self.t > self.t_lim[0], self.t < self.t_lim[1])]
        image_2_plot = image_2_plot.T
        p_compression = int(len(image_2_plot[0])/400)+1
        q_compression = int(len(image_2_plot)/400)+1
        image_2_plot = [[image_2_plot[q][p] for p in range(0,len(image_2_plot[0]),p_compression)] for q in range(0,len(image_2_plot),q_compression)]  # Compress image for faster rendering
        fiber_length_m = float(parameter["fiber_length_m"]["Widget"].get())
        self.ax2.imshow(image_2_plot, interpolation='bilinear', cmap=cm.RdYlGn,
               origin='lower', extent=[self.t_lim[0], self.t_lim[1], 0, fiber_length_m],
               vmax=abs(self.UU).max(), vmin=-abs(self.UU).max(), aspect = 'auto')
        x = self.t_lim
        y = [0,0]
        self.image_line = self.ax2.plot(x,y,'c')[0]
        
    def setup_zslider(self,_=None):
        # Initialize plots
        #Get plot x limits
        freq_lim = np.array([-1,1])*self.STD_freq[-1] * 5 /self.T0
        if freq_lim[1]> self.freq[-1] or any(np.isnan(freq_lim)):
            freq_lim = [self.freq[0]/self.T0,self.freq[-1]/self.T0]
        
        #Set up time slice plot at z
        self.fig3, self.ax3 = plt.subplots(figsize=(4, 6), nrows = 2)
        self.ax3[0].set_xlabel('T [ps]')
        self.ax3[0].set_ylabel('Power/P0')
        self.ax3[0].set_title('STUD Pulse')
        self.ax3[0].set_xlim(self.t_lim)
        fig3_text0 = "σₜ = " + '{:.2e}'.format(self.STD_t[0]) + " ps"
        self.fig3_text0_holder = self.fig3.text(0.15,0.8,fig3_text0)
        self.TPlot_line = self.ax3[0].plot(self.t,self.UU[0])[0]
        
        #Set up spectrum slice at z
        self.ax3[1].set_xlabel('Δω [THz]')
        self.ax3[1].set_ylabel('Power [Normalized]')
        self.ax3[1].set_xlim(freq_lim)
        fig3_text1 = "σω = " + '{:.2e}'.format(self.STD_freq[0]) + " [THz]"
        self.fig3_text1_holder = self.fig3.text(0.15,0.4,fig3_text1)
        self.fPlot_line = self.ax3[1].plot(self.freq/self.T0,self.spect[0])[0]
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master = self.output_tab)
        self.canvas3.get_tk_widget().grid(row=0,column = 2, columnspan = 2)
        
        # Set up z slider
        self.ztext = 'z = 0 [m]'
        self.zslider_label = ttk.Label(self.output_tab, text = self.ztext)
        self.zslider_label.grid(row = 1, column = 2)
        step_num =   int(parameter["step_num"]["Widget"].get())
        self.zslider= ttk.Scale(self.output_tab, from_ = 0, to = step_num-1)
        self.zslider.bind("<Motion>",self.zslider_activate)
        self.zslider.grid(row=1,column = 3)
        
        
    def zslider_activate(self, _=None):
        self.n = int(self.zslider.get())
        deltaz = float(parameter["deltaz"  ]["Widget"].get())
        z = self.n * deltaz
        self.ztext = 'z = ' + '{:.3f}'.format(z) + ' [m]'
        self.zslider_label.config(text = self.ztext)
        self.TPlot_line.set_ydata(self.UU[self.n])
        self.fPlot_line.set_ydata(self.spect[self.n])
        self.image_line.set_ydata([z, z])
        
        fig3_text0 = "σₜ = " + '{:.2f}'.format(self.STD_t[self.n]) + " [ps]"
        fig3_text1 = "σω = " + '{:.2f}'.format(self.STD_freq[self.n]/self.T0) + " [THz]"
        
        self.fig3_text0_holder.set_text(fig3_text0 )
        self.fig3_text1_holder.set_text(fig3_text1 )
        self.canvas2.draw() #  Draw Line on 2D plot
        self.canvas3.draw() # Draw slice plots
        
    def setup_compensator(self,_=None):
        #Set up figure
        self.fig4, self.ax4 = plt.subplots(figsize=(4, 6))
        plt.subplots_adjust(left= 0.1, right = 0.95)
        self.ax4.set_xlabel('T [ps]')
        self.ax4.set_ylabel('Power/P0')
        self.ax4.set_title('Recompression')
        self.ax4.set_xlim(self.t_lim)
        self.dPlot_line = self.ax4.plot(self.t,self.UU[-1])[0]
        self.canvas4 = FigureCanvasTkAgg(self.fig4, master = self.output_tab)
        self.canvas4.get_tk_widget().grid(row=0,column = 4, columnspan = 2)
        
        #Characterize initial spike train
        spike_w, spike_A, spike_t_new, spike_dt, spike_E, STUD_text = characterize_STUD(self.spike_t, self.UU[-1], self.t, self.T0, self.P0)
        
        self.fig4.textlabel = self.fig4.text(0.15,0.5,STUD_text)
        
        # Get initial total dispersion
        beta2  = float(parameter["beta2"]["Widget"].get())
        beta3  = float(parameter["beta3"]["Widget"].get())
        #deltaz = float(parameter["deltaz"]["Widget"].get())
        #step_num = int(parameter["step_num"]["Widget"].get())
        fiber_length_m = float(parameter["fiber_length_m"]["Widget"].get())
        self.b2_0 = -fiber_length_m * beta2 /self.T0**2#-deltaz * (step_num-1) * beta2
        self.b3_0 = -fiber_length_m * beta3 / self.T0**3#-deltaz * (step_num-1) * beta3
        
        # Set up z_out button Which sets the right column data to be as if it came after z data.
        self.zout_label_text = r'z_out = ' + '{:.2f}'.format(fiber_length_m) + ' [m]'
        self.zout_label = ttk.Label(self.output_tab, text = self.zout_label_text )
        self.zout_label.grid(row = 1, column = 4)
        
        self.zout_button = ttk.Button(self.output_tab, text = r'set $z_{out}$')
        self.zout_button.grid(row = 1, column = 5)
        self.zout_button.bind("<Button>", lambda event: self.update_zout())
        
        
        # Set up b2 and b3 sliders which adjusts the right column output to 
        self.b2slider = ttk.Scale(self.output_tab, from_ = -10 * abs(self.b2_0 + float(parameter["C"]["Widget"].get())), to = 5 * abs(self.b2_0) + float(parameter["C"]["Widget"].get()))
        self.b3slider = ttk.Scale(self.output_tab, from_ = -5 * abs(self.b3_0), to = 5 * abs(self.b3_0))
        
        self.b2slider.grid(row=2, column = 5)
        self.b3slider.grid(row=3, column = 5)
        
        self.b2slider.set(self.b2_0)
        self.b3slider.set(self.b3_0)
        
        b2_text = "GDD: " + format(self.b2_0, ".2e") + " [ps^2]" 
        b3_text = "TOD: " + format(self.b3_0, ".2e") + " [ps^3]" 
        self.b2slider_label = ttk.Label(self.output_tab,text = b2_text)
        self.b3slider_label = ttk.Label(self.output_tab,text = b3_text)
        
        self.b2slider.bind("<Motion>", lambda event: self.dslider(event, b2 = self.b2slider.get(),b3 = self.b3slider.get()))
        self.b3slider.bind("<Motion>", lambda event: self.dslider(event, b2 = self.b2slider.get(),b3 = self.b3slider.get()))
        
        self.b2slider_label.grid(row=2, column = 4)
        self.b3slider_label.grid(row=3, column = 4)
        
        #Setup dispersion reset buttons
        self.linear_button = ttk.Button(self.output_tab, text = "Linear reset")
        self.linear_button.bind("<Button>",lambda event: self.dslider(event, b2 = self.b2_0,b3 = self.b3_0))
        self.linear_button.grid(row = 5, column = 4)
        
        self.optimize_button = ttk.Button(self.output_tab, text = "Optimize")
        self.optimize_button.bind("<Button>",lambda event: self.optimize(event, b2 = self.b2_0,b3 = self.b3_0))
        self.optimize_button.grid(row = 5, column = 5)
    
    def update_zout(self,_=None):
        self.n = int(self.zslider.get())
        self.zout_n = self.n
        deltaz = float(parameter["deltaz"  ]["Widget"].get())
        self.zout_label_text = r'$z_{out}$ = ' + '{:.3f}'.format(self.zout_n * deltaz) + ' [m]'
        self.zout_label.config(text = self.zout_label_text)
    
        beta2 = float(parameter["beta2"]["Widget"].get())
        beta3 = float(parameter["beta3"]["Widget"].get())
        self.b2_0 = -deltaz * self.n * beta2 /self.T0**2  #-deltaz * (step_num-1) * beta2
        self.b3_0 = -deltaz * self.n * beta3 / self.T0**3 #-deltaz * (step_num-1) * beta3
        self.dslider()
        
        
    
    def apply_dispersion(self, b2, b3, f, omega):
        dispersion = np.exp((0.5j * b2 * omega**2  + 1j/6 * b3 * omega**3)) #Phase factor
        f_temp = ifft(f)*dispersion
        return fft(f_temp)
    
    def dslider(self, _=None, b2 = 0, b3 = 0):
        self.b2slider.set(b2)
        self.b3slider.set(b3)
        b2_text = "b2: " + format(b2, ".2e") + " [ps^2]" 
        b3_text = "b3: " + format(b3, ".2e") + " [ps^3]" 
        self.b2slider_label.config(text = b2_text)
        self.b3slider_label.config(text = b3_text)
        y = self.apply_dispersion(b2, b3, self.U[self.n], self.omega)
        y = abs(y)**2
        self.dPlot_line.set_ydata(y)
        if self.ax4.get_ylim()[1] < max(y):
            self.ax4.set_ylim([0,max(y)*1.2])
            
        #Update text
        spike_w, spike_A, spike_t_new, spike_dt, spike_E, STUD_text = characterize_STUD(self.spike_t, y, self.t, self.T0, self.P0)
        self.fig4.textlabel.set_text(STUD_text)
        self.canvas4.draw()
    
        
    def function_to_minimize(self, vars):
        b2, b3 = vars
        temp = self.apply_dispersion(b2,b3,self.U[self.n], self.omega)
        return -max(abs(temp)**2)
        
    def optimize(self,_=None, b2 = 0, b3 = 0):
        result = minimize(self.function_to_minimize, [b2, b3], method = 'Nelder-Mead')
        [b2_max, b3_max] = result.x
        self.dslider(b2 = b2_max, b3 = b3_max)
        
    def switch_tab(self,_=None, b2 = 0, b3 = 0):    
        self.notebook.select(1)
    
    def quit_me(self,_=None):
        print('quit')
        self.input_tab.quit()
        self.input_tab.destroy()
        self.output_tab.quit()
        self.output_tab.destroy()
        self.root.quit()
        self.root.destroy()
    
try:
    root = tk.Tk()

    #Window Settings
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width *0.86)
    window_height = int(screen_height * 0.85)
    center_x = int(screen_width/2 - window_width / 2)
    center_y = int(screen_height/2 - window_height / 2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    app = STUD_GUI(root)
    
    root.mainloop()
    
except(KeyboardInterrupt, SystemExit):
    print("Loop finished")
    #root.exit()
    #sys.modules[__name__].__dict__.clear() #This lets you run the tkinter a second time.
    sys.exit()
