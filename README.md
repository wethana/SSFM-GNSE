# SSFM-GNSE
Basic Split-Step Fourier Method code (Python) for solving the Generalized Nonlinear Schrodinger Equation in PM fibers

This code was written out of necessity as part of Ethan Welch's PhD thesis "Using Spike Trains of Uneven Duration and Delay (STUD pulses) to Mitigate Laser-Plasma Instabilities." 

The SSFM method is based off of G. P. Agrawal, Nonlinear Fiber Optics, sixth edition ed. (Academic Press, Cambridge, MA, 2019) [1]. It models the GMSE 

$$
 \frac{\partial U}{\partial Z} + \frac{\alpha}{2} L_DU + sign(\beta_2) \frac{1}{2}\frac{\partial^2 U}{\partial \tau^2} + sign(\beta_3) \frac{1}{6} \frac{L_{D2}}{L_{D3}} \frac{\partial^3 U}{\partial \tau^3}
    =i N^2\left(|U|^2U + is \frac{\partial}{\partial \tau}(|U|^2U - \tau_R U \frac{\partial |U|^2}{\partial \tau})\right)
$$

The nonlinear solver is based on [2].

Start with pulse_propagation_minimal_example.py which is a self contained GNSE solver. It uses the same engine as the other examples, and should be easily modifiable to suite your needs. 

Pulse_propagation_GUI3.py offers a more advanced GUI to modify the input parameters and visualize the pulse evolution. It also needs Pulse_propagation_functions.py in the same directory to run. 

[1] G. P. Agrawal, Nonlinear Fiber Optics, sixth edition ed. (Academic Press, Cambridge, MA, 2019)

[2] R. Deiterding and S. W. Poole, “Robust split-step Fourier methods for simulating
the propagation of ultra-short pulses in single- and two-mode optical communication
fibers”, 2015.
