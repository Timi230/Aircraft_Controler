from atm_std import get_cte_atm
import numpy as np
import matplotlib.pyplot as plt
import control 
from control.matlab import *
import math as m
from scipy.interpolate import interp1d
from pylab import *
from sisopy31 import *





#---------------------------------------------------------------------

#Aircraft parameters
g = 9.788
mass = 8400                      # mass of the aircraft: kg
c = 0.52                         # aircraft centering
S = 34                           # surface: m^2
r_g = 2.65                       # radius of gyrations: m
l_ref = 5.24                     # reference length: m
l_t = (3/2) * l_ref              # total length: m
Mach = 1.49                      # number of Mach
altitude_ft = 10085               # altitude oof the aircraft: ft
altitude_m = altitude_ft * 0.3048 # conversion of the altitude: m
Iyy = mass * r_g**2              # inertial tensor in y axis

hgeo,rho,V_sound = get_cte_atm(altitude_m)   # gravity, density and speed of sound at altitude z
print(f"altitude géopotentielle: {hgeo}, density: {rho}, speed of sound: {V_sound} \n")

#Aircraft characteristics
Cx0 = 0.0325                     # drag coefficient for null incidence
Cz_alpha = 2.4                   # lift gradient coefficient WRT alpha
Czδm = 0.48                      # lift gradient coefficient WRT δm
δm0 = -0.002                     # equilibirum fin deflection for null lift
alpha0 = 0.008                   # incidence for null lift/fin deflection
f = 0.609                        # aerodynamic center of body and wings: meter
fδ = 0.9                         # aerodynamic center of fins: meter
k = 0.42                         # polar coefficient
Cmq = -0.32                      # damping coefficient 
V_eq = Mach * V_sound            # speed of the aircraft at the equ point
Q = (1/2) * rho * V_eq**2        # dynamic pressure at V_eq speed
Xf = - (f *l_t)                  # position of Xf on the aircraft: m
Xg = - (c * l_t)                 # position of Xg on the aircraft: m
Xfδ = - (fδ * l_t)               # position of Xfdelta on the aircraft: m
X = Xf-Xg                        # X
Y = Xfδ - Xg                     # Y


alpha_eq0   = 0                     # Initialisation de l'angle d'incidence au point d'équilibre : radians
Fp_xeq0     = 0                     # Initialisation de la première force au point d'équilibre
epsilon     = 10**-6                # Précision pour convergence
max_iter    = 1000                  # Limite maximale d'itérations pour prévenir les boucles infinies

# Initialisation des variables
alpha_eqold = alpha_eq0        
alpha_eq    = 0.1                   # Valeur initiale pour alpha_eq (ajustée pour aider la convergence)
Fp_xeq      = Fp_xeq0               # Valeur initiale pour Fp_xeq

i = 0

#---------------------------------------------------------------------
for ite in range(max_iter):
    print(alpha_eq)
    # Calculate equilibrium coefficients
    C_Z_eq = (mass*g - Fp_xeq0*np.sin(alpha_eq)) / (Q*S)
    C_X_eq = Cx0 + k*(C_Z_eq*C_Z_eq)
    C_X_delta_m = 2*k*C_Z_eq*Czδm
    
    # Update control surface deflection
    delta_m_eq = δm0 - (((C_X_eq*np.sin(alpha_eq) + C_Z_eq*np.cos(alpha_eq))*X) / ((C_X_delta_m*np.sin(alpha_eq) + Czδm*np.cos(alpha_eq))*(Y-X)))

    # Update angle of attack and force along x-axis
    alpha_new = alpha_eq0 + (C_Z_eq / Cz_alpha) - (Czδm / Cz_alpha)*delta_m_eq
    Fp_xeq0_new = (Q*S*C_X_eq)/np.cos(alpha_new)
    
    # Update alpha_eq for next iteration and force along x-axis
    alpha_eq = alpha_new
    Fp_xeq0 = Fp_xeq0_new
    
# Check for convergence
    if abs(alpha_new - alpha_eq) < epsilon:
        print("Convergence achieved")
        break

    else:
        print("Convergence not achieved within the maximum number of iterations")
        
# Results
print(f"Equilibrium angle: {alpha_eq}")