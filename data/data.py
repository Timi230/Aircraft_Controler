from outils.atm_std import get_cte_atm
import math as m
#--------------------------------------------------------------------

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