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


# ---------------------------------------------------------------------------------------

# Matrices du système
A = np.array([
    [-0.0591, -0.02, -0.094, 0, 0, 0],
    [0.0389, 0, 2.1824, 0, 0, 0],
    [-0.0389, 0, -2.1824, 1, 0, 0],
    [0, 0, -43.6329, -1.1223, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 489.1036, 0, 0, 0, 0]
])
B = np.array([[0], [0.4306], [-0.4306], [-77.36], [0], [0]])
C = np.eye(6)
D = np.zeros((6, 1))
# Calcul du LQR
Q = np.diag([1, 1, 1, 1, 100, 1])  # Pondération des états
R = np.diag([[0.1]])  # Pondération des commandes

print("R", R)

poles_A = np.linalg.eigvals(A)
print("\nA poles:", poles_A)

K, _, _ = control.lqr(A, B, Q, R)
print("Gain matrix K:", K)

# Pôles en boucle fermée


# Représentation du système en boucle fermée
A_closed = A - B @ K

poles_closed = np.linalg.eigvals(A_closed)
print("Closed-loop poles:", poles_closed)



# Vérification des pôles en boucle fermée
poles, damping_ratios, frequencies = control.damp(control.ss(A_closed, B, C, D))
print("\nClosed-loop poles, damping ratios, and frequencies:")
for i in range(len(poles)):
    print(f"Pole: {poles[i]:.4f}, Damping: {damping_ratios[i]:.4f}, Frequency: {frequencies[i]:.4f} Hz")

# Simulez la réponse en boucle fermée
time = np.linspace(0, 10, 1000)  # Temps
response, time = control.matlab.step(control.ss(A_closed, B, C, D), time)


print("shape of repsonse :", response.shape)
print("shape of repsonse :", time.shape)

# Extraire la sortie associée à l'altitude (5ᵉ état)


# Tracer la réponse indiciaire
plt.figure(figsize=(10, 6))
plt.plot(time, response[:,0,0], label="Altitude response (closed-loop)", color="blue")
plt.title("Step Response with LQ Controller")
plt.xlabel("Time (s)")
plt.ylabel("Altitude (m)")
plt.grid()
plt.legend()
plt.show()