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

#---------------------------------------------------------------------

# Find equilibrium point

#---------------------------------------------------------------------

def angle():
    # ————— Initialisation du point d'équilibre pour l'étude de l'aéronef —————#

    alpha_eq0   = 0                     # Initialisation de l'angle d'incidence au point d'équilibre : radians
    Fp_xeq0     = 0                     # Initialisation de la première force au point d'équilibre
    epsilon     = 10**-6                # Précision pour convergence
    max_iter    = 1000                  # Limite maximale d'itérations pour prévenir les boucles infinies

    # Initialisation des variables
    alpha_eqold = alpha_eq0        
    alpha_eq    = 0.1                   # Valeur initiale pour alpha_eq (ajustée pour aider la convergence)
    Fp_xeq      = Fp_xeq0               # Valeur initiale pour Fp_xeq

    i = 0
    while abs(alpha_eq - alpha_eqold) >= epsilon and i < max_iter:
        C_Zeq = (1 / (Q * S)) * (mass * g - Fp_xeq * m.sin(alpha_eqold)) #page 49
        C_Xeq = Cx0 + k * (C_Zeq)**2 #page 48
        C_Xδm = 2 * k * C_Zeq * Czδm

        # Calcul de la déflection de gouverne δmeq
        denom = (C_Xδm * m.sin(alpha_eqold) + Czδm * m.cos(alpha_eqold))
        if denom == 0:  # Vérification pour éviter division par zéro
            print("Erreur : Division par zéro détectée dans le calcul de δmeq")
            break

        δmeq = δm0 - ((C_Xeq * m.sin(alpha_eqold) + C_Zeq * m.cos(alpha_eqold)) / denom) * (X / (Y - X))

        # Mise à jour de alpha_eqold pour la prochaine itération
        alpha_eqold = alpha_eq
        alpha_eq = alpha0 + (C_Zeq / Cz_alpha) - (Czδm / Cz_alpha) * δmeq

        # Vérification si alpha_eq est dans la plage correcte
        if not (-m.pi/2 < alpha_eq < m.pi/2):
            print(f"Erreur : alpha_eq hors plage ({alpha_eq})")
            break

        # Vérification pour s'assurer que cos(alpha_eq) n'est pas proche de zéro
        cos_alpha = m.cos(alpha_eq)
        if abs(cos_alpha) < 1e-6:
            print("Erreur : cos(alpha_eq) proche de zéro")
            break

        # Mise à jour de Fp_xeq en fonction du nouvel angle alpha_eq
        Fp_xeq = (Q * S * C_Xeq) / cos_alpha

        i += 1
        print(f"Itération {i}: alpha_eq = {round(alpha_eq, 10)}")

    # Vérification si la boucle a convergé
    if i >= max_iter:
        print("Avertissement : Le calcul n'a pas convergé après le nombre maximal d'itérations")
    else:
        print("-------------")
        print(f"alpha_eq final = {round(alpha_eq, 10)} trouvé après {i} itérations")

    return C_Zeq, C_Xeq, C_Xδm, alpha_eq, Fp_xeq

# Appel de la fonction améliorée
C_Zeq, C_Xeq, C_Xδm, alpha_eq, Fp_xeq = angle()

# Affichage des résultats finaux
print("-------------")
print(f"C_Zeq = {round(C_Zeq, 10)}")
print(f"C_Xeq = {round(C_Xeq, 10)}")
print(f"C_Xδm = {round(C_Xδm, 10)}")
print(f"alpha_eq = {round(alpha_eq, 10)}")
print(f"Fp_xeq = {round(Fp_xeq, 10)}")
print("-------------")

def space_model(C_Zeq,C_Xeq,C_Xδm, alpha_eq, Fp_xeq):

    #SS MODEL INIT
    gamma_eq   = 0
    Cz         = C_Zeq
    Cx_alpha   = 2 * k * Cz * Cz_alpha
    Fτ         = 0

    Cm_alpha   = (X / l_ref) * (Cx_alpha * m.sin(alpha_eq) + Cz_alpha * m.cos(alpha_eq))
    Cmδm       = (Y / l_ref) * (C_Xδm * m.sin(alpha_eq) + Czδm * m.cos(alpha_eq))

    Xv         = (2 * Q * S * C_Xeq) / (mass * V_eq)
    X_alpha    = (Fp_xeq / (mass * V_eq)) * m.sin(alpha_eq) + (Q * S * Cx_alpha) / (mass * V_eq)
    X_gamma    = g * m.cos(gamma_eq) / V_eq
    Xδm        = (Q * S * C_Xδm) / (mass * V_eq)
    Xτ         = - (Fτ * m.cos(alpha_eq)) / (mass * V_eq)

    mv         = 0
    m_alpha    = (Q * S * l_ref * Cm_alpha) / Iyy
    mq         = (Q * S * l_ref**2 * Cmq) / (V_eq * Iyy)
    mδm        = (Q * S * l_ref * Cmδm) / Iyy

    Zv         = (2 * Q * S * C_Zeq) / (mass * V_eq)
    Z_alpha    = (Fp_xeq * m.cos(alpha_eq)) / (mass * V_eq) + (Q * S * Cz_alpha) / (mass * V_eq)
    Z_gamma    = (g * m.sin(gamma_eq)) / V_eq
    Zδm        = (Q * S * Czδm) / (mass * V_eq)
    Zτ         = (Fτ * m.sin(alpha_eq)) / (mass * V_eq)


    A = np.array([[-Xv, -X_gamma, -X_alpha, 0, 0, 0],
                [Zv, 0, Z_alpha, 0, 0, 0],
                [-Zv, 0, -Z_alpha, 1, 0, 0],
                [0, 0, m_alpha, mq, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, V_eq, 0, 0, 0, 0]])

    B = np.array([[0],[Zδm],[-Zδm],[mδm],[0],[0]])
    C = np.eye(6)
    D = np.zeros((6,1))

    print("A = \n",A)
    print("\nB = \n",B)
    print("\nC = \n",C)
    print("\nD = \n",D) 

    eigenA = np.linalg.eigvals(A)
    # print("\n Eigen values of A =\n",eigenA) # compute the eigenvalues of A 
               

    return A, B, C, D

A,B, C, D = space_model(C_Zeq, C_Xeq, C_Xδm, alpha_eq, Fp_xeq)

sys = control.ss(A,B,C,D)
control.matlab.damp(sys)  # calcul of the damping ratio

def calculate_thrust():
    # Calcul de la traînée aérodynamique
    F_drag = (1/2) * rho * V_eq**2 * S * C_Xeq
    print(f"Traînée (F_drag): {F_drag:.2f} N")
    
    # Angle de vol en radians (gamma_eq est généralement 0 pour un vol horizontal)
    gamma_eq = 0
    
    # Composante longitudinale de la force gravitationnelle
    F_gravity = mass * g * m.sin(gamma_eq)
    print(f"Force gravitationnelle (F_gravity): {F_gravity:.2f} N")
    
    # Composante longitudinale de la force aérodynamique (Fp_xeq)
    Fp_xeq = (Q * S * C_Xeq) / m.cos(alpha_eq)
    print(f"Force longitudinale (Fp_xeq): {Fp_xeq:.2f} N")
    
    # Calcul de la poussée
    F_tau = F_drag + F_gravity - Fp_xeq
    print(f"Charge de poussée calculée (F_τ): {F_tau:.2f} N")
    
    return F_tau

# Appel de la fonction avec affichage des résultats intermédiaires
F_tau = calculate_thrust()
