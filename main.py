import numpy as np
import matplotlib.pyplot as plt
import control 
from control.matlab import *
import math as m
from scipy.interpolate import interp1d
from pylab import *
from outils.sisopy31VS import *

from data.data import *
from outils.library import *




print("\n----------------------------------")
print("Find equilibrium point")
print("----------------------------------")

C_Zeq, C_Xeq, C_Xδm, alpha_eq, Fp_xeq = equilibrium_angle()

print(f"C_Zeq = {round(C_Zeq, 10)}")
print(f"C_Xeq = {round(C_Xeq, 10)}")
print(f"C_Xδm = {round(C_Xδm, 10)}")
print(f"alpha_eq = {round(alpha_eq, 10)}")
print(f"Fp_xeq = {round(Fp_xeq, 10)}")


A,B, C, D = space_model(C_Zeq, C_Xeq, C_Xδm, alpha_eq, Fp_xeq)

sys = control.ss(A,B,C,D)
control.matlab.damp(sys)  

print("\n----------------------------------")
print("Trust function")
print("----------------------------------")

# Appel de la fonction avec affichage des résultats intermédiaires
F_tau = calculate_thrust()

print("\n----------------------------------")
print("Short period mode")
print("----------------------------------")


TqDm_tf = short_period(A,B)

print("\n----------------------------------")
print("Phugoid period mode")
print("----------------------------------")

phugoid_mode(A,B)

#----------------------------------------------------------------
# Kr
#----------------------------------------------------------------


Anew, Bnew, Cq, Dnew, Kr = get_Kr(A,B)


print("\n----------------------------------")
print("Q feedback loop")
print("----------------------------------")


Aq, Bq, Cq, Dq, C_alpha = q_feedback_loop(Anew, Bnew, Cq, Dnew, Kr, TqDm_tf)



print("\n----------------------------------")
print("γ feedback loop")
print("----------------------------------")


A_gamma, B_gamma, D_gamma = gamma_feedback_loop(Aq, Bq, Dq)


print("\n----------------------------------")
print("Z feedback loop")
print("----------------------------------")


Cl_State_space_z, Az, Bz, Dz= z_feedback_loop(A_gamma, B_gamma, D_gamma)


print("\n----------------------------------")
print("Saturation")
print("----------------------------------")



alpha_max, gamma_max_bissection, gamma_max_scaling, sys_final = saturation(A_gamma, B_gamma, alpha_eq, alpha0, 3.2)
print(ss2tf(sys_final))


print("\n----------------------------------")
print("Change gravity center")
print("----------------------------------")



A_new, B_new, sys_new = modify_state_space()
control.matlab.damp(sys_new)



print("\n----------------------------------")
print("LQ Controller")
print("----------------------------------")



K, A_closed = LQController(A, B, C, D)

print("\nClosed-loop poles, damping ratios, and frequencies:")

control.damp(control.ss(A_closed, B, C, D))

