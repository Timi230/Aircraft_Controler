from atm_std import get_cte_atm
import numpy as np
import matplotlib.pyplot as plt
import control 
from control.matlab import *
import math as m
from scipy.interpolate import interp1d
from pylab import *
from sisopy31VS import *





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

def angle(X=X, Y=Y):
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

def space_model(C_Zeq,C_Xeq,C_Xδm, alpha_eq, Fp_xeq, X=X, Y=Y):

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

print(sys.ninputs)  # Nombre d'entrées
print(sys.noutputs) # Nombre de sorties

k = sisotool(minreal(sys[0,0]))

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

print("\n----------------------------------")
print("STUDY OF THE UNCONTROLLED AIRCRAFT")
print("----------------------------------")

print("\n(SHORT PERIOD MODE)")
print("-------------")
def short_period(A,B):
    Asp= A[2:4,2:4]
    Bsp= B[2:4,0:1]

    Csa = np.matrix([[1, 0]])
    Csq = np.matrix([[0, 1]])
    Ds = np.matrix([[0]])
    TaDm_ss = control.ss(Asp, Bsp, Csa, Ds)    # creation of the system for the alpha variable

    print("State Space representation\n")

    control.matlab.damp(TaDm_ss)             # computing the eigenvalues, the pulsation and damping ratio of the system

    print("\nTransfer function alpha/δm = ")
    TaDm_tf = control.tf(TaDm_ss)

    print(TaDm_tf)
    print("\nStatic gain of alpha/δm = %f"%(control.dcgain(TaDm_tf)))
    TqDm_ss = control.ss(Asp, Bsp, Csq, Ds)    # creation of the system for the q variable

    print("\nTransfer function q/δm = ")
    TqDm_tf = control.ss2tf(TqDm_ss)

    print(TqDm_tf)
    print("\nStatic gain of q/δm =%f \n"%(dcgain(TqDm_tf)))

    plt.figure(1)  # step response of the alpha and q variables for the short period mode
    Ya, Ta = control.matlab.step(TaDm_tf, arange(0,10,0.01))
    Yq, Tq = control.matlab.step(TqDm_tf, arange(0,10,0.01))
    plt.plot(Ta, Ya, "b", Tq, Yq, "r", lw = 2)
    plt.plot([0, Ta[-1]], [Ya[-1], Ya[-1]], 'k--', lw = 1)
    plt.plot([0, Ta[-1]], [1.05 * Ya[-1], 1.05 * Ya[-1]], 'k--', lw = 1)
    plt.plot([0, Ta[ -1 ]], [0.95 * Ya[-1], 0.95 * Ya[-1]], 'k--', lw = 1)
    plt.plot([0, Tq[-1]], [Yq[-1], Yq[-1]], 'k--', lw = 1)
    plt.plot([0, Tq[-1]], [1.05 * Yq[-1], 1.05 * Yq[-1]], 'k--', lw = 1)
    plt.plot([0, Tq[-1]], [0.95 * Yq[-1], 0.95 * Yq[-1]], 'k--', lw = 1)
    plt.minorticks_on()
    grid(True)

    plt.title(r'Step response $alpha/δm$ et $q/δm$')
    plt.legend((r'$alpha/δm$',r'$q/δm$'))
    plt.xlabel('Time (s)')
    plt.ylabel(r'$alpha$ (rad) & $q$ (rad/s)')

    # computing the settling time for both variables of the associated response
    Osa, Tra, Tsa = step_info(Ta, Ya)
    Osq, Trq, Tsq = step_info(Tq, Yq)
    yya = interp1d(Ta, Ya)
    # plt.plot(Tsa, yya(Tsa), 'bs')
    # plt.text(Tsa, yya(Tsa), Tsa)
    yyq = interp1d(Tq, Yq)
    # plt.plot(Tsq, yyq(Tsq), 'rs')
    # plt.text(Tsq, yyq(Tsq), Tsq)
    plt.show()
    print("α Settling time 5%% = %f s" %Tsa)
    print("q Settling time 5%% = %f s" %Tsq)
    savefig("stepalphaq.pdf")

    return TqDm_tf

TqDm_tf = short_period(A,B)


print("\n(PHUGOID MODE)")
print("-------------")
def phugoid_mode(A,B):
    A_phugo = A[0:2, 0:2]
    B_phugo = B[0:2, 0:1]

    Cpv = np.matrix([[1, 0]])
    Cpg = np.matrix([[0, 1]])
    Dp = np.matrix([[0]])
    TvDm_ss = control.ss(A_phugo , B_phugo, Cpv, Dp)   # creation of the system for the V variable

    print("\n State Space representation of the phugoid period")
    print(TvDm_ss)

    control.matlab.damp(TvDm_ss)             # computing the eigenvalues, the pulsation and damping ratio of the system

    print("\nTransfer function V/δm = ")

    TvDm_tf = control.tf(TvDm_ss)

    print(TvDm_tf)
    print("\nStatic gain of V/δm = %f"%(control.dcgain(TvDm_tf)))

    TgDm_ss = control.ss(A_phugo, B_phugo, Cpg, Dp)

    print("\nTransfer function gamma/δm = ")

    TgDm_tf = control.ss2tf(TgDm_ss)

    print(TgDm_tf)
    print("Static gain of gamma/δm =%f\n"%(dcgain(TgDm_tf)))

    plt.figure(2)                            # plot the step response of the V and gamma variables for the phugoid mode

    Yv, Tv = control.matlab.step(TvDm_tf, arange(0, 700, 0.1))
    Yg, Tg = control.matlab.step(TgDm_tf, arange(0, 700, 0.1))
    plt.plot(Tv, Yv, "b", Tg, Yg,"r",lw = 2)
    plt.plot([0, Tv[-1]], [Yv[-1], Yv[-1]], 'k--', lw = 1)
    plt.plot([0, Tv[-1]], [1.05 * Yv[-1], 1.05 * Yv[-1] ], 'k--', lw = 1)
    plt.plot([0, Tv[-1]], [0.95 * Yv[-1], 0.95 * Yv[-1] ], 'k--', lw = 1)
    plt.plot([0, Tg[-1]], [Yg[-1], Yg[-1]], 'k--', lw = 1)
    plt.plot([0, Tg[-1]], [1.05 * Yg[-1], 1.05 * Yg[-1]], 'k--', lw = 1)
    plt.plot([0, Tg[-1]], [0.95 * Yg[-1], 0.95 * Yg[-1]], 'k--', lw = 1)

    plt.minorticks_on()
    grid(True)

    plt.title(r'Step response $V/δm$ et $𝛾/δm$')
    plt.legend((r'$V/δm$',r'$gamma/δm$'))
    plt.xlabel('Time (s)')
    plt.ylabel(r'$V$ (rad) & $gamma$ (rad/s)')
    plt.show()

                                            # computing the settling time for both variables of the associated response

    Osv, Trv, Tsv = step_info(Tv, Yv)
    Osg, Trg, Tsg = step_info(Tg, Yg)
    yyv = interp1d(Tv, Yv)
    # plt.plot(Tsv, yyv(Tsv), 'bs')
    # plt.text(Tsv, yyv(Tsv) +1, Tsv)
    yyg = interp1d(Tg, Yg)
    # plt.plot(Tsg, yyg(Tsg), 'rs')
    # plt.text(Tsg, yyg(Tsg) -1, Tsg)

    print("V Settling time 5%% = %f s" %Tsv)
    print("𝛾 Settling time 5%% = %f s\n" %Tsg)

    savefig("stepVgamma.pdf")

phugoid_mode(A,B)

print("\n----------------------------------")
print("CONTROLLERS SYNTHESIS")
print("----------------------------------")
def get_KRvalue(A,B):
    # to be adapt to the nex state vector of size (5,1)
    Anew = A[1:, 1:]                           # defining the new matrices and vectors for the system 
    Bnew = B[1:]
    Dnew = 0

    Cq = np.array([[0], [0], [1], [0], [0]]).T # output matrix for q 
                                            
    SS_q = control.ss(Anew, Bnew, Cq, Dnew)    # compute the state space and transfer function associated to find Kr
    TF_q = control.ss2tf(SS_q)
    sisotool(-TF_q) 

    Kr = -0.13225   # From the sisotool when xi = 0.75 still 
    return Anew, Bnew, Cq, Dnew, Kr

Anew, Bnew, Cq, Dnew, Kr = get_KRvalue(A,B)


print("\n----------------------------------")
print("Q FEEDBACK LOOP")
print("----------------------------------")
def q_feedback_loop(Anew, Bnew, Cq, Dnew, Kr, TqDm_tf):
    Aq = Anew - Kr * Bnew @ Cq
    Bq = Kr * Bnew
    Dq = Kr * Dnew
    Closed_State_space_q = control.ss(Aq, Bq, Cq, Dq)  #create the the state space 
    print("Closed loop of the State Space representation of q :\n", Closed_State_space_q)
    control.matlab.damp(Closed_State_space_q)

    print("\n——————————— TF close loop ———————————\n")
    Closed_Tf_ss_q = control.tf(Closed_State_space_q)  #transfer function of the state space 
    print("Transfer Function of the closed loop : ", Closed_Tf_ss_q)

    print("\n————————— Pole close loop ———————————\n")
    control.matlab.damp(Closed_State_space_q)

    print("\n—————————— Step response  ——————————\n")
    plt.figure(3)
    Yqcl, Tqcl = control.matlab.step(Closed_Tf_ss_q,np.arange(0, 5, 0.01))
    plt.plot(Tqcl, Yqcl, "b", lw = 2)
    plt.plot([0, Tqcl[-1]], [Yqcl[-1], Yqcl[-1]], 'k--', lw =  1)
    plt.plot([0, Tqcl[-1]], [1.05 * Yqcl[-1], 1.05 * Yqcl[-1]], 'k--',lw = 1)
    plt.plot([0, Tqcl[-1]], [0.95 * Yqcl[-1], 0.95 * Yqcl[-1]], 'k--',lw = 1)
    plt.minorticks_on()
    plt.grid(visible = True, which = 'both')
    plt.title(r'Step response $q/q_c$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$q$ (rad/s)')

    Osqcl, Trqcl, Tsqcl = step_info(Tqcl, Yqcl)
    yyqcl = interp1d(Tqcl, Yqcl)
    plt.plot(Tsqcl, yyqcl(Tsqcl), 'rs')
    plt.text(Tsqcl, yyqcl(Tsqcl) - 0.02, Tsqcl)
    plt.show()
    print('q Settling time 5%% = %f s'%Tsqcl)

    print("\n———————————————— Plot of the open loop response, the closed loop response without filter and the closed loop response with the washout filter ———————————————————————\n")
    𝜏 = 0.7
    tf_washout_filter = control.tf([𝜏, 0], [𝜏, 1])
    tf_washout_filter_closed = control.feedback(Kr, TqDm_tf * tf_washout_filter)

    C_alpha = [0, 1, 0, 0, 0]
    ss_alpha = control.ss(Anew, Bnew, C_alpha, Dnew)

    tf_alpha = control.tf(ss_alpha)
    tf_alpha_washout = control.series(1 / Kr, tf_washout_filter_closed, tf_alpha)
    tf_alpha_no_washout = control.series(1 / Kr, control.feedback(Kr, TqDm_tf), tf_alpha)
    t = np.arange(0, 15, 0.01)

    plt.figure(5)
    y, t = control.matlab.step(tf_alpha, t)
    plt.plot(t, y, label = "Alpha alpha", color = "red")
    y, t = control.matlab.step(tf_alpha_no_washout, t)
    plt.plot(t, y, label = "Alpha alpha no washout", color = "blue")
    y, t = control.matlab.step(tf_alpha_washout, t)
    plt.plot(t, y, linestyle = (0, (5, 10)), color = "green", label = "Alpha alpha washout")
    plt.title("Washout filter")
    plt.grid()
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel(r'$alpha$')
    plt.show()

    return Aq, Bq, Cq, Dq, C_alpha

Aq, Bq, Cq, Dq, C_alpha = q_feedback_loop(Anew, Bnew, Cq, Dnew, Kr, TqDm_tf)



print("\n----------------------------------")
print("GAMMA FEEDBACK LOOP")
print("----------------------------------")
def gamma_feedback_loop(Aq, Bq, Dq):

    print("\n————————————————— Closed loop state space representation ————————————————\n")
    C_gamma  = np.array([[1],[0],[0],[0],[0]]).T

    SS_gamma  = control.ss(Aq, Bq, C_gamma , Dq)
    TF_gamma  = minreal(control.tf(SS_gamma))
    sisotool(TF_gamma)
    K_gamma = 16.42
    A_gamma = Aq - K_gamma * Bq @ C_gamma
    B_gamma = K_gamma * Bq
    D_gamma = K_gamma * Dq
    Cl_State_space_gamma = control.ss(A_gamma, B_gamma, C_gamma, D_gamma)  #create the the state space 
    Cl_Tf_ss_gamma = control.tf(Cl_State_space_gamma)
    print(Cl_State_space_gamma)
    control.matlab.damp(Cl_State_space_gamma)

    print("\n————————————————— TF closed loop ————————————————\n")
    print(Cl_Tf_ss_gamma)

    print("\n————————————————— Pole closed loop ————————————————\n")
    control.matlab.damp(Cl_State_space_gamma)


    print("\n————————————————————Plot of the step response of the closed loop —————————————\n")

    plt.figure(5)
    Y_gamma_cl,T_gamma_cl = control.matlab.step(Cl_Tf_ss_gamma,np.arange(0,5,0.01))
    plt.plot(T_gamma_cl,Y_gamma_cl,"b",lw=2)
    plt.plot([0,T_gamma_cl[-1]],[Y_gamma_cl[-1],Y_gamma_cl[-1]],'k--',lw = 1)
    plt.plot([0,T_gamma_cl[-1]],[1.05 * Y_gamma_cl[-1],1.05 * Y_gamma_cl[-1]],'k--',lw = 1)
    plt.plot([0,T_gamma_cl[-1]],[0.95 * Y_gamma_cl[-1],0.95 * Y_gamma_cl[-1]],'k--',lw = 1)
    plt.minorticks_on()
    plt.grid(visible = True, which ='both')
    plt.title(r'Step response $gamma/gamma_c$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$gamma$ (rad/s)')
    plt.show()

    Os𝛾cl,Tr𝛾cl,Ts_gamma_cl=step_info(T_gamma_cl,Y_gamma_cl)
    yY_gamma_cl=interp1d(T_gamma_cl,Y_gamma_cl)
    plt.plot(Ts_gamma_cl,yY_gamma_cl(Ts_gamma_cl),'rs')
    plt.text(Ts_gamma_cl,yY_gamma_cl(Ts_gamma_cl)-0.02,Ts_gamma_cl)
    print('gamma Settling time 5%% = %f s'%Ts_gamma_cl)

    return A_gamma, B_gamma, D_gamma

A_gamma, B_gamma, D_gamma = gamma_feedback_loop(Aq, Bq, Dq)


print("\n----------------------------------")
print("Z FEEDBACK LOOP")
print("----------------------------------")
def Z_feedback_loop(A_gamma, B_gamma, D_gamma):

    Cz = np.array([[0],[0],[0],[0],[1]]).T
    SS_z = control.ss(A_gamma, B_gamma, Cz, D_gamma)
    TF_z = control.ss2tf(SS_z)

    Kz = 0.00010

    Az = A_gamma - Kz * B_gamma @ Cz
    Bz = Kz * B_gamma
    Dz = Kz * D_gamma

    Cl_State_space_z = control.ss(Az, Bz, Cz, Dz)
    print("Closed loop of the State Space representation of $z$ :\n", Cl_State_space_z)

    print("\n————————————————— TF closed loop ————————————————\n")
    Cl_Tf_ss_z = control.tf(Cl_State_space_z)
    print(Cl_Tf_ss_z)

    print("\n————————————————— Pole closed loop ————————————————\n")
    control.matlab.damp(Cl_State_space_z)

    print("\n————————————————————Plot of the step response of the closed loop —————————————\n")

    plt.figure(6)
    Yzcl,Tzcl = control.matlab.step(Cl_Tf_ss_z,np.arange(0,5,0.01))
    plt.plot(Tzcl,Yzcl,"b",lw = 2)
    plt.plot([0,Tzcl[-1]],[Yzcl[-1],Yzcl[-1]],'k--',lw = 1)
    plt.plot([0,Tzcl[-1]],[1.05 * Yzcl[-1],1.05 * Yzcl[-1]],'k--',lw = 1)
    plt.plot([0,Tzcl[-1]],[0.95 * Yzcl[-1],0.95 * Yzcl[-1]],'k--',lw = 1)
    plt.minorticks_on()
    plt.grid(visible = True, which ='both')
    plt.title(r'Step response $z/z_c$')
    plt.xlabel('Time (s)')
    plt.ylabel(r'$z$ (rad/s)')
    plt.show()

    Oszcl,Trzcl,Tszcl=step_info(Tzcl,Yzcl)
    yyzcl=interp1d(Tzcl,Yzcl)
    plt.plot(Tszcl,yyzcl(Tszcl),'rs')
    plt.text(Tszcl,yyzcl(Tszcl)-0.02,Tszcl)
    print('z Settling time 5%% = %f s'%Tszcl)

    return Cl_State_space_z, Az, Bz, Dz

Cl_State_space_z, Az, Bz, Dz= Z_feedback_loop(A_gamma, B_gamma, D_gamma)


print("\n----------------------------------")
print("ADD A SAT IN GAMMA CONTROL LOOP")
print("----------------------------------")

def saturation(A_gamma_2, B_gamma_2, alpha_eq, alpha0, delta_nz_target):
    """
    Fonction pour évaluer alpha_max et trouver gamma_max par la méthode de bissection.
    """
    # Matrices d'état
    A_gamma = A_gamma_2
    B_gamma = B_gamma_2
    C_alpha_sat = np.array([[0, 1, 0, 0, 0]])
    D_alpha_sat = 0

    # Système d'état à transfert
    sys_gamma_alpha = ss(A_gamma, B_gamma, C_alpha_sat, D_alpha_sat)
    TF_gamma_alpha = ss2tf(sys_gamma_alpha)

    # Calcul de alpha_max en fonction de alpha_eq, alpha0, et Δnz_target
    alpha_max = alpha_eq + (alpha_eq - alpha0) * delta_nz_target

    # Fonction f(gamma) = max(step_response) - alpha_max
    def f(gamma, TF, alpha_max):
        response = control.matlab.step(gamma * TF)[0]
        f_gamma = np.max(response) - alpha_max
        return f_gamma

    # Méthode de bissection
    def dichotomie(f, a, b, e, TF, alpha_max):
        """
        Trouver la valeur de gamma_max en utilisant la méthode de bissection.
        """
        delta = abs(b - a)
        while delta > e:
            m = (a + b) / 2
            if f(m, TF, alpha_max) == 0:
                return m
            elif f(a, TF, alpha_max) * f(m, TF, alpha_max) > 0:
                a = m
            else:
                b = m
            delta = abs(b - a)
        return (a + b) / 2

    # Recherche de gamma_max par bissection
    gamma_opt = dichotomie(f, 0, 10, 1e-15, TF_gamma_alpha, alpha_max)

    # Méthode alternative pour gamma_opt
    alpha_max_step = np.max(control.matlab.step(TF_gamma_alpha)[0])
    gamma_opt_2 = alpha_max / alpha_max_step

    # Affichage des résultats
    print('Optimal gamma (Bissection Method):')
    print(f"gamma = {gamma_opt:.6f}")
    print('Optimal gamma (Scaling Method):')
    print(f"gamma = {gamma_opt_2:.6f}")

    # Recalculer et afficher les nouvelles matrices d'état avec gamma_max
    B_new = gamma_opt * B_gamma
    sys_updated = ss(A_gamma, B_new, C_alpha_sat, D_alpha_sat)

    print("\nUpdated State-Space Representation:")
    print("A =\n", sys_updated.A)
    print("B =\n", sys_updated.B)
    print("C =\n", sys_updated.C)
    print("D =\n", sys_updated.D)

    return alpha_max, gamma_opt, gamma_opt_2, sys_updated

alpha_max, gamma_max_bissection, gamma_max_scaling, sys_final = saturation(A_gamma, B_gamma, alpha_eq, alpha0, 3.2)
print(ss2tf(sys_final))


print("\n----------------------------------")
print("CHANGE GRAVITY CENTER")
print("----------------------------------")

def modify_state_space(c_initial=c):
    """
    Modifie la représentation d'espace d'état en fonction du nouveau centre de gravité (c).
    """
    
    # Calcul de la nouvelle valeur de c
    c_new = c_initial * 1.1  # c = f * 1.1
    
    # Calcul des nouvelles valeurs de X et Y
    Xg_new = - (c_new * l_t)  
    X_new = Xf - Xg_new                      
    Y_new = Xfδ - Xg_new
    
    # Calcul des nouvelles matrices d'état
    C_Zeq, C_Xeq, C_Xδm, alpha_eq, Fp_xeq = angle(X_new, Y_new)
    
    A,B, C, D = space_model(C_Zeq, C_Xeq, C_Xδm, alpha_eq, Fp_xeq, X_new, Y_new)

    return A, B

A_new, B_new = modify_state_space()

sys_new = control.ss(A_new, B_new, C, D)
control.matlab.damp(sys)  

time, response = control.matlab.step(sys_new)
response_selected = response[0, 0, :]  


print("Time shape:", time.shape)
print("Response shape:", response_selected.shape)


plt.figure()
plt.plot(time, response_selected)
plt.plot([0, time[-1]], [response_selected[-1], response_selected[-1]], 'k--', lw=1, label="Steady-state value")
plt.plot([0, time[-1]], [1.05 * response_selected[-1], 1.05 * response_selected[-1]], 'k--', lw=1, label="5% upper bound")
plt.plot([0, time[-1]], [0.95 * response_selected[-1], 0.95 * response_selected[-1]], 'k--', lw=1, label="5% lower bound")
plt.title("Step Response (New State-Space)")
plt.xlabel("Time (s)")
plt.ylabel("Response")
plt.grid()
plt.show()















































# print("\n----------------------------------")
# print("FLIGHT MANAGEMENT")
# print("----------------------------------")
# import numpy as np
# import control
# import matplotlib.pyplot as plt

# def flight_management(Aq, Bq, Az, Bz, Dz, Cl_State_space_z):
#     initial_altitude = 0             # mètres
#     Cruise_altitude = 1828.8        # mètres
#     final_altitude = 800            # mètres
#     gamma_ascent = np.deg2rad(12)   # angle de montée en radians
#     gamma_descent = np.deg2rad(-6)  # angle de descente en radians

#     # Vecteur temps pour chaque phase
#     t_initial = np.linspace(0, 10, 1000)
#     t_ascent = np.linspace(0, 15.3, 1000)
#     t_cruise = np.linspace(0, 100, 1000)
#     t_descent = np.linspace(0, 12.3, 1000)
#     t_final = np.linspace(0, 10, 1000)

#     #————————————— Initial Cruise Phase ————————————————#
#     initial = np.zeros((5,)) 
#     initial[-2] = initial_altitude  # Altitude initiale dans l'état

#     u_initial = np.ones_like(t_initial) * initial_altitude  # Signal d'entrée
#     altitude_initial, state_initial = control.forced_response(
#         Cl_State_space_z, T=t_initial, U=u_initial, X0=initial
#     )

#     plt.figure(7)
#     plt.plot(t_initial, altitude_initial, label='initial phase')
#     plt.title("Initial phase of the aircraft")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Altitude (meters)")
#     plt.show()

#     #———————— Ascent Phase with a Constant Flight Path Angle ————————#
#     initial_ascent = state_initial[-1, :]  # Dernier état de la phase initiale
#     Cz = np.array([[0, 0, 0, 0, 1]])  # Matrice de sortie
#     SS_gamma_h = control.ss(Aq, Bq * gamma_ascent, Cz, Dz)

#     u_ascent = np.ones_like(t_ascent) * gamma_ascent  # Signal d'entrée
#     altitude_ascent, state_ascent = control.forced_response(
#         SS_gamma_h, T=t_ascent, U=u_ascent, X0=initial_ascent
#     )

#     plt.figure(8)
#     plt.plot(t_ascent, altitude_ascent, label='ascent')
#     plt.title("Altitude during ascent phase at γ = " + str(round(np.rad2deg(gamma_ascent), 3)) + "°")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Altitude (meters)")
#     plt.show()

#     #——————————— Cruise Flight at Constant Altitude ———————————#
#     initial_cruise = state_ascent[-1]  # Dernier état de la montée
#     Cl_State_space_zh = control.ss(Az, Bz * Cruise_altitude, Cz, Dz)

#     u_cruise = np.ones_like(t_cruise) * Cruise_altitude  # Signal d'entrée
#     altitude_cruise, state_cruise = control.forced_response(
#         Cl_State_space_zh, T=t_cruise, U=u_cruise, X0=initial_cruise
#     )

#     plt.figure(9)
#     plt.plot(t_cruise, altitude_cruise, label='cruise')
#     plt.title("Cruise phase of the aircraft")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Altitude (meters)")
#     plt.show()

#     #—————————— Descent Phase with a Constant Flight Path Angle ——————————#
#     initial_descent = state_cruise[-1]  # Dernier état du vol de croisière
#     SS_gammad = control.ss(Aq, Bq * gamma_descent, Cz, Dz)

#     u_descent = np.ones_like(t_descent) * gamma_descent  # Signal d'entrée
#     altitude_descent, state_descent = control.forced_response(
#         SS_gammad, T=t_descent, U=u_descent, X0=initial_descent
#     )

#     plt.figure(10)
#     plt.plot(t_descent, altitude_descent, label='descent')
#     plt.title("Altitude during descent phase at γ = " + str(round(np.rad2deg(gamma_descent), 3)) + "°")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Altitude (meters)")
#     plt.show()

#     #———————— Final Phase: Level Flight at Constant Altitude —————————#
#     initial_finalcruise = state_descent[-1]  # Dernier état de la descente
#     Cl_State_space_zd = control.ss(Az, Bz * final_altitude, Cz, Dz)

#     u_final = np.ones_like(t_final) * final_altitude  # Signal d'entrée
#     altitude_finalcruise, state_finalcruise = control.forced_response(
#         Cl_State_space_zd, T=t_final, U=u_final, X0=initial_finalcruise
#     )

#     plt.figure(11)
#     plt.plot(t_final, altitude_finalcruise, label='final cruise')
#     plt.title("Final cruise phase of the aircraft")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Altitude (meters)")
#     plt.show()

#     #——————————————— Plot All Phases Together ——————————————#
#     plt.figure(12)
#     t_1 = t_ascent + t_initial[-1]
#     t_2 = t_cruise + t_1[-1]
#     t_3 = t_descent + t_2[-1]
#     t_4 = t_final + t_3[-1]

#     plt.plot(t_initial, altitude_initial, label='start cruise')
#     plt.plot(t_1, altitude_ascent, label='ascent')
#     plt.plot(t_2, altitude_cruise, label='cruise')
#     plt.plot(t_3, altitude_descent, label='descent')
#     plt.plot(t_4, altitude_finalcruise, label='final cruise')

#     plt.title("Flight phases")
#     plt.xlabel("Time (seconds)")
#     plt.ylabel("Altitude (meters)")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Appel de la fonction avec vos paramètres
# flight_management(Aq, Bq, Az, Bz, Dz, Cl_State_space_z) 


print("\n----------------------------------")
print("FLIGHT MANAGEMENT")
print("----------------------------------")
def flight_management(Aq, Bq, Az, Bz, Dz, Cl_State_space_z):
    initial_altitude = 0             #metres 
    Cruise_altitude = 1828.8          #metres
    final_altitude = 800            #metres
    gamma_ascent = np.deg2rad(12)        #path angle ascent setting: degree
    gamma_descent = np.deg2rad(-6)     #path angle descent setting: degree

    #————————————— Initial Cruise Phase ————————————————#

    initial  = np.zeros((5,1)) 
    initial[-2,0] = initial_altitude

    altitude_initial, time_initial, state_initial = control.matlab.step(
        initial_altitude*Cl_State_space_z,
        T = 10,
        X0 = initial,
        return_x = True)

    plt.figure(7)

    plt.plot(time_initial,altitude_initial,label = 'initial phase')
    plt.title("Initial phase of the aircraft")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Altitude (meters)")
    plt.show()

    #———————————————— An ascent phase with a constant flight path angle ——————————————————#

    initial_ascent = state_initial[-1,:]
    Cz = np.array([[0],[0],[0],[0],[1]]).T
    SS_gamma_h  = control.ss(Aq, Bq * gamma_ascent, Cz , Dz)

    altitude_ascent, time_ascent, state_ascent = control.matlab.step(
        SS_gamma_h,
        T = 15.3,
        X0 = initial_ascent,
        return_x = True)

    plt.figure(8)

    plt.plot(time_ascent,altitude_ascent,label = 'ascent')
    plt.title("Altitude during the ascent phase at $gamma$ = " + str(round(np.rad2deg(gamma_ascent),3)) + " degrees")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Altitude (metres)")
    plt.show()

    #—————————————————— A cruise flight at constant altitude (of about 100s) ————————————————#

    initial_cruise = state_ascent[-1,:]

    Cl_State_space_zh = control.ss(Az, Bz * Cruise_altitude, Cz, Dz)

    altitude_cruise, time_cruise, state_cruise = control.matlab.step(
        Cl_State_space_zh,
        T = 100,
        X0 = initial_cruise,
        return_x = True)

    plt.figure(9)

    plt.plot(time_cruise,altitude_cruise,label = 'cruise')
    plt.title("Cruise phase of the aircraft")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Altitude (meters)")
    plt.show()

    #————————————— A descent phase with a constant flight path angle ———————————————#

    initial_descent = state_cruise[-1,:]

    SS_gammad  = control.ss(Aq, Bq * gamma_descent , Cz , Dz)
    altitude_descent, time_descent, state_descent = control.matlab.step(
        SS_gammad,
        T = 12.3,
        X0 = initial_descent,
        return_x = True)

    plt.figure(10)

    plt.plot(time_descent,altitude_descent,label = 'descent')
    plt.title("Altitude during the descent phase at $gamma$ = " + str(round(np.rad2deg(gamma_descent),3)) + " degrees")
    plt.xlabel("time (seconds)")
    plt.ylabel("Altitude (meters)")
    plt.show()

    #——————————————————— A final flare and a small phase of level flight at constant altitude ————————————————#

    initial_finalcruise = state_descent[-1,:]

    Cl_State_space_zd = control.ss(Az, Bz * final_altitude , Cz, Dz)
    altitude_finalcruise, time_finalcruise, state_finalcruise = control.matlab.step(
        Cl_State_space_zd,
        T = 10,
        X0 = initial_finalcruise,
        return_x = True)

    plt.figure(11)
    plt.plot(time_finalcruise,altitude_finalcruise,label = 'final cruise')
    plt.title("Final Cruise phase of the aircraft")
    plt.xlabel("time (seconds)")
    plt.ylabel("Altitude (meters)")
    plt.show()

    #—————————————————————- All the phases of the aircraft ———————————————————#

    plt.figure(12)
    plt.plot(time_initial,altitude_initial,label = 'start cruise')

    time_1 = time_ascent + time_initial[-1]
    plt.plot(time_1,altitude_ascent,label = 'ascent')

    time_2 = time_1[-1] + time_cruise
    plt.plot(time_2,altitude_cruise,label = 'cruise')

    time_3 = time_descent + time_2[-1]
    plt.plot(time_3,altitude_descent,label = 'descent') 

    time_4 = time_finalcruise + time_3[-1]
    plt.plot(time_4,altitude_finalcruise,label = 'final cruise')

    plt.title("Flight phases")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.legend()
    plt.grid(True)
    plt.show()

#flight_management(Aq, Bq, Az, Bz, Dz, Cl_State_space_z)
