import numpy as np
import matplotlib.pyplot as plt
import control 
from control.matlab import *
import math as m
from scipy.interpolate import interp1d
from pylab import *
from outils.sisopy31VS import *


from data.data import *


def equilibrium_angle(X=X, Y=Y):
    """
    Calculate the equilibrium angle of attack for the aircraft.

    Parameters:
        X (float): A constant value related to the aircraft's geometry.
        Y (float): A constant value related to the aircraft's geometry.

    Returns:
        C_Zeq (float): The equilibrium lift coefficient.
        C_Xeq (float): The equilibrium drag coefficient.
        C_Xδm (float): The equilibrium derivative of drag coefficient with respect to the aileron deflection.
        alpha_eq (float): The equilibrium angle of attack in radians.
        Fp_xeq (float): The equilibrium longitudinal force.
    """
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

def space_model(C_Zeq,C_Xeq,C_Xδm, alpha_eq, Fp_xeq, X=X, Y=Y):
    """
    This function calculates the state space matrices A, B, C, and D for the aircraft model.

    Parameters:
        C_Zeq (float): Equivalent vertical force coefficient.
        C_Xeq (float): Equivalent longitudinal force coefficient.
        C_Xδm (float): Equivalent longitudinal force coefficient due to aileron deflection.
        alpha_eq (float): Equivalent angle of attack.
        Fp_xeq (float): Equivalent longitudinal force.
        X (float): Constant value. Default value is X.
        Y (float): Constant value. Default value is Y.

    Returns:
        A (numpy.ndarray): State space matrix A.
        B (numpy.ndarray): State space matrix B.
        C (numpy.ndarray): State space matrix C.
        D (numpy.ndarray): State space matrix D.
    """
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

def calculate_thrust():
    """
    Calculates the thrust force acting on the aircraft.

    Parameters:
        None

    Returns:
        F_tau (float): The calculated thrust force in Newtons.

        The thrust force is calculated using the drag force, gravity force, and the longitudinal force.
    """
    # Calcul de la traînée aérodynamique
    
    C_Zeq, C_Xeq, C_Xδm, alpha_eq, Fp_xeq = equilibrium_angle()

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

def short_period(A,B):
    """
    Description:
        This function computes and analyzes the short-period dynamics of an aircraft using its state-space representation. 
        It calculates transfer functions, static gains, step responses, and settling times for the short-period mode.

    Parameters:
        A (numpy.ndarray): The state matrix of the system.
        B (numpy.ndarray): The input matrix of the system.

    Returns:
        control.TransferFunction: The transfer function of the system for the variable q with respect to δm.
    """
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

    return TqDm_tf

def phugoid_mode(A,B):
    """
    Description:
        This function computes and analyzes the phugoid mode dynamics of an aircraft using its state-space representation.
        It calculates transfer functions, static gains, step responses, and settling times for the phugoid mode.

    Parameters:
        A (numpy.ndarray): The state matrix of the system.
        B (numpy.ndarray): The input matrix of the system.

    Returns:
        None: This function does not return a value but prints the results and plots the step responses of V and γ.
    """

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
    
def get_Kr(A,B):
    """
    Description:
        This function computes the state-space and transfer function representations for a specific system output (q) 
        and uses the MATLAB-like SISO tool to analyze the system and derive a specific gain value (Kr).

    Parameters:
        A (numpy.ndarray): The state matrix of the system.
        B (numpy.ndarray): The input matrix of the system.

    Returns:
        tuple: A tuple containing the following:
            - Anew (numpy.ndarray): The reduced state matrix.
            - Bnew (numpy.ndarray): The reduced input matrix.
            - Cq (numpy.ndarray): The output matrix for q.
            - Dnew (int): The feedthrough matrix.
            - Kr (float): The computed gain value from the SISO tool.
    """

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

def q_feedback_loop(Anew, Bnew, Cq, Dnew, Kr, TqDm_tf):
    """
    Description:
        This function creates a feedback loop for the q variable in an aircraft control system.
        It calculates the closed-loop state-space representation, transfer function, and step responses.
        It also demonstrates the effect of a washout filter on the closed-loop system.

    Parameters:
        Anew (numpy.ndarray): The reduced state matrix.
        Bnew (numpy.ndarray): The reduced input matrix.
        Cq (numpy.ndarray): The output matrix for q.
        Dnew (int): The feedthrough matrix.
        Kr (float): The feedback gain for the closed-loop system.
        TqDm_tf (control.TransferFunction): The open-loop transfer function for q with respect to δm.

    Returns:
        tuple: A tuple containing the following:
            - Aq (numpy.ndarray): The state matrix of the closed-loop system.
            - Bq (numpy.ndarray): The input matrix of the closed-loop system.
            - Cq (numpy.ndarray): The output matrix of the closed-loop system (unchanged from input).
            - Dq (int): The feedthrough matrix of the closed-loop system.
            - C_alpha (list): The output matrix for α in the state-space representation.
    """

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

def gamma_feedback_loop(Aq, Bq, Dq):
    """
    Description:
        This function implements a feedback loop for the γ (gamma) variable in an aircraft control system.
        It calculates the closed-loop state-space representation, transfer function, and step responses.
        The function also determines and uses a feedback gain for the closed-loop system.

    Parameters:
        Aq (numpy.ndarray): The state matrix of the q feedback loop.
        Bq (numpy.ndarray): The input matrix of the q feedback loop.
        Dq (int): The feedthrough matrix of the q feedback loop.

    Returns:
        tuple: A tuple containing the following:
            - A_gamma (numpy.ndarray): The state matrix of the γ closed-loop system.
            - B_gamma (numpy.ndarray): The input matrix of the γ closed-loop system.
            - D_gamma (int): The feedthrough matrix of the γ closed-loop system.
    """


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

def z_feedback_loop(A_gamma, B_gamma, D_gamma):
    """
    Description:
        This function implements a feedback loop for the z variable in an aircraft control system.
        It calculates the closed-loop state-space representation, transfer function, and step responses.
        The function also determines and uses a feedback gain for the closed-loop system.

    Parameters:
        A_gamma (numpy.ndarray): The state matrix of the γ feedback loop.
        B_gamma (numpy.ndarray): The input matrix of the γ feedback loop.
        D_gamma (int): The feedthrough matrix of the γ feedback loop.

    Returns:
        tuple: A tuple containing the following:
            - Cl_State_space_z (control.StateSpace): The closed-loop state-space representation of the z system.
            - Az (numpy.ndarray): The state matrix of the z closed-loop system.
            - Bz (numpy.ndarray): The input matrix of the z closed-loop system.
            - Dz (int): The feedthrough matrix of the z closed-loop system.
    """
    
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

def saturation(A_gamma_2, B_gamma_2, alpha_eq, alpha0, delta_nz_target):
    """
    Description:
        This function evaluates the maximum angle of attack (α_max) and computes the optimal gain (γ_max) for an aircraft control system using the bisection method. It also updates the state-space representation of the system based on the calculated γ_max.

    Parameters:
        A_gamma_2 (numpy.ndarray): The state matrix of the γ feedback loop.
        B_gamma_2 (numpy.ndarray): The input matrix of the γ feedback loop.
        alpha_eq (float): The equilibrium angle of attack.
        alpha0 (float): The initial angle of attack.
        delta_nz_target (float): The target change in normal acceleration (Δnz).

    Returns:
        tuple: A tuple containing the following:
            - alpha_max (float): The maximum angle of attack calculated based on the equilibrium angle and target Δnz.
            - gamma_opt (float): The optimal gain computed using the bisection method.
            - gamma_opt_2 (float): The optimal gain computed using the scaling method.
            - sys_updated (control.StateSpace): The updated state-space representation of the system with the new gain.
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

def modify_state_space(c_initial=c):
    """
    Description:
        This function modifies the state-space representation of an aircraft system by updating the center of gravity (c). 
        It recalculates the necessary parameters and state-space matrices based on the new center of gravity.

    Parameters:
        c_initial (float): The initial center of gravity (default is `c` from the global context).

    Returns:
        tuple: A tuple containing the updated state-space matrices:
            - A (numpy.ndarray): The state matrix.
            - B (numpy.ndarray): The input matrix.
    """
    
    # Calcul de la nouvelle valeur de c
    c_new = c_initial * 1.1  # c = f * 1.1
    
    # Calcul des nouvelles valeurs de X et Y
    Xg_new = - (c_new * l_t)  
    X_new = Xf - Xg_new                      
    Y_new = Xfδ - Xg_new
    
    # Calcul des nouvelles matrices d'état
    C_Zeq, C_Xeq, C_Xδm, alpha_eq, Fp_xeq = equilibrium_angle(X_new, Y_new)
    
    A_new, B_new, C, D = space_model(C_Zeq, C_Xeq, C_Xδm, alpha_eq, Fp_xeq, X_new, Y_new)
    
    sys_new = control.ss(A_new, B_new, C, D)
      

    response, time = control.matlab.step(sys_new)
    response_selected = response[:, 0, 0]  

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

    return A_new, B_new, sys_new

def LQController(A, B, C, D):
    """
    Description:
        This function designs a Linear Quadratic (LQ) controller for an aircraft system using the Linear Quadratic Regulator (LQR) method.
        It calculates the LQR gain matrix (K) and evaluates the closed-loop poles and damping ratios.

    Parameters:
        A (numpy.ndarray): The state matrix of the system.
        B (numpy.ndarray): The input matrix of the system.

    Returns:
        numpy.ndarray: The LQR gain matrix (K).
    """
    
    # Pondérations Q et R
    Q = np.diag([1, 1, 1, 1, 100, 1])  # Pondération des états
    R = np.diag([[0.1]])  # Pondération des commandes

    K, _, _ = control.lqr(A, B, Q, R)
    print("Gain matrix K:", K)

    # Pôles en boucle fermée
    A_closed = A - B @ K
    
    time = np.linspace(0, 10, 1000)  
    response, time = control.matlab.step(control.ss(A_closed, B, C, D), time)



    plt.figure(figsize=(10, 6))
    plt.plot(time, response[:,0,0], label="Altitude response (closed-loop)", color="blue")
    plt.title("Step Response with LQ Controller")
    plt.xlabel("Time (s)")
    plt.ylabel("Altitude (m)")
    plt.grid()
    plt.legend()
    plt.show()

    return K, A_closed