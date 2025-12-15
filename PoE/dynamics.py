import numpy as np
from numba import njit, prange
import PoE.kinematics as kine
import PoE.helper as helper
import PoE.lieTheory as lie

tol = 1e-5

@njit(cache=True)
def inverse_Dynamics(S, MList, GList, thetaList, thetaDotList, thetaDotDotList, g, Ftip):
    """Compute using RNEA - recursive Newton Euler algorithm
    S: 6xn twist matrix
    Mlist: 4x4xn Home matrix with respect to each consecutive joint
    Glist: 6x6xn inertia space matrix with respect to each movable link (dynamic link)
    thetaList/thetaDotList/thetaDotDotList: 4x1 angle/velocity/acceleration of each joint
    g: earth gravitional force with respect to base frame
    Ftip: EE external applied force with respect to base frame
    
    return: torqueList needed for each joint
    """
    n_joint = thetaList.size
    
    Mi = np.eye(4, dtype=np.float64)
    Ai = np.zeros((6, n_joint), dtype=np.float64)
    AdTi = np.zeros((6, 6, n_joint + 1), dtype=np.float64)
    Vi = np.zeros((6, n_joint + 1), dtype=np.float64)
    Vdi = np.zeros((6, n_joint + 1), dtype=np.float64)
    Vdi[3, 0] = -g[0]
    Vdi[4, 0] = -g[1]
    Vdi[5, 0] = -g[2]
    
    AdTi[:, :, n_joint] = lie.adjoint_T(helper.transInv_SE3(MList[:, :, n_joint]))
    Fi = Ftip.copy()
    tauList = np.zeros(n_joint, dtype=np.float64)
    
    # Forward pass
    for i in range(n_joint):
        Mi = Mi @ MList[:, :, i]
        Mi_inv = helper.transInv_SE3(Mi)
        Ai[:, i] = lie.adjoint_T(Mi_inv) @ S[:, i]
        
        se3_neg = lie.twistTo_se3(Ai[:, i] * (-thetaList[i]))
        exp_se3_neg = lie.exp_se3(se3_neg)
        MList_inv = helper.transInv_SE3(MList[:, :, i])
        AdTi[:, :, i] = lie.adjoint_T(exp_se3_neg @ MList_inv)
        
        Vi[:, i + 1] = AdTi[:, :, i] @ Vi[:, i] + Ai[:, i] * thetaDotList[i]
        
        adjV = lie.adjoint_V(Vi[:, i + 1])
        Vdi[:, i + 1] = AdTi[:, :, i] @ Vdi[:, i] + Ai[:, i] * thetaDotDotList[i] + adjV @ Ai[:, i] * thetaDotList[i]
    
    # Backward pass
    for i in range(n_joint - 1, -1, -1):
        G_i = GList[:, :, i]
        V_i1 = Vi[:, i + 1]
        Vd_i1 = Vdi[:, i + 1]
        
        adjV_i1 = lie.adjoint_V(V_i1)
        GV = G_i @ V_i1
        GVd = G_i @ Vd_i1
        
        Fi = AdTi[:, :, i + 1].T @ Fi + GVd - adjV_i1.T @ GV
        
        tau_i = 0.0
        for j in range(6):
            tau_i += Fi[j] * Ai[j, i]
        tauList[i] = tau_i
    
    return tauList

@njit(cache=True)
def forward_Dynamics(S, MList, GList, thetaList, thetaDotList, tauList, g, Ftip):
    """Compute forward dynamics using inverse dynamics!
    S: 6xn twist matrix
    Mlist: 4x4xn Home matrix with respect to each consecutive joint
    Glist: 6x6xn inertia space matrix with respect to each movable link (dynamic link)
    thetaList/thetaDotList/tauList: 4x1 angle/velocity/torque of each joint
    g: earth gravitional force with respect to base frame
    Ftip: EE external applied force with respect to base frame
    
    return: torqueList needed for each joint 
    """
    MassMatrix = mass_Matrix(S, MList, GList, thetaList)
    CoriolisVector = coriolis_Vector(S, MList, GList, thetaList, thetaDotList)
    EEWrench = EEWrenchForces_Vector(S, MList, GList, thetaList, Ftip)
    GravVec = gravitionalForce_Vector(S, MList, GList, thetaList, g)
    
    rhs = tauList - CoriolisVector - GravVec - EEWrench
    return np.linalg.solve(MassMatrix, rhs)

@njit(cache=True)
def mass_Matrix(S, MList, GList, thetaList):
    """Compute mass matrix for forward dynamics
    S: 6xn twist matrix
    Mlist: 4x4xn Home matrix with respect to each consecutive joint
    Glist: 6x6xn inertia space matrix with respect to each movable link (dynamic link)
    thetaList: 4x1 angle of each joint
    
    return: 4x4 mass matrix
    """
    n_joint = thetaList.size
    
    MassMatrix = np.zeros((n_joint, n_joint), dtype=np.float64)
    g = np.zeros(3, dtype=np.float64)
    Ftip = np.zeros(6, dtype=np.float64)
    
    for i in range(n_joint):
        thetaDotList = np.zeros(n_joint, dtype=np.float64)
        thetaDotDotList = np.zeros(n_joint, dtype=np.float64)
        thetaDotDotList[i] = 1.0
        MassMatrix[:, i] = inverse_Dynamics(S, MList, GList, thetaList, thetaDotList, thetaDotDotList, g, Ftip)
    
    return MassMatrix

@njit(cache=True)
def coriolis_Vector(S, MList, GList, thetaList, thetaDotList):
    """Compute coriolis vector using inverse dynamic func
    S: 6xn twist matrix
    Mlist: 4x4xn Home matrix with respect to each consecutive joint
    Glist: 6x6xn inertia space matrix with respect to each movable link (dynamic link)
    thetaList/thetaDotList: 4x1 angle/velocity of each joint
    
    return: coriolis vector with each element is coriolis force each joint applied
    """
    n_joint = thetaList.size
    thetaDotDotList = np.zeros(n_joint, dtype=np.float64)
    g = np.zeros(3, dtype=np.float64)
    Ftip = np.zeros(6, dtype=np.float64)
    
    return inverse_Dynamics(S, MList, GList, thetaList, thetaDotList, thetaDotDotList, g, Ftip)

@njit(cache=True)
def gravitionalForce_Vector(S, MList, GList, thetaList, g):
    """Compute gravitional force vector using inverse dynamic func
    S: 6xn twist matrix
    Mlist: 4x4xn Home matrix with respect to each consecutive joint
    Glist: 6x6xn inertia space matrix with respect to each movable link (dynamic link)
    thetaList: 4x1 angle/velocity of each joint
    g: earth gravitional force with respect to base frame
    
    return: gravitational force vector with each element is gravitional force each joint applied
    """
    n_joint = thetaList.size
    thetaDotList = np.zeros(n_joint, dtype=np.float64)
    thetaDotDotList = np.zeros(n_joint, dtype=np.float64)
    Ftip = np.zeros(6, dtype=np.float64)
    
    return inverse_Dynamics(S, MList, GList, thetaList, thetaDotList, thetaDotDotList, g, Ftip)

@njit(cache=True)
def EEWrenchForces_Vector(S, MList, GList, thetaList, Ftip):
    """Compute gravitional force vector using inverse dynamic func
    S: 6xn twist matrix
    Mlist: 4x4xn Home matrix with respect to each consecutive joint
    Glist: 6x6xn inertia space matrix with respect to each movable link (dynamic link)
    thetaList: 4x1 angle/velocity of each joint
    Ftip: EE external applied force with respect to base frame
    
    return: wrench force vector applied to EE 
    """
    n_joint = thetaList.size
    thetaDotList = np.zeros(n_joint, dtype=np.float64)
    thetaDotDotList = np.zeros(n_joint, dtype=np.float64)
    g = np.zeros(3, dtype=np.float64)
    
    return inverse_Dynamics(S, MList, GList, thetaList, thetaDotList, thetaDotDotList, g, Ftip)