import PoE.kinematics as kine
import PoE.helper as helper
import PoE.lieTheory as lie

import numpy as np

tol = 1e-5

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
    
    Mi = np.eye(4,4, dtype=np.float32)
    Ai = np.zeros((6, n_joint), dtype=np.float32)
    AdTi = np.zeros((6,6,n_joint+1), dtype=np.float32)
    Vi = np.zeros((6, n_joint+1), dtype=np.float32)
    Vdi = np.zeros((6,n_joint+1), dtype=np.float32)
    Vdi[3:6, 0] = -g 
    AdTi[:,:, n_joint] = lie.adjoint_T(helper.transInv_SE3(MList[:,:,n_joint]))
    Fi = Ftip.copy()
    tauList = np.zeros(n_joint, dtype=np.float32)
    
    for i in range(0,n_joint):
        Mi = Mi @ MList[:,:, i]
        Ai[:, i] = lie.adjoint_T(helper.transInv_SE3(Mi)) @ S[:, i]
        AdTi[:,:, i] = lie.adjoint_T(lie.exp_se3(lie.twistTo_se3(Ai[:,i]*(-thetaList[i]))) @ helper.transInv_SE3(MList[:,:,i]))
        Vi[:,i+1] = AdTi[:,:,i] @ Vi[:, i] + Ai[:, i] * thetaDotList[i]
        Vdi[:, i+1] = AdTi[:,:,i] @ Vdi[:,i] + Ai[:,i] * thetaDotDotList[i] + lie.adjoint_V(Vi[:, i+1]) @ Ai[:, i] * thetaDotList[i]
    
    for i in range(n_joint-1, -1, -1):
        Fi = AdTi[:,:,i+1].T @ Fi + (GList[:,:, i] @ Vdi[:,i+1]).ravel() - (lie.adjoint_V(Vi[:, i+1]).T @ (GList[:,:,i] @ Vi[:, i+1])).ravel()
        tauList[i] = Fi @ Ai[:, i]
    
    return tauList.ravel()

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
    CoriolisVector = coriolis_Vector(S, MList, GList, thetaList, thetaDotList )
    EEWrench = EEWrenchForces_Vector(S, MList, GList, thetaList, Ftip)
    GravVec = gravitionalForce_Vector(S, MList, GList, thetaList, g)
    
    return np.linalg.solve(MassMatrix, tauList-CoriolisVector - GravVec - EEWrench)
        
def mass_Matrix(S, MList, GList, thetaList):
    """Compute mass matrix for forward dynamics
        S: 6xn twist matrix
        Mlist: 4x4xn Home matrix with respect to each consecutive joint
        Glist: 6x6xn inertia space matrix with respect to each movable link (dynamic link)
        thetaList: 4x1 angle of each joint
        
        return: 4x4 mass matrix
    """
    n_joint = thetaList.size
    
    MassMatrix = np.zeros((n_joint,n_joint), dtype=np.float32)
    g = np.zeros(3, dtype=np.float32)
    Ftip = np.zeros(6, dtype=np.float32)
    
    
    for i in range(0, n_joint):
        thetaDotList = np.zeros_like(thetaList)
        thetaDotDotList = np.zeros_like(thetaList)  
        thetaDotDotList[i] = 1
        MassMatrix[:, i] = inverse_Dynamics(S, MList, GList, thetaList, thetaDotList, thetaDotDotList, g, Ftip)
    return MassMatrix 
    
def coriolis_Vector(S, MList, GList, thetaList, thetaDotList):
    """Compute coriolis vector using inverse dynamic func
        S: 6xn twist matrix
        Mlist: 4x4xn Home matrix with respect to each consecutive joint
        Glist: 6x6xn inertia space matrix with respect to each movable link (dynamic link)
        thetaList/thetaDotList: 4x1 angle/velocity of each joint
        
        return: coriolis vector with each element is coriolis force each joint applied
    """
    thetaDotDotList = np.zeros_like(thetaList)
    g = np.zeros(3, dtype=np.float32)
    Ftip = np.zeros(6, dtype=np.float32)
    
    return inverse_Dynamics(S, MList, GList, thetaList, thetaDotList, thetaDotDotList, g, Ftip)
    
def gravitionalForce_Vector(S, MList, GList, thetaList, g):
    """Compute gravitional force vector using inverse dynamic func
        S: 6xn twist matrix
        Mlist: 4x4xn Home matrix with respect to each consecutive joint
        Glist: 6x6xn inertia space matrix with respect to each movable link (dynamic link)
        thetaList: 4x1 angle/velocity of each joint
        g: earth gravitional force with respect to base frame
        
        return: gravitational force vector with each element is gravitional force each joint applied
    """
    thetaDotList = np.zeros_like(thetaList)
    thetaDotDotList = np.zeros_like(thetaList)
    Ftip = np.zeros(6, dtype=np.float32)
    
    return inverse_Dynamics(S, MList, GList, thetaList, thetaDotList, thetaDotDotList, g, Ftip)

def EEWrenchForces_Vector(S, MList, GList, thetaList, Ftip):
    """Compute gravitional force vector using inverse dynamic func
        S: 6xn twist matrix
        Mlist: 4x4xn Home matrix with respect to each consecutive joint
        Glist: 6x6xn inertia space matrix with respect to each movable link (dynamic link)
        thetaList: 4x1 angle/velocity of each joint
        Ftip: EE external applied force with respect to base frame
        
        return: wrench force vector applied to EE 
    """
    thetaDotList = np.zeros_like(thetaList)
    thetaDotDotList = np.zeros_like(thetaDotList)
    g = np.zeros(3, dtype=np.float32)
    
    return inverse_Dynamics(S, MList, GList, thetaList, thetaDotList, thetaDotDotList, g, Ftip)