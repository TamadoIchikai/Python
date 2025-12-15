import numpy as np
from numba import njit, prange
import PoE.helper as helper
import PoE.lieTheory as lie

@njit(cache=True)
def PoE_transform(S, M, thetaList):
    """
    S: 6xn twist matrix 
    M: home matrix belong to SE(3)
    thetaList: nx1 rotated angle theta 
    
    return: transformation matrix T belong to SE(3)
    """
    T = np.eye(4, dtype=np.float64)
    n_joints = S.shape[1]
    
    for i in range(n_joints):
        S_i = S[:, i] * thetaList[i]
        se3 = lie.twistTo_se3(S_i)
        T = T @ lie.exp_se3(se3)
    
    return T @ M

@njit(cache=True)
def jacobian_Space(S, thetaList):
    """
    S: 6xn twist matrix with n is number of joints
    thetaList: 1D array with each element is joint's variable
    
    return: 6xn jacobian space matrix
    """
    n_joints = len(thetaList)
    Js = S.copy()
    T = np.eye(4, dtype=np.float64)
    
    for i in range(1, n_joints):
        S_prev = S[:, i-1] * thetaList[i-1]
        se3 = lie.twistTo_se3(S_prev)
        T = T @ lie.exp_se3(se3)
        adjT = lie.adjoint_T(T)
        Js[:, i] = adjT @ S[:, i]
    
    return Js

@njit(cache=True)
def twist_Error(Tsb, T_d):
    """
    Tsb: SE(3) matrix - current robot transformation matrix
    T_d: SE(3) matrix - desired robot transformation matrix
    
    return: 6x1 twist error matrix Vs including [w; v]
    """
    Tsb_inv = helper.transInv_SE3(Tsb)
    T_error = Tsb_inv @ T_d
    se3_error = lie.log_SE3(T_error)
    twist = lie.se3To_Twist(se3_error)
    adjTsb = lie.adjoint_T(Tsb)
    return adjTsb @ twist