import PoE.helper as helper
import PoE.lieTheory as lie
import numpy as np

def PoE_transform(S, M, thetaList):
    """
        S: 6xn twist matrix 
        M: home matrix belong to SE(3)
        thetaList: nx1 rotated angle theta 
        
        return: transformation matrix T belong to SE(3)
    """
    T = np.eye(4)
    for i in range(S.shape[1]):
        se3 = lie.twistTo_se3(S[:,i]) * thetaList[i]
        T = T @ lie.exp_se3(se3) 
    return T @ M
 
def jacobian_Space(S, thetaList):
    """
        S: 6xn twist matrix with n is number of joints
        thetaList: 1D array with each element is joint's variable
        
        return: 6xn jacobian space matrix
    """
    Js = S.copy()
    T = np.eye(4,4, dtype=np.float32)
    
    for i in range(1, len(thetaList)):
        T = T @ (lie.exp_se3(lie.twistTo_se3(S[:, i-1]*thetaList[i-1])))
        Js[:, i] = lie.adjoint_T(T) @ S[:, i]
        
    return Js
    

twist_Error = lambda Tsb, T_d: lie.adjoint_T(Tsb) @ (lie.se3To_Twist(lie.log_SE3(helper.transInv_SE3(Tsb) @ T_d)))
"""
    Tsb: SE(3) matrix - current robot transformation matrix
    T_d: SE(3) matrix - desired robot transformation matrix
    
    return: 6x1 twist error matrix Vs including [w; v]
"""