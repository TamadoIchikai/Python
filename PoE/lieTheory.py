import numpy as np
import PoE.helper as helper
   
tol = 1e-5

adjoint_T = lambda T: (
    (lambda R, p: np.vstack([
        np.hstack([R, np.zeros((3,3))]),
        np.hstack([vectorTo_so3(p.ravel()) @ R, R])
    ]))(*helper.TransMatTo_Rp(T))
)
"""
    T: 4x4 transformation matrix
    
    return: 6x6 adjoint theory for transformation matrix - AdjT = [R, 0; [p] @ R, R]
"""

adjoint_V = lambda V: np.vstack([
    np.hstack([vectorTo_so3(V[:3]), np.zeros((3,3))]),
    np.hstack([vectorTo_so3(V[3:]), vectorTo_so3(V[:3])])
])
"""
    V: 6x1 twist error matrix
    
    return: 6x6 adjoint form - AdjV = [[w], zeros(3,3); [v], [w]]
""" 

vectorTo_so3 = lambda v: np.array([[0, -v.ravel()[2], v.ravel()[1]],
                                   [v.ravel()[2], 0, -v.ravel()[0]],
                                   [-v.ravel()[1], v.ravel()[0], 0]], dtype=np.float32)
"""
    vector: 3x1 vector
    
    return: skew symmetric matrix representation of input vector
"""

so3To_vector = lambda S: np.array([S[2,1], S[0,2], S[1,0]]).reshape(3,1)
"""
    so3: skew symmetric matrix
    
    return: 3x1 vector representation
"""

twistTo_se3 = lambda S: np.vstack([np.hstack([vectorTo_so3(S[:3]),S[3:].reshape(3,1)]), np.zeros((1,4))])
"""
    S: 6x1 twist matrix 
    
    return: se(3) matrix
"""

se3To_Twist = lambda se3: np.vstack([so3To_vector(se3[:3, :3]), se3[:3,3].reshape(3,1)]).ravel()
"""
    se3: se(3) matrix
    
    return: 6x1 twist matrix
"""

def compute_ScrewMat(w, q, n=None):
    """
        w: angular velocity omega, 
        q translation distance 
        n: prismatic joint position [joint position, axis_code] where axis_code ∈ {-3, -2, -1, 1, 2, 3} for -z, -y, -x, x, y, z
        Notice that if n=0 then this robot is pure revolute joint
        
        return: twist matrix S
    """ 
    n_joint = w.shape[1]
    
    S = np.zeros((6,n_joint))
    
    for i in range(n_joint):
        S[:,i] = np.hstack([w[:, i], np.cross(-w[:, i], q[:, i])])
    
    if n is None:
        return S
    else:
        n = np.array(n, dtype=int).ravel()
        assert n.size == 2
        
        axis_code = {
                1:  np.array([[0,0,0, 1,0,0]]).T,
                -1: np.array([[0,0,0,-1,0,0]]).T,
                2:  np.array([[0,0,0, 0,1,0]]).T,
                -2: np.array([[0,0,0, 0,-1,0]]).T,
                3:  np.array([[0,0,0, 0,0,1]]).T,
                -3: np.array([[0,0,0, 0,0,-1]]).T,
        }
        
        if n[1] not in axis_code:
            raise ValueError("Invalid n. Use ±1 (x), ±2 (y), ±3 (z).")
        else:
            S[:, n[0]] = axis_code[n[1]].ravel()
        return S

   
def exp_se3(se3):
    """With se(3) include: 
        [:3, :3]: [w] .* theta,
        [:3, 3]: v .* theta 
        
        return: exponential map of se(3) = SE(3)
    """
    omegatheta = so3To_vector(se3[0:3, 0:3]).ravel()
    theta, omegaHatVector = helper.vectorTo_angle(omegatheta)

    if np.isclose(theta, 0, atol=tol):
        return np.vstack([np.hstack([np.eye(3), se3[:3, 3].reshape(3,1)]), np.hstack([np.zeros((1,3)), [[1]]])])

    else:
        omegaHatMat = vectorTo_so3(omegaHatVector)
        R = exp_so3(se3[:3, :3])
        p = (np.eye(3) * theta + (1 - np.cos(theta)) * omegaHatMat + (theta - np.sin(theta)) * omegaHatMat @ omegaHatMat) @ se3[:3, 3]/theta
        return np.vstack([np.hstack([R, p.reshape(3,1)]), np.hstack([np.zeros((1,3)), [[1]]])])


def exp_so3(so3):
    """Using rodrigue closed form 
        so(3): 3x3 matrix [w] .* theta) 
        
        return: exponential map of so(3) = SO(3)
    """
    omegaTheta = so3To_vector(so3)
    theta, omegaHatVector = helper.vectorTo_angle(omegaTheta)
    omegaHatMat = vectorTo_so3(omegaHatVector)

    # I don't check if theta is closed to zeros because function exp_so3() already checked it
    return np.eye(3) + np.sin(theta) * omegaHatMat + (1 - np.cos(theta)) * omegaHatMat @ omegaHatMat

def log_SE3(T):
    """
        T: SE(3) matrix
        
        return: logarithm of a SE(3) matrix = se(3)
    """
    R, p = helper.TransMatTo_Rp(T)
    
    omegatheta = log_SO3(R)
    if np.allclose(omegatheta, 0, atol=tol):
        return np.vstack([np.hstack([np.zeros((3,3)), p.reshape(3,1)]), np.zeros((1,4))])
    else:
        theta = np.acos((np.trace(R)-1)/2)
        cot_half = np.cos(theta / 2.0) / np.sin(theta / 2.0)
        p = (np.eye(3,3) - omegatheta/2 + (1/theta - cot_half/2) * (omegatheta @ omegatheta)/theta) @ p.reshape(3,1)
        return np.vstack([np.hstack([omegatheta, p]), np.zeros((1,4))])

def log_SO3(R):
    """
       R: SO(3) orientation matrix
      
       return: logarithm of SO(3) (inverse exp_so3) = so(3)
    """
    acosInput = (np.trace(R) - 1)/2
    
    if acosInput >= 1:
        return np.zeros((3,3))
    elif acosInput <= -1:
        if (1+R[0,0]) < tol:
            omg = (1/np.sqrt(2*(1+R[0,0]))) * [1+R[0,0], R[1,0], R[2,0]]
        elif (1+R[1,1]) < tol:
            omg = (1/np.sqrt(2*(1+R[1,1]))) * [R[0,1], 1+R[1,1], R[2,1]]
        else:
            omg = (1/np.sqrt(2*(1+R[2,2]))) * [R[0,2], R[1,2], 1+R[2,2]]
        return vectorTo_so3(np.pi * omg)
    else:
        theta = np.acos(acosInput)
        return (theta * (1/(2*np.sin(theta)))) * (R-R.T)