import numpy as np
from numba import njit, prange

tol = 1e-5

@njit(cache=True)
def adjoint_T(T):
    """
    T: 4x4 transformation matrix
    
    return: 6x6 adjoint theory for transformation matrix - AdjT = [R, 0; [p] @ R, R]
    """
    R = T[:3, :3].copy()
    p = T[:3, 3].copy()
    
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = R
    # [p] @ R
    p_skew = vectorTo_so3(p)
    result[3:6, :3] = p_skew @ R
    result[3:6, 3:6] = R
    return result

@njit(cache=True)
def adjoint_V(V):
    """
    V: 6x1 twist error matrix
    
    return: 6x6 adjoint form - AdjV = [[w], zeros(3,3); [v], [w]]
    """
    result = np.zeros((6, 6), dtype=np.float64)
    w_skew = vectorTo_so3(V[:3])
    v_skew = vectorTo_so3(V[3:])
    result[:3, :3] = w_skew
    result[3:6, :3] = v_skew
    result[3:6, 3:6] = w_skew
    return result

@njit(cache=True)
def vectorTo_so3(v):
    """
    vector: 3x1 vector
    
    return: skew symmetric matrix representation of input vector
    """
    v_flat = v.ravel()
    result = np.zeros((3, 3), dtype=np.float64)
    result[0, 1] = -v_flat[2]
    result[0, 2] = v_flat[1]
    result[1, 0] = v_flat[2]
    result[1, 2] = -v_flat[0]
    result[2, 0] = -v_flat[1]
    result[2, 1] = v_flat[0]
    return result

@njit(cache=True)
def so3To_vector(S):
    """
    so3: skew symmetric matrix
    
    return: 3x1 vector representation
    """
    result = np.zeros(3, dtype=np.float64)
    result[0] = S[2, 1]
    result[1] = S[0, 2]
    result[2] = S[1, 0]
    return result

@njit(cache=True)
def twistTo_se3(S):
    """
    S: 6x1 twist matrix 
    
    return: se(3) matrix
    """
    result = np.zeros((4, 4), dtype=np.float64)
    result[:3, :3] = vectorTo_so3(S[:3])
    result[0, 3] = S[3]
    result[1, 3] = S[4]
    result[2, 3] = S[5]
    return result

@njit(cache=True)
def se3To_Twist(se3):
    """
    se3: se(3) matrix
    
    return: 6x1 twist matrix
    """
    result = np.zeros(6, dtype=np.float64)
    vec = so3To_vector(se3[:3, :3])
    result[0] = vec[0]
    result[1] = vec[1]
    result[2] = vec[2]
    result[3] = se3[0, 3]
    result[4] = se3[1, 3]
    result[5] = se3[2, 3]
    return result

@njit(cache=True)
def compute_ScrewMat(w, q, n_prismatic_joint=-1, n_prismatic_axis=0):
    """
    w: angular velocity omega, 
    q translation distance 
    n_prismatic_joint: prismatic joint index (-1 means none)
    n_prismatic_axis: axis_code ∈ {-3, -2, -1, 1, 2, 3} for -z, -y, -x, x, y, z
    
    return: twist matrix S
    """ 
    n_joint = w.shape[1]
    S = np.zeros((6, n_joint), dtype=np.float64)
    
    for i in range(n_joint):
        if i == n_prismatic_joint:
            # Prismatic joint: no rotation, only translation
            # Extract direction from axis code
            if n_prismatic_axis == 1:  # +x
                S[3, i] = 1.0
            elif n_prismatic_axis == -1:  # -x
                S[3, i] = -1.0
            elif n_prismatic_axis == 2:  # +y
                S[4, i] = 1.0
            elif n_prismatic_axis == -2:  # -y
                S[4, i] = -1.0
            elif n_prismatic_axis == 3:  # +z
                S[5, i] = 1.0
            elif n_prismatic_axis == -3:  # -z
                S[5, i] = -1.0
        else:
            # Revolute joint
            w_i = w[:, i]
            q_i = q[:, i]
            S[:3, i] = w_i
            v = np.cross(-w_i, q_i)
            S[3:, i] = v
    
    return S


@njit(cache=True)
def vectorTo_angle(omegaTheta):
    """
    omegaTheta: 3x1 w .* theta
    
    return: 
        constant theta
        3x1 w hat
    """
    theta = np.sqrt(omegaTheta[0]**2 + omegaTheta[1]**2 + omegaTheta[2]**2)
    omegaHat = np.zeros(3, dtype=np.float64)
    if theta > tol:
        omegaHat[0] = omegaTheta[0] / theta
        omegaHat[1] = omegaTheta[1] / theta
        omegaHat[2] = omegaTheta[2] / theta
    return theta, omegaHat

@njit(cache=True)
def exp_se3(se3):
    """With se(3) include: 
        [:3, :3]: [w] .* theta,
        [:3, 3]: v .* theta 
        
        return: exponential map of se(3) = SE(3)
    """
    omegatheta = so3To_vector(se3[:3, :3])
    theta, omegaHatVector = vectorTo_angle(omegatheta)

    result = np.zeros((4, 4), dtype=np.float64)
    result[3, 3] = 1.0

    if theta < tol:
        result[0, 0] = 1.0
        result[1, 1] = 1.0
        result[2, 2] = 1.0
        result[0, 3] = se3[0, 3]
        result[1, 3] = se3[1, 3]
        result[2, 3] = se3[2, 3]
        return result

    omegaHatMat = vectorTo_so3(omegaHatVector)
    R = exp_so3(se3[:3, :3])
    
    omegaHatMat2 = omegaHatMat @ omegaHatMat
    G = np.eye(3) * theta + (1.0 - np.cos(theta)) * omegaHatMat + (theta - np.sin(theta)) * omegaHatMat2
    
    v = se3[:3, 3]
    p = G @ v / theta
    
    result[:3, :3] = R
    result[0, 3] = p[0]
    result[1, 3] = p[1]
    result[2, 3] = p[2]
    return result

@njit(cache=True)
def exp_so3(so3):
    """Using rodrigue closed form 
        so(3): 3x3 matrix [w] .* theta) 
        
        return: exponential map of so(3) = SO(3)
    """
    omegaTheta = so3To_vector(so3)
    theta, omegaHatVector = vectorTo_angle(omegaTheta)
    
    if theta < tol:
        return np.eye(3, dtype=np.float64)
    
    omegaHatMat = vectorTo_so3(omegaHatVector)
    omegaHatMat2 = omegaHatMat @ omegaHatMat
    
    return np.eye(3, dtype=np.float64) + np.sin(theta) * omegaHatMat + (1.0 - np.cos(theta)) * omegaHatMat2

@njit(cache=True)
def log_SE3(T):
    """
    T: SE(3) matrix
    
    return: logarithm of a SE(3) matrix = se(3)
    """
    R = T[:3, :3].copy()
    p = T[:3, 3].copy()
    
    omegatheta = log_SO3(R)
    
    # Check if omegatheta is all zeros
    norm_omega = 0.0
    for i in range(3):
        for j in range(3):
            norm_omega += omegatheta[i, j] ** 2
    
    result = np.zeros((4, 4), dtype=np.float64)
    
    if norm_omega < tol * tol:
        result[0, 3] = p[0]
        result[1, 3] = p[1]
        result[2, 3] = p[2]
        return result
    
    trace_R = R[0, 0] + R[1, 1] + R[2, 2]
    acos_input = (trace_R - 1.0) / 2.0
    if acos_input > 1.0:
        acos_input = 1.0
    elif acos_input < -1.0:
        acos_input = -1.0
    theta = np.arccos(acos_input)
    
    cot_half = np.cos(theta / 2.0) / np.sin(theta / 2.0)
    omegatheta2 = omegatheta @ omegatheta
    
    G_inv = np.eye(3, dtype=np.float64) - omegatheta / 2.0 + (1.0 / theta - cot_half / 2.0) * omegatheta2 / theta
    p_new = G_inv @ p
    
    result[:3, :3] = omegatheta
    result[0, 3] = p_new[0]
    result[1, 3] = p_new[1]
    result[2, 3] = p_new[2]
    return result

@njit(cache=True)
def log_SO3(R):
    """
    R: SO(3) orientation matrix
    
    return: logarithm of SO(3) (inverse exp_so3) = so(3)
    """
    trace_R = R[0, 0] + R[1, 1] + R[2, 2]
    acosInput = (trace_R - 1.0) / 2.0
    
    if acosInput >= 1.0:
        return np.zeros((3, 3), dtype=np.float64)
    elif acosInput <= -1.0:
        omg = np.zeros(3, dtype=np.float64)
        if (1.0 + R[2, 2]) > tol:
            omg[0] = R[0, 2]
            omg[1] = R[1, 2]
            omg[2] = 1.0 + R[2, 2]
            omg = omg / np.sqrt(2.0 * (1.0 + R[2, 2]))
        elif (1.0 + R[1, 1]) > tol:
            omg[0] = R[0, 1]
            omg[1] = 1.0 + R[1, 1]
            omg[2] = R[2, 1]
            omg = omg / np.sqrt(2.0 * (1.0 + R[1, 1]))
        else:
            omg[0] = 1.0 + R[0, 0]
            omg[1] = R[1, 0]
            omg[2] = R[2, 0]
            omg = omg / np.sqrt(2.0 * (1.0 + R[0, 0]))
        return vectorTo_so3(np.pi * omg)
    else:
        theta = np.arccos(acosInput)
        factor = theta / (2.0 * np.sin(theta))
        R_diff = R - R.T
        return factor * R_diff