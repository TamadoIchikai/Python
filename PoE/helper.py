import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

tol = 1e-5

@njit(cache=True)
def discrete_Integrator(x_prev, u, Ts):
    """
    Stateless integrator (forward Euler).
    x_prev: numpy array shape (n,) or scalar
    u: numpy array same shape as x_prev or scalar
    Ts: sample time (scalar)
    returns x_next with same shape as x_prev
    """
    return x_prev + Ts * u

@njit(cache=True)
def discrete_Derivative(x_prev, x_curr, Ts):
    """
    Stateless discrete derivative (backward Euler).
    x_prev: numpy array shape (n,) or scalar  (state at k-1)
    x_curr: numpy array same shape           (state at k)
    Ts: sample time (scalar)
    returns derivative estimate x_dot with same shape
    """
    return (x_curr - x_prev) / Ts

@njit(cache=True)
def TransMatTo_Rp(T):
    """
    T: 4x4 transformation matrix
    
    return: 3x3 orientation matrix R and 3x1 translation matrix p 
    """
    R = T[:3, :3].copy()
    p = T[:3, 3].copy()
    return R, p

@njit(cache=True)
def RpTo_TransMat(R, p):
    """
    R: SO(3) orientation matrix
    p: 3x1 translation matrix
    
    return: SE(3)
    """
    result = np.zeros((4, 4), dtype=np.float64)
    result[:3, :3] = R
    p_flat = p.ravel()
    result[0, 3] = p_flat[0]
    result[1, 3] = p_flat[1]
    result[2, 3] = p_flat[2]
    result[3, 3] = 1.0
    return result

@njit(cache=True)
def transInv_SE3(T):
    """Inverse a SE(3) matrix using transpose(SO(3)) matrix for more efficiency
    T: SE(3) matrix
    
    return: inv(T)
    """
    R = T[:3, :3].copy()
    p = T[:3, 3].copy()
    
    R_T = R.T
    p_new = -R_T @ p
    
    result = np.zeros((4, 4), dtype=np.float64)
    result[:3, :3] = R_T
    result[0, 3] = p_new[0]
    result[1, 3] = p_new[1]
    result[2, 3] = p_new[2]
    result[3, 3] = 1.0
    return result

@njit(cache=True)
def vectorTo_angle(omegaTheta):
    """
    omegaTheta: 3x1 w .* theta
    
    return: 
        constant theta
        3x1 w hat
    """
    omegaTheta_flat = omegaTheta.ravel()
    theta = np.sqrt(omegaTheta_flat[0]**2 + omegaTheta_flat[1]**2 + omegaTheta_flat[2]**2)
    omegaHat = np.zeros(3, dtype=np.float64)
    if theta > tol:
        omegaHat[0] = omegaTheta_flat[0] / theta
        omegaHat[1] = omegaTheta_flat[1] / theta
        omegaHat[2] = omegaTheta_flat[2] / theta
    return theta, omegaHat

@njit(cache=True)
def dls_inverse(J, lambda_val=1e-4):
    """
    Damped Least-Squares inverse of a Jacobian.
    Handles tall (m>=n) and wide (m<n) Jacobians.

    Args:
        J : Jacobian matrix (m x n)
        lambda_val : damping coefficient (scalar)

    Returns:
        J_inv : DLS inverse (n x m)
    """
    lam2 = lambda_val * lambda_val
    m, n = J.shape

    if m >= n:
        # Overdetermined or square: J⁺ = (Jᵀ J + λ² I)⁻¹ Jᵀ
        A = J.T @ J + lam2 * np.eye(n)
        J_inv = np.linalg.solve(A, J.T)
    else:
        # Underdetermined: J⁺ = Jᵀ (J Jᵀ + λ² I)⁻¹
        B = J @ J.T + lam2 * np.eye(m)
        J_inv = J.T @ np.linalg.solve(B, np.eye(m))

    return J_inv

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
def mcI(m, c, I):
    """Compute 6x6 inertia space matrix including parallel axis acounted for offset from CoM
    m: link's weight
    c: displacement from CoM to rotation axis relative to it's frame
    I: 3x3 rotational inertia about CoM
    
    return: 6x6 spatial inertia matrix
    """
    cHat = vectorTo_so3(c)
    cHat_cHatT = cHat @ cHat.T
    
    result = np.zeros((6, 6), dtype=np.float64)
    result[:3, :3] = I + m * cHat_cHatT
    result[:3, 3:6] = m * cHat
    result[3:6, :3] = -m * cHat
    result[3:6, 3:6] = m * np.eye(3, dtype=np.float64)
    return result

# Validation functions - not JIT compiled as they're for debugging
def validate_theta_Pose(theta_Pose):
    theta_Pose = np.asarray(theta_Pose, dtype=np.float64).ravel()
    n_joint = theta_Pose.size
    return theta_Pose, n_joint

def validate_q(q, n_joint):
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (3, n_joint):
        raise ValueError(f"q must be shape (3, {n_joint}), got {q.shape}")
    return q

def validate_w(w, n_joint):
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (3, n_joint):
        raise ValueError(f"w must be shape (3, {n_joint}), got {w.shape}")
    return w

def validate_M(M):
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"M must be shape (4, 4), got {M.shape}")
    return M

def validate_S(S, n_joint):
    S = np.asarray(S, dtype=np.float64)
    if S.shape != (6, n_joint):
        raise ValueError(f"S must be shape (6, {n_joint}), got {S.shape}")
    return S

def validate_joint_inputs(q, w, M, S, theta_Pose):
    theta_Pose, n_joint = validate_theta_Pose(theta_Pose)
    q = validate_q(q, n_joint)
    w = validate_w(w, n_joint)
    M = validate_M(M)
    S = validate_S(S, n_joint)
    return q, w, M, S, theta_Pose

def validate_Ftip(Ftip):
    Ftip = np.asarray(Ftip, dtype=np.float64).ravel()
    if Ftip.shape != (6,):
        raise ValueError(f"Ftip must be shape (6,), got {Ftip.shape}")
    return Ftip

def validate_g(g):
    g = np.asarray(g, dtype=np.float64).ravel()
    if g.shape != (3,):
        raise ValueError(f"g must be shape (3,), got {g.shape}")
    return g

def validate_MList(MList, n_joint):
    MList = np.asarray(MList, dtype=np.float64)
    if MList.shape != (4, 4, n_joint+1):
        raise ValueError(f"MList must be shape (4, 4, {n_joint+1}), got {MList.shape}")
    return MList

def validate_GList(GList, n_joint):
    GList = np.asarray(GList, dtype=np.float64)
    if GList.shape != (6, 6, n_joint):
        raise ValueError(f"GList must be shape (6, 6, {n_joint}), got {GList.shape}")
    return GList

def validate_Dynamics_Inputs(Ftip, g, MList, GList, theta_Pose):
    theta_Pose = np.asarray(theta_Pose, dtype=np.float64).ravel()
    n_joint = theta_Pose.size
    Ftip = validate_Ftip(Ftip)
    g = validate_g(g)
    MList = validate_MList(MList, n_joint)
    GList = validate_GList(GList, n_joint)
    return Ftip, g, MList, GList, theta_Pose

def plot_Continuous(posInput_log, posOutput_log, tVec, extras=None,
                    figsize=(14, 8), max_cols=3, save_path=None):
    posInput_log = np.asarray(posInput_log)
    posOutput_log = np.asarray(posOutput_log)
    tVec = np.asarray(tVec)
    if extras is None:
        extras = {}
    else:
        extras = {k: np.asarray(v) for k, v in extras.items()}

    lengths = [len(tVec), len(posInput_log), len(posOutput_log)]
    for arr in extras.values():
        lengths.append(len(arr))
    n = min(lengths)
    if n == 0:
        raise ValueError("No data points to plot.")
    if any(L != n for L in lengths):
        print(f"Warning: length mismatch detected, trimming to {n} samples.")
        t = tVec[:n]
        posIn = posInput_log[:n]
        posOut = posOutput_log[:n]
        extras = {k: v[:n] for k, v in extras.items()}
    else:
        t = tVec
        posIn = posInput_log
        posOut = posOutput_log

    xin, yin, zin = posIn[:, 0], posIn[:, 1], posIn[:, 2]
    xout, yout, zout = posOut[:, 0], posOut[:, 1], posOut[:, 2]
    error = posOut - posIn
    ex, ey, ez = error[:, 0], error[:, 1], error[:, 2]

    extra_series = []
    for name, arr in extras.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            extra_series.append((name, arr.ravel()))
        elif arr.ndim == 2:
            cols = arr.shape[1]
            for c in range(cols):
                label = f"{name} (c{c})"
                extra_series.append((label, arr[:, c]))
        else:
            reshaped = arr.reshape(arr.shape[0], -1)
            cols = reshaped.shape[1]
            for c in range(cols):
                label = f"{name} (c{c})"
                extra_series.append((label, reshaped[:, c]))

    base_rows = 2
    base_cols = 3
    n_extra = len(extra_series)
    if n_extra == 0:
        total_rows = base_rows
        total_cols = base_cols
    else:
        extra_rows = int(np.ceil(n_extra / float(max_cols)))
        total_rows = int(base_rows + extra_rows)
        total_cols = int(max(base_cols, max_cols))

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(total_rows, total_cols, hspace=1.0, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, xin, label="xin")
    ax1.plot(t, xout, label="xout")
    ax1.set_title("X input vs X output"); ax1.set_xlabel("Time (s)"); ax1.set_ylabel("X (m)")
    ax1.grid(True); ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, yin, label="yin"); ax2.plot(t, yout, label="yout")
    ax2.set_title("Y input vs Y output"); ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Y (m)")
    ax2.grid(True); ax2.legend()

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, zin, label="zin"); ax3.plot(t, zout, label="zout")
    ax3.set_title("Z input vs Z output"); ax3.set_xlabel("Time (s)"); ax3.set_ylabel("Z (m)")
    ax3.grid(True); ax3.legend()

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, ex, label="x error")
    ax4.set_title("X-axis Error"); ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Error (m)")
    ax4.grid(True); ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, ey, label="y error")
    ax5.set_title("Y-axis Error"); ax5.set_xlabel("Time (s)"); ax5.set_ylabel("Error (m)")
    ax5.grid(True); ax5.legend()

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, ez, label="z error")
    ax6.set_title("Z-axis Error"); ax6.set_xlabel("Time (s)"); ax6.set_ylabel("Error (m)")
    ax6.grid(True); ax6.legend()

    if n_extra:
        start_idx = base_rows
        for idx, (label, series) in enumerate(extra_series):
            row = start_idx + (idx // max_cols)
            col = idx % max_cols
            row = int(row); col = int(col)
            ax = fig.add_subplot(gs[row, col])
            ax.plot(t, series, label=label)
            ax.set_title(label); ax.set_xlabel("Time (s)")
            ax.grid(True); ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return None
