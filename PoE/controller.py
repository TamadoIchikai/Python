import numpy as np
from numba import njit
import PoE.helper as helper

class PID_Discrete:
    def __init__(
        self,
        Kp,
        Ki,
        Kd,
        Ts,
        Kb=None,
        outputLimit=None,
        initial_integral=None,
        initial_prev_error=None
    ):
        """
        PID controller with anti-windup, back calculation, and initial state input.
        """
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.Kd = float(Kd)
        self.Ts = float(Ts)
        
        if outputLimit is not None:
            self.outputLimit_low = float(outputLimit[0])
            self.outputLimit_high = float(outputLimit[1])
            self.has_limit = True
        else:
            self.outputLimit_low = 0.0
            self.outputLimit_high = 0.0
            self.has_limit = False

        if Kb is None:
            if Ki > 0 and Kd > 0:
                self.Kb = 1.0 / np.sqrt(Ki * Kd)
            else:
                self.Kb = 1.0
        else:
            self.Kb = float(Kb)

        if initial_integral is not None:
            self._I = np.asarray(initial_integral, dtype=np.float64).ravel()
        else:
            self._I = None

        if initial_prev_error is not None:
            self._prev_error = np.asarray(initial_prev_error, dtype=np.float64).ravel()
        else:
            self._prev_error = None
        
        self._initialized = False

    def update(self, error):
        error = np.asarray(error, dtype=np.float64).ravel()

        # Initialize states on first call
        if self._I is None:
            self._I = np.zeros_like(error, dtype=np.float64)
        if self._prev_error is None:
            self._prev_error = error.copy()

        u_clipped, self._I, self._prev_error = pid_update(
            error, self._I, self._prev_error,
            self.Kp, self.Ki, self.Kd, self.Ts, self.Kb,
            self.outputLimit_low, self.outputLimit_high, self.has_limit
        )

        # Return scalar if input was scalar-like
        if u_clipped.size == 1:
            return u_clipped[0]
        return u_clipped

class FuzzySugeno:
    def __init__(self, e_range, de_range, ruleTable_dKp, ruleTable_dKd, map_val=None):
        self.e_range = e_range
        self.de_range = de_range
        self.map_val = {"ZO": 0.00, "S": 0.33, "M": 0.66, "L": 1.00} if map_val is None else map_val

        # Precompute membership functions
        self.mfs_e = gen7tri(e_range)
        self.mfs_de = gen7tri(de_range)

        # Precompute consequent constants
        self.kp_const = build_const_matrix(ruleTable_dKp, self.map_val)
        self.kd_const = build_const_matrix(ruleTable_dKd, self.map_val)

    def evaluate(self, nE=1001, nDE=1001):
        e_vec = np.linspace(self.e_range[0], self.e_range[1], nE)
        de_vec = np.linspace(self.de_range[0], self.de_range[1], nDE)
        Y_kp, Y_kd = eval_grid(e_vec, de_vec, self.mfs_e, self.mfs_de, self.kp_const, self.kd_const)
        return e_vec, de_vec, Y_kp, Y_kd

    def save_npz(self, path="FuzzySugeno_e_Theta1.npz", nE=1001, nDE=1001):
        e_vec, de_vec, Y_kp, Y_kd = self.evaluate(nE=nE, nDE=nDE)
        np.savez(path,
                 evec_Theta1=e_vec,
                 devec_Theta1=de_vec,
                 Ymat_dKp_Theta1=Y_kp,
                 Ymat_dKd_Theta1=Y_kd)

@njit(cache=True)
def pid_update(error, I_prev, prev_error, Kp, Ki, Kd, Ts, Kb, outputLimit_low, outputLimit_high, has_limit):
    """
    JIT-compiled PID update function.
    
    Returns: u_clipped, I_new, prev_error_new
    """
    # Update integral
    if Ki > 0:
        I_new = helper.discrete_Integrator(I_prev, error, Ts)
    else:
        I_new = I_prev.copy()
    
    # Derivative term
    d_err = helper.discrete_Derivative(prev_error, error, Ts)
    
    # PID output
    u = Kp * error + Ki * I_new + Kd * d_err
    
    # Apply output limits
    if has_limit:
        u_clipped = np.clip(u, outputLimit_low, outputLimit_high)
    else:
        u_clipped = u.copy()
    
    # Anti-windup correction
    if Ki > 0 and has_limit:
        aw_term = Kb * (u_clipped - u)
        I_new = I_new + aw_term
    
    return u_clipped, I_new, error.copy()



def gen7tri(rng):
    a, b = rng
    step = (b - a) / 6.0
    centers = np.linspace(a, b, 7)
    mfs = np.zeros((7, 3))
    for i in range(7):
        if i == 0:
            mfs[i] = [centers[i], centers[i], centers[i + 1]]
        elif i == 6:
            mfs[i] = [centers[i - 1], centers[i], centers[i]]
        else:
            mfs[i] = [centers[i - 1], centers[i], centers[i + 1]]
    return mfs

@njit
def trimf(x, abc):
    a, b, c = abc
    y = np.zeros_like(x)
    left = (a != b) & (x >= a) & (x < b)
    y[left] = (x[left] - a) / (b - a)
    center = (x == b)
    y[center] = 1.0
    right = (b != c) & (x > b) & (x <= c)
    y[right] = (c - x[right]) / (c - b)
    return y

@njit
def trimf_batch(x, mfs):
    # mfs: (7,3), x: (N,) -> (7,N)
    out = np.empty((mfs.shape[0], x.shape[0]), dtype=np.float64)
    for i in range(mfs.shape[0]):
        out[i] = trimf(x, mfs[i])
    return out

@njit
def eval_grid(e_vec, de_vec, mfs_e, mfs_de, kp_const, kd_const):
    mu_e = trimf_batch(e_vec, mfs_e)
    mu_de = trimf_batch(de_vec, mfs_de)
    M, N = mu_de.shape[1], mu_e.shape[1]
    w = np.minimum(mu_e[:, None, None, :], mu_de[None, :, :, None])  # (7,7,M,N)
    
    w_flat  = w.reshape(49, M, N)
    kp_flat = kp_const.reshape(49)
    kd_flat = kd_const.reshape(49)
    
    # Weighted average (wtaver): divide by sum of weights
    kp_num = np.sum(w_flat * kp_flat[:, None, None], axis=0)
    kd_num = np.sum(w_flat * kd_flat[:, None, None], axis=0)
    denom  = np.sum(w_flat, axis=0) + 1e-12  # Avoid division by zero
    
    kp = kp_num / denom
    kd = kd_num / denom
    return kp, kd

def build_const_matrix(table, map_val):
    out = np.zeros((7,7), dtype=np.float64)
    for i in range(7):
        for j in range(7):
            out[i, j] = map_val[table[i][j]]
    return out

