import os
import numpy as np
import PoE.helper as helper

from numba import njit
from scipy.interpolate import RegularGridInterpolator

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

    def save_npz(self, theta_N, fileNamePath, nE=1001, nDE=1001):
        e_vec, de_vec, Y_kp, Y_kd = self.evaluate(nE=nE, nDE=nDE)

        np.savez(
            fileNamePath,
            **{
                f"evec_Theta_{theta_N}": e_vec,
                f"devec_Theta_{theta_N}": de_vec,
                f"Ymat_dKp_Theta_{theta_N}": Y_kp,
                f"Ymat_dKd_Theta_{theta_N}": Y_kd,
            },
        )
        

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
    mu_e = trimf_batch(e_vec, mfs_e)    # (7, N)
    mu_de = trimf_batch(de_vec, mfs_de)  # (7, M)
    
    N = mu_e.shape[1]  # num e samples
    M = mu_de.shape[1]  # num de samples
    
    # Initialize output matrices
    kp = np.zeros((M, N), dtype=np.float64)
    kd = np.zeros((M, N), dtype=np.float64)
    
    # For each output grid point
    for m in range(M):
        for n in range(N):
            # Compute rule weights: w[i,j] = min(mu_e[i,n], mu_de[j,m])
            kp_num = 0.0
            kd_num = 0.0
            denom = 0.0
            
            for i in range(7):
                for j in range(7):
                    w_ij = min(mu_e[i, n], mu_de[j, m])
                    kp_num += w_ij * kp_const[i, j]
                    kd_num += w_ij * kd_const[i, j]
                    denom += w_ij
            
            # Weighted average (wtaver)
            denom = max(denom, 1e-12)  # Avoid division by zero
            kp[m, n] = kp_num / denom
            kd[m, n] = kd_num / denom
    
    return kp, kd

def build_const_matrix(table, map_val):
    out = np.zeros((7,7), dtype=np.float64)
    for i in range(7):
        for j in range(7):
            out[i, j] = map_val[table[i][j]]
    return out


class LookupTable2D:
    """
    2-D Lookup Table with linear interpolation and clip extrapolation.
    Mimics MATLAB/Simulink's 2-D Lookup Table block.
    """
    def __init__(self, breakpoints1, breakpoints2, table_data):
        """
        Parameters
        ----------
        breakpoints1 : array-like
            Breakpoints for first input (e.g., evec_Theta1). Shape: (M,)
        breakpoints2 : array-like
            Breakpoints for second input (e.g., devec_Theta1). Shape: (N,)
        table_data : array-like
            2-D table data. Shape: (N, M) where rows correspond to breakpoints2
            and columns correspond to breakpoints1.
        """
        self.bp1 = np.asarray(breakpoints1).ravel()
        self.bp2 = np.asarray(breakpoints2).ravel()
        self.table = np.asarray(table_data)
        
        # Create interpolator with linear method and clipping bounds
        # Note: RegularGridInterpolator expects table shape to match (len(bp2), len(bp1))
        self._interp = RegularGridInterpolator(
            (self.bp2, self.bp1),  # (rows, cols) = (de, e)
            self.table,
            method='linear',
            bounds_error=False,
            fill_value=None  # Use nearest for extrapolation
        )
    
    def __call__(self, u1, u2):
        """
        Evaluate the lookup table.
        
        Parameters
        ----------
        u1 : float or array-like
            First input (corresponds to breakpoints1, e.g., e_theta)
        u2 : float or array-like
            Second input (corresponds to breakpoints2, e.g., de_theta)
        
        Returns
        -------
        float or ndarray
            Interpolated output value(s)
        """
        u1 = np.asarray(u1)
        u2 = np.asarray(u2)
        
        # Clip inputs to breakpoint ranges (extrapolation = clip)
        u1_clipped = np.clip(u1, self.bp1[0], self.bp1[-1])
        u2_clipped = np.clip(u2, self.bp2[0], self.bp2[-1])
        
        # Handle scalar vs array inputs
        scalar_input = u1.ndim == 0 and u2.ndim == 0
        
        if scalar_input:
            pts = np.array([[u2_clipped, u1_clipped]])
        else:
            u1_clipped = np.atleast_1d(u1_clipped)
            u2_clipped = np.atleast_1d(u2_clipped)
            pts = np.column_stack([u2_clipped.ravel(), u1_clipped.ravel()])
        
        result = self._interp(pts)
        
        if scalar_input:
            return float(result[0])
        return result.reshape(u1.shape) if u1.ndim > 0 else result.reshape(u2.shape)


class FuzzyLookupController:
    """
    Controller using pre-computed fuzzy lookup tables for dKp and dKd.
    """
    def __init__(self, npz_path):
        """
        Load lookup tables from .npz file.
        
        Parameters
        ----------
        npz_path : str
            Path to .npz file containing evec, devec, Ymat_dKp, Ymat_dKd
        """
        data = np.load(npz_path)
        
        # Detect key naming convention
        keys = list(data.keys())
        
        # Find evec and devec keys
        evec_key = [k for k in keys if 'evec' in k.lower()][0]
        devec_key = [k for k in keys if 'devec' in k.lower()][0]
        dkp_key = [k for k in keys if 'dkp' in k.lower()][0]
        dkd_key = [k for k in keys if 'dkd' in k.lower()][0]
        
        evec = data[evec_key]
        devec = data[devec_key]
        Ymat_dKp = data[dkp_key]
        Ymat_dKd = data[dkd_key]
        
        self.lut_dKp = LookupTable2D(evec, devec, Ymat_dKp)
        self.lut_dKd = LookupTable2D(evec, devec, Ymat_dKd)
    
    def get_gains(self, e, de):
        """
        Get dKp and dKd for given error and error derivative.
        
        Parameters
        ----------
        e : float or array-like
            Error value(s)
        de : float or array-like
            Error derivative value(s)
        
        Returns
        -------
        tuple
            (dKp, dKd) values
        """
        dKp = self.lut_dKp(e, de)
        dKd = self.lut_dKd(e, de)
        return dKp, dKd

