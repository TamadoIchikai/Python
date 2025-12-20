import numpy as np
import time
import os
from numba import njit, prange

import PoE.lieTheory as lie
import PoE.kinematics as kine
import PoE.helper as helper
import PoE.dynamics as dyna
import PoE.controller as controller
from PoE.optimizer import GBO


# Fixed dKd rule table (symmetric, kept constant during optimization)
FIXED_RULE_TABLE_DKD = [
    ["L","M","M","S","M","M","L"],
    ["M","S","S","ZO","S","S","M"],
    ["M","S","ZO","ZO","ZO","S","M"],
    ["S","ZO","ZO","ZO","ZO","ZO","S"],
    ["M","S","ZO","ZO","ZO","S","M"],
    ["M","S","S","ZO","S","S","M"],
    ["L","M","M","S","M","M","L"],
]

# Output labels mapping (1-indexed for GBO, maps to string labels)
LABELS_OUT = ["ZO", "S", "M", "L"]  # indices 0,1,2,3 -> 1,2,3,4 in GBO

# Numeric mapping for rule values
MAP_VAL = {"ZO": 0.00, "S": 0.33, "M": 0.66, "L": 1.00}


def decode_rules_from_params(params: np.ndarray) -> list:
    """
    Convert 49 optimization parameters (values 1-4) to 7x7 rule table.
    """
    params_int = np.round(params).astype(int)
    params_int = np.clip(params_int, 1, 4)
    rule_indices = params_int.reshape(7, 7)
    
    rule_table = []
    for i in range(7):
        row = []
        for j in range(7):
            idx = rule_indices[i, j] - 1
            row.append(LABELS_OUT[idx])
        rule_table.append(row)
    
    return rule_table


def build_const_matrix_from_params(params: np.ndarray) -> np.ndarray:
    """
    Convert 49 optimization parameters directly to numeric constant matrix.
    Bypasses string labels for speed.
    """
    params_int = np.round(params).astype(int)
    params_int = np.clip(params_int, 1, 4)
    
    # Map: 1->0.00, 2->0.33, 3->0.66, 4->1.00
    val_map = np.array([0.00, 0.33, 0.66, 1.00], dtype=np.float64)
    
    out = np.zeros((7, 7), dtype=np.float64)
    for i in range(49):
        row = i // 7
        col = i % 7
        out[row, col] = val_map[params_int[i] - 1]
    
    return out


def build_fixed_dKd_matrix() -> np.ndarray:
    """Build the fixed dKd constant matrix."""
    out = np.zeros((7, 7), dtype=np.float64)
    for i in range(7):
        for j in range(7):
            out[i, j] = MAP_VAL[FIXED_RULE_TABLE_DKD[i][j]]
    return out


# Pre-compute fixed dKd matrix at module load
FIXED_DKD_MATRIX = build_fixed_dKd_matrix()


@njit(cache=True)
def gen7tri_numba(a: float, b: float) -> np.ndarray:
    """Generate 7 triangular membership function parameters."""
    step = (b - a) / 6.0
    centers = np.zeros(7, dtype=np.float64)
    for i in range(7):
        centers[i] = a + i * step
    
    mfs = np.zeros((7, 3), dtype=np.float64)
    for i in range(7):
        if i == 0:
            mfs[i, 0] = centers[0]
            mfs[i, 1] = centers[0]
            mfs[i, 2] = centers[1]
        elif i == 6:
            mfs[i, 0] = centers[5]
            mfs[i, 1] = centers[6]
            mfs[i, 2] = centers[6]
        else:
            mfs[i, 0] = centers[i - 1]
            mfs[i, 1] = centers[i]
            mfs[i, 2] = centers[i + 1]
    return mfs


@njit(cache=True)
def trimf_scalar(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership function for scalar input."""
    if x <= a:
        return 0.0
    elif x < b:
        if a == b:
            return 1.0
        return (x - a) / (b - a)
    elif x == b:
        return 1.0
    elif x < c:
        if b == c:
            return 1.0
        return (c - x) / (c - b)
    else:
        return 0.0


@njit(cache=True)
def eval_fuzzy_scalar(e: float, de: float, 
                      mfs_e: np.ndarray, mfs_de: np.ndarray,
                      kp_const: np.ndarray, kd_const: np.ndarray) -> tuple:
    """
    Evaluate fuzzy Sugeno controller for single (e, de) input.
    Returns (dKp, dKd) normalized to [0, 1].
    """
    # Compute membership values for e
    mu_e = np.zeros(7, dtype=np.float64)
    for i in range(7):
        mu_e[i] = trimf_scalar(e, mfs_e[i, 0], mfs_e[i, 1], mfs_e[i, 2])
    
    # Compute membership values for de
    mu_de = np.zeros(7, dtype=np.float64)
    for i in range(7):
        mu_de[i] = trimf_scalar(de, mfs_de[i, 0], mfs_de[i, 1], mfs_de[i, 2])
    
    # Sugeno inference (weighted average)
    num_kp = 0.0
    num_kd = 0.0
    denom = 0.0
    
    for i in range(7):
        for j in range(7):
            w = mu_e[j] * mu_de[i]  # AND = min or product; using product
            num_kp += w * kp_const[i, j]
            num_kd += w * kd_const[i, j]
            denom += w
    
    if denom > 1e-10:
        dKp = num_kp / denom
        dKd = num_kd / denom
    else:
        dKp = 0.5
        dKd = 0.5
    
    return dKp, dKd


@njit(cache=True)
def build_lookup_table(mfs_e: np.ndarray, mfs_de: np.ndarray,
                       kp_const: np.ndarray, kd_const: np.ndarray,
                       e_vec: np.ndarray, de_vec: np.ndarray) -> tuple:
    """
    Build 2D lookup tables for dKp and dKd.
    Returns (Y_kp, Y_kd) with shape (nDE, nE).
    """
    nE = e_vec.shape[0]
    nDE = de_vec.shape[0]
    
    Y_kp = np.zeros((nDE, nE), dtype=np.float64)
    Y_kd = np.zeros((nDE, nE), dtype=np.float64)
    
    for m in range(nDE):
        for n in range(nE):
            dKp, dKd = eval_fuzzy_scalar(e_vec[n], de_vec[m], mfs_e, mfs_de, kp_const, kd_const)
            Y_kp[m, n] = dKp
            Y_kd[m, n] = dKd
    
    return Y_kp, Y_kd


@njit(cache=True)
def interp2d_scalar(x: float, y: float, 
                    x_vec: np.ndarray, y_vec: np.ndarray, 
                    Z: np.ndarray) -> float:
    """
    Bilinear interpolation for scalar (x, y) input.
    x corresponds to columns (axis 1), y corresponds to rows (axis 0).
    """
    nx = x_vec.shape[0]
    ny = y_vec.shape[0]
    
    # Clip to bounds
    x_clipped = max(x_vec[0], min(x, x_vec[nx - 1]))
    y_clipped = max(y_vec[0], min(y, y_vec[ny - 1]))
    
    # Find indices
    ix = 0
    for i in range(nx - 1):
        if x_vec[i + 1] >= x_clipped:
            ix = i
            break
    else:
        ix = nx - 2
    
    iy = 0
    for i in range(ny - 1):
        if y_vec[i + 1] >= y_clipped:
            iy = i
            break
    else:
        iy = ny - 2
    
    # Interpolation weights
    dx = x_vec[ix + 1] - x_vec[ix]
    dy = y_vec[iy + 1] - y_vec[iy]
    
    if dx > 1e-12:
        tx = (x_clipped - x_vec[ix]) / dx
    else:
        tx = 0.0
    
    if dy > 1e-12:
        ty = (y_clipped - y_vec[iy]) / dy
    else:
        ty = 0.0
    
    # Bilinear interpolation
    z00 = Z[iy, ix]
    z01 = Z[iy, ix + 1]
    z10 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]
    
    z0 = z00 * (1.0 - tx) + z01 * tx
    z1 = z10 * (1.0 - tx) + z11 * tx
    
    return z0 * (1.0 - ty) + z1 * ty


@njit(cache=True)
def random_index_pair_numba(n: int, seed: int) -> np.ndarray:
    """Generate random index pairs for tic-tac-toe trajectory."""
    np.random.seed(seed)
    
    # Create unique pairs
    unique_pairs = np.zeros((9, 2), dtype=np.int64)
    idx = 0
    for i in range(3):
        for j in range(3):
            unique_pairs[idx, 0] = i
            unique_pairs[idx, 1] = j
            idx += 1
    
    n_blocks = (n + 8) // 9
    result = np.zeros((n_blocks * 9, 2), dtype=np.int64)
    
    for b in range(n_blocks):
        # Copy unique pairs
        block = unique_pairs.copy()
        
        # Fisher-Yates shuffle
        for i in range(8, 0, -1):
            j = int(np.random.rand() * (i + 1))
            temp0 = block[i, 0]
            temp1 = block[i, 1]
            block[i, 0] = block[j, 0]
            block[i, 1] = block[j, 1]
            block[j, 0] = temp0
            block[j, 1] = temp1
        
        for i in range(9):
            result[b * 9 + i, 0] = block[i, 0]
            result[b * 9 + i, 1] = block[i, 1]
    
    return result[:n]


@njit(cache=True)
def tic_tac_toe_gen_numba(random_pairs: np.ndarray, start_time: float, 
                          step_time: float, current_time: float) -> tuple:
    """Generate XY position from tic-tac-toe pattern."""
    x_list = np.array([0.4, 0.5, 0.6], dtype=np.float64)
    y_list = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    
    elapsed = current_time - start_time
    current_idx = int(max(0.0, elapsed) // step_time)
    
    n_pairs = random_pairs.shape[0]
    if current_idx >= n_pairs:
        current_idx = n_pairs - 1
    
    i0 = random_pairs[current_idx, 0]
    i1 = random_pairs[current_idx, 1]
    
    return x_list[i0], y_list[i1], current_idx


@njit(cache=True)
def z_axis_up_down_numba(z_init: float, z_reach: float, start_time: float,
                         step_time: float, current_time: float) -> float:
    """Generate Z position with up-down motion."""
    elapsed = current_time - start_time
    time_in_cycle = elapsed % step_time
    half_step = step_time / 2.0
    
    if time_in_cycle <= half_step:
        ratio = time_in_cycle / half_step
        z = z_init + (z_reach - z_init) * ratio
    else:
        time_in_part = time_in_cycle - half_step
        ratio = time_in_part / half_step
        z = z_reach - (z_reach - z_init) * ratio
    
    return z


@njit(cache=True)
def simulation_loop_fuzzy(
    # Robot configuration
    S: np.ndarray,
    M: np.ndarray,
    MList: np.ndarray,
    GList: np.ndarray,
    g: np.ndarray,
    Ftip: np.ndarray,
    theta_Pose: np.ndarray,
    T_d_flat: np.ndarray,
    tauLim: np.ndarray,
    tauInit: np.ndarray,
    wLim: np.ndarray,
    # Simulation params
    SAMPLE_TIME: float,
    n_steps: int,
    # Trajectory params
    random_pairs: np.ndarray,
    start_time: float,
    step_time_xy: float,
    step_time_z: float,
    z_init: float,
    z_reach: float,
    # Fuzzy lookup tables
    e_vec: np.ndarray,
    de_vec: np.ndarray,
    Y_kp: np.ndarray,
    Y_kd: np.ndarray,
    # Base PID gains
    Kp_base_1: float,
    Kd_base_1: float,
    Kp_base_2: float,
    Kd_base_2: float,
    # Gain adjustment ranges
    dKp_range_1_lo: float,
    dKp_range_1_hi: float,
    dKd_range_1_lo: float,
    dKd_range_1_hi: float,
    dKp_range_2_lo: float,
    dKp_range_2_hi: float,
    dKd_range_2_lo: float,
    dKd_range_2_hi: float,
) -> float:
    """
    Main simulation loop with fuzzy PID control.
    Returns combined ISE + ITAE cost.
    """
    TRAJ_START = 2.0
    
    # Initialize states
    thetaRun_IK = theta_Pose.copy()
    thetaDotRun_IK = np.zeros(4, dtype=np.float64)
    
    thetaRun_Actual = theta_Pose.copy()
    thetaDotRun_Actual = np.zeros(4, dtype=np.float64)
    thetaDotDotRun_Actual = np.zeros(4, dtype=np.float64)
    
    # PID states for IK controllers
    IK_WzXY_I = np.zeros(3, dtype=np.float64)
    IK_WzXY_prev_error = np.zeros(3, dtype=np.float64)
    IK_Z_I = 0.0
    IK_Z_prev_error = 0.0
    
    # PID states for torque controllers
    Torque_I = tauInit.copy()
    Torque_prev_error = np.zeros(4, dtype=np.float64)
    
    # Previous theta errors for fuzzy derivative
    prev_error_theta_1 = 0.0
    prev_error_theta_2 = 0.0
    
    # Cost accumulators
    ITAE_x = 0.0
    ITAE_y = 0.0
    ISE_x = 0.0
    ISE_y = 0.0
    
    # IK PID gains
    Kp_IK_WzXY = 80.0
    Kd_IK_WzXY = 15.0
    Kp_IK_Z = 45.0
    Kd_IK_Z = 5.0
    
    # Fixed torque PID gains for theta 3 and 4
    Kp_3 = 60.0
    Kd_3 = 5.0
    Kp_4 = 40.0
    Ki_4 = 20.0
    Kd_4 = 30.0
    
    Vs_PID = np.zeros(6, dtype=np.float64)
    torqueEffort = np.zeros(4, dtype=np.float64)
    
    for i in range(n_steps):
        t = i * SAMPLE_TIME
        
        # Get trajectory position
        if t < start_time:
            pos_traj = M[:3, 3].copy()
        else:
            x_traj, y_traj, _ = tic_tac_toe_gen_numba(random_pairs, start_time, step_time_xy, t)
            z_traj = z_axis_up_down_numba(z_init, z_reach, start_time, step_time_z, t)
            pos_traj = np.array([x_traj, y_traj, z_traj], dtype=np.float64)
        
        # Build T_traj
        T_traj = np.eye(4, dtype=np.float64)
        T_traj[0, 3] = pos_traj[0]
        T_traj[1, 3] = pos_traj[1]
        T_traj[2, 3] = pos_traj[2]
        
        # Current transformation
        Tsb = kine.PoE_transform(S, M, thetaRun_Actual)
        
        # Twist error
        Vs = kine.twist_Error(Tsb, T_traj)
        
        # IK PID for WzXY
        Vs_WzXY = Vs[2:5].copy()
        d_err_WzXY = (Vs_WzXY - IK_WzXY_prev_error) / SAMPLE_TIME
        Vs_PID_WzXY = Kp_IK_WzXY * Vs_WzXY + Kd_IK_WzXY * d_err_WzXY
        IK_WzXY_prev_error = Vs_WzXY.copy()
        
        # IK PID for Z
        Vs_Z = Vs[5]
        d_err_Z = (Vs_Z - IK_Z_prev_error) / SAMPLE_TIME
        Vs_PID_Z = Kp_IK_Z * Vs_Z + Kd_IK_Z * d_err_Z
        IK_Z_prev_error = Vs_Z
        
        Vs_PID[0] = Vs[0]
        Vs_PID[1] = Vs[1]
        Vs_PID[2] = Vs_PID_WzXY[0]
        Vs_PID[3] = Vs_PID_WzXY[1]
        Vs_PID[4] = Vs_PID_WzXY[2]
        Vs_PID[5] = Vs_PID_Z
        
        # Jacobian and IK
        Js = kine.jacobian_Space(S, thetaRun_Actual)
        JsInv = helper.dls_inverse(Js, 1e-3)
        thetaDotRun_IK = JsInv @ Vs_PID
        
        # Apply velocity limits
        for j in range(4):
            if thetaDotRun_IK[j] < -wLim[j]:
                thetaDotRun_IK[j] = -wLim[j]
            elif thetaDotRun_IK[j] > wLim[j]:
                thetaDotRun_IK[j] = wLim[j]
        
        # Integrate IK
        for j in range(4):
            thetaRun_IK[j] = thetaRun_IK[j] + thetaDotRun_IK[j] * SAMPLE_TIME
        
        # Theta errors
        error_theta = thetaRun_IK - thetaRun_Actual
        
        # Error derivatives for fuzzy
        de_theta_1 = (error_theta[0] - prev_error_theta_1) / SAMPLE_TIME
        de_theta_2 = (error_theta[1] - prev_error_theta_2) / SAMPLE_TIME
        
        # Fuzzy gain lookup
        dKp_1_norm = interp2d_scalar(error_theta[0], de_theta_1, e_vec, de_vec, Y_kp)
        dKd_1_norm = interp2d_scalar(error_theta[0], de_theta_1, e_vec, de_vec, Y_kd)
        dKp_2_norm = interp2d_scalar(error_theta[1], de_theta_2, e_vec, de_vec, Y_kp)
        dKd_2_norm = interp2d_scalar(error_theta[1], de_theta_2, e_vec, de_vec, Y_kd)
        
        # Scale to gain adjustments
        dKp_1 = dKp_range_1_lo + dKp_1_norm * (dKp_range_1_hi - dKp_range_1_lo)
        dKd_1 = dKd_range_1_lo + dKd_1_norm * (dKd_range_1_hi - dKd_range_1_lo)
        dKp_2 = dKp_range_2_lo + dKp_2_norm * (dKp_range_2_hi - dKp_range_2_lo)
        dKd_2 = dKd_range_2_lo + dKd_2_norm * (dKd_range_2_hi - dKd_range_2_lo)
        
        # Current gains
        Kp_1 = Kp_base_1 + dKp_1
        Kd_1 = Kd_base_1 + dKd_1
        Kp_2 = Kp_base_2 + dKp_2
        Kd_2 = Kd_base_2 + dKd_2
        
        # Torque PID computations
        d_err_torque = (error_theta - Torque_prev_error) / SAMPLE_TIME
        
        # Theta 1
        u1 = Kp_1 * error_theta[0] + Kd_1 * d_err_torque[0]
        if u1 < -tauLim[0]:
            u1 = -tauLim[0]
        elif u1 > tauLim[0]:
            u1 = tauLim[0]
        torqueEffort[0] = u1
        
        # Theta 2
        u2 = Kp_2 * error_theta[1] + Kd_2 * d_err_torque[1]
        if u2 < -tauLim[1]:
            u2 = -tauLim[1]
        elif u2 > tauLim[1]:
            u2 = tauLim[1]
        torqueEffort[1] = u2
        
        # Theta 3
        u3 = Kp_3 * error_theta[2] + Kd_3 * d_err_torque[2]
        if u3 < -tauLim[2]:
            u3 = -tauLim[2]
        elif u3 > tauLim[2]:
            u3 = tauLim[2]
        torqueEffort[2] = u3
        
        # Theta 4 (with integral)
        Torque_I[3] = Torque_I[3] + error_theta[3] * SAMPLE_TIME
        u4 = Kp_4 * error_theta[3] + Ki_4 * Torque_I[3] + Kd_4 * d_err_torque[3]
        if u4 < -tauLim[3]:
            u4 = -tauLim[3]
        elif u4 > tauLim[3]:
            u4 = tauLim[3]
        torqueEffort[3] = u4
        
        Torque_prev_error = error_theta.copy()
        prev_error_theta_1 = error_theta[0]
        prev_error_theta_2 = error_theta[1]
        
        # Forward dynamics
        thetaDotDotRun_Actual = dyna.forward_Dynamics(
            S, MList, GList, thetaRun_Actual, thetaDotRun_Actual, torqueEffort, g, Ftip
        )
        
        # Check for invalid values
        valid = True
        for j in range(4):
            if not np.isfinite(thetaDotDotRun_Actual[j]):
                valid = False
                break
        
        if not valid:
            return 1e10
        
        # Integrate dynamics
        for j in range(4):
            thetaDotRun_Actual[j] = thetaDotRun_Actual[j] + thetaDotDotRun_Actual[j] * SAMPLE_TIME
            thetaRun_Actual[j] = thetaRun_Actual[j] + thetaDotRun_Actual[j] * SAMPLE_TIME
        
        # Position errors
        Tsb_new = kine.PoE_transform(S, M, thetaRun_Actual)
        error_x = pos_traj[0] - Tsb_new[0, 3]
        error_y = pos_traj[1] - Tsb_new[1, 3]
        
        # Accumulate cost after trajectory starts
        if t >= TRAJ_START:
            t_offset = t - TRAJ_START
            ITAE_x += t_offset * abs(error_x) * SAMPLE_TIME
            ITAE_y += t_offset * abs(error_y) * SAMPLE_TIME
            ISE_x += error_x * error_x * SAMPLE_TIME
            ISE_y += error_y * error_y * SAMPLE_TIME
    
    # Combined cost
    cost = (ISE_x + ISE_y) + 0.1 * (ITAE_x + ITAE_y)
    return cost


def simulation_cost_FuzzyPID(rule_params: np.ndarray, sim_config: dict) -> float:
    """
    Cost function for Fuzzy PID optimization using ITAE.
    Wrapper that prepares data and calls the Numba-optimized simulation loop.
    """
    # Build dKp constant matrix from parameters
    kp_const = build_const_matrix_from_params(rule_params)
    kd_const = FIXED_DKD_MATRIX
    
    # Build membership functions
    e_range = sim_config['e_range']
    de_range = sim_config['de_range']
    mfs_e = gen7tri_numba(e_range[0], e_range[1])
    mfs_de = gen7tri_numba(de_range[0], de_range[1])
    
    # Build lookup tables
    nE = 51  # Reduced for speed during optimization
    nDE = 51
    e_vec = np.linspace(e_range[0], e_range[1], nE)
    de_vec = np.linspace(de_range[0], de_range[1], nDE)
    
    Y_kp, Y_kd = build_lookup_table(mfs_e, mfs_de, kp_const, kd_const, e_vec, de_vec)
    
    # Get config values
    S = sim_config['S']
    M = sim_config['M']
    MList = sim_config['MList']
    GList = sim_config['GList']
    g = sim_config['g']
    Ftip = sim_config['Ftip']
    theta_Pose = sim_config['theta_Pose']
    tauLim = sim_config['tauLim']
    tauInit = sim_config['tauInit']
    wLim = sim_config['wLim']
    SAMPLE_TIME = sim_config['SAMPLE_TIME']
    n_steps = sim_config['n_steps']
    random_pairs = sim_config['random_pairs']
    
    # Trajectory params
    start_time = 2.0
    step_time_xy = 3.0
    step_time_z = 1.5
    z_init = M[2, 3]
    z_reach = sim_config['z_reach']
    
    # Flatten T_d for passing to numba
    T_d = sim_config['T_d']
    T_d_flat = T_d.flatten()
    
    # Call optimized simulation
    cost = simulation_loop_fuzzy(
        S, M, MList, GList, g, Ftip, theta_Pose, T_d_flat, tauLim, tauInit, wLim,
        SAMPLE_TIME, n_steps, random_pairs, start_time, step_time_xy, step_time_z, z_init, z_reach,
        e_vec, de_vec, Y_kp, Y_kd,
        sim_config['Kp_base_1'], sim_config['Kd_base_1'],
        sim_config['Kp_base_2'], sim_config['Kd_base_2'],
        sim_config['dKp_range_1'][0], sim_config['dKp_range_1'][1],
        sim_config['dKd_range_1'][0], sim_config['dKd_range_1'][1],
        sim_config['dKp_range_2'][0], sim_config['dKp_range_2'][1],
        sim_config['dKd_range_2'][0], sim_config['dKd_range_2'][1],
    )
    
    return cost


def setup_simulation_config():
    """Setup simulation configuration dictionary."""
    # Dimensions in meters
    l1 = 0.35
    l2 = 0.45
    d1 = 0.284
    d2 = 0.1
    d3 = 0.334
    
    a1 = 0.035
    b1 = 0.095
    a2 = 0.02
    b2 = 0.08
    r3 = 0.033
    c1 = 0.05
    c2 = 0.08
    
    theta_Pose = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0], dtype=np.float64)
    theta_D = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0], dtype=np.float64)
    
    n = (3, -3)
    
    q = np.array([[0, l1, l1+l2, l1+l2],
                  [0, 0, 0, 0],
                  [0, d1, d1+d2, d1+d2]], dtype=np.float64)
    
    w = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 0],
                  [1, 1, 1, 0]], dtype=np.float64)
    
    M = np.array([[1, 0, 0, l1+l2],
                  [0, 1, 0, 0],
                  [0, 0, 1, d1+d2],
                  [0, 0, 0, 1]], dtype=np.float64)
    
    n_joint = theta_Pose.size
    n_prismatic_joint, n_prismatic_axis = n
    S = lie.compute_ScrewMat(w, q, n_prismatic_joint, n_prismatic_axis)
    
    # Dynamics setup
    m1, m2, m3, m4 = 3.0, 2.0, 1.5, 1.5
    g = np.array([0, 0, -9.8], dtype=np.float64)
    tauInit = np.array([0, 0, 0, m4 * g[2]], dtype=np.float64)
    Ftip = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
    
    MList = np.zeros((4, 4, n_joint+1), dtype=np.float64)
    MList[:,:,0] = np.eye(4)
    MList[:,:,1] = np.eye(4)
    MList[0,3,1] = l1
    MList[2,3,1] = d1
    MList[:,:,2] = np.eye(4)
    MList[0,3,2] = l2
    MList[2,3,2] = d2
    MList[:,:,3] = np.eye(4)
    MList[:,:,4] = np.eye(4)
    
    I1CoM = (m1/12) * np.diag([a1**2 + b1**2, l1**2 + a1**2, b1**2 + l1**2])
    I2CoM = (m2/12) * np.diag([a2**2 + b2**2, l2**2 + a2**2, b2**2 + l2**2])
    I3CoM = m3 * np.diag([(1/12)*(d3**2 + 3*r3**2), (1/12)*(d3**2 + 3*r3**2), (1/2)*r3**2])
    I4CoM = I3CoM.copy()
    
    f1 = np.array([l1/2 - c1, 0.0, 0.0], dtype=np.float64)
    f2 = np.array([l2/2 - c2, 0.0, 0.0], dtype=np.float64)
    f3 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    f4 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    
    GList = np.zeros((6, 6, n_joint), dtype=np.float64)
    GList[:,:,0] = helper.mcI(m1, f1, I1CoM)
    GList[:,:,1] = helper.mcI(m2, f2, I2CoM)
    GList[:,:,2] = helper.mcI(m3, f3, I3CoM)
    GList[:,:,3] = helper.mcI(m4, f4, I4CoM)
    
    # Simulation parameters
    SAMPLE_TIME = 0.001
    TIME_STOP = 15.0
    n_steps = int(TIME_STOP / SAMPLE_TIME)
    
    wLim = np.array([1, 2, 2, 0.2], dtype=np.float64)
    tauLim = np.array([20, 15, 5, 20], dtype=np.float64)
    
    # Fuzzy controller parameters
    e_range = (-0.1, 0.1)
    de_range = (-1.0, 1.0)
    
    # Base PID gains
    Kp_base_1 = 100.0
    Kd_base_1 = 30.0
    Kp_base_2 = 80.0
    Kd_base_2 = 40.0
    
    # Gain adjustment ranges
    dKp_range_1 = (0.0, 200.0)
    dKd_range_1 = (0.0, 50.0)
    dKp_range_2 = (0.0, 150.0)
    dKd_range_2 = (0.0, 40.0)
    
    # Pre-compute trajectory random pairs
    random_pairs = random_index_pair_numba(1000, seed=42)
    
    # Desired transformation
    T_d = kine.PoE_transform(S, M, theta_D)
    z_reach = T_d[2, 3] - 0.1  # Example reach depth
    
    return {
        'S': S,
        'M': M,
        'MList': MList,
        'GList': GList,
        'g': g,
        'Ftip': Ftip,
        'theta_Pose': theta_Pose,
        'theta_D': theta_D,
        'T_d': T_d,
        'z_reach': z_reach,
        'tauLim': tauLim,
        'tauInit': tauInit,
        'wLim': wLim,
        'SAMPLE_TIME': SAMPLE_TIME,
        'TIME_STOP': TIME_STOP,
        'n_steps': n_steps,
        'e_range': e_range,
        'de_range': de_range,
        'Kp_base_1': Kp_base_1,
        'Kd_base_1': Kd_base_1,
        'Kp_base_2': Kp_base_2,
        'Kd_base_2': Kd_base_2,
        'dKp_range_1': dKp_range_1,
        'dKd_range_1': dKd_range_1,
        'dKp_range_2': dKp_range_2,
        'dKd_range_2': dKd_range_2,
        'random_pairs': random_pairs,
    }


def get_default_dKp_rules():
    """Get default dKp rule table for comparison."""
    return [
        ["L","L","M","M","S","ZO","ZO"],
        ["L","M","M","S","ZO","S","ZO"],
        ["M","M","S","ZO","S","M","M"],
        ["M","S","ZO","ZO","ZO","S","M"],
        ["M","S","ZO","S","M","M","L"],
        ["ZO","S","M","S","M","L","L"],
        ["ZO","ZO","M","M","L","L","L"],
    ]


def encode_rules_to_params(rule_table: list) -> np.ndarray:
    """Convert 7x7 rule table to 49 parameters."""
    label_to_idx = {"ZO": 1, "S": 2, "M": 3, "L": 4}
    params = np.zeros(49, dtype=np.float64)
    
    for i in range(7):
        for j in range(7):
            params[i * 7 + j] = label_to_idx[rule_table[i][j]]
    
    return params


def print_rule_table(rule_table: list, title: str = "Rule Table"):
    """Pretty print a rule table."""
    print(f"\n{title}:")
    print("-" * 50)
    header = "      " + "  ".join([f"de{j}" for j in range(7)])
    print(header)
    for i, row in enumerate(rule_table):
        print(f"e{i}:   " + "  ".join([f"{r:>2}" for r in row]))
    print("-" * 50)


def warmup_jit():
    """Warm up JIT compilation before timing."""
    print("Warming up JIT compilation...")
    
    # Create minimal config for warmup
    sim_config = setup_simulation_config()
    sim_config['n_steps'] = 100  # Very short simulation
    
    default_rules = get_default_dKp_rules()
    default_params = encode_rules_to_params(default_rules)
    
    # Run once to compile
    _ = simulation_cost_FuzzyPID(default_params, sim_config)
    print("JIT warmup complete.\n")


def test_single_evaluation(rule_params, sim_config, label=""):
    """Test a single cost evaluation."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC: {label}")
    print("="*60)
    
    rule_table = decode_rules_from_params(rule_params)
    print_rule_table(rule_table, "dKp Rule Table")
    
    start = time.perf_counter()
    cost = simulation_cost_FuzzyPID(rule_params, sim_config)
    elapsed = time.perf_counter() - start
    
    print(f"Cost: {cost:.6f}")
    print(f"Evaluation time: {elapsed:.3f}s")
    print("="*60)
    return cost


if __name__ == "__main__":
    # Warm up JIT
    warmup_jit()
    
    # Setup simulation config
    sim_config = setup_simulation_config()
    
    # Test with default rule table
    default_rules = get_default_dKp_rules()
    default_params = encode_rules_to_params(default_rules)
    
    print("Testing default fuzzy PID rules...")
    default_cost = test_single_evaluation(default_params, sim_config, "Default Rules")
    
    # Benchmark multiple evaluations
    print("\nBenchmarking speed (5 evaluations)...")
    times = []
    for i in range(5):
        start = time.perf_counter()
        _ = simulation_cost_FuzzyPID(default_params, sim_config)
        times.append(time.perf_counter() - start)
    print(f"Average evaluation time: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
    
    # GBO parameters
    nP = 12          # Population size
    MaxIt = 100      # Maximum iterations (reduced for demo)
    dim = 49         # 7x7 rule table
    
    # Bounds: rule indices 1-4 (ZO, S, M, L)
    lb = np.ones(dim, dtype=np.float64)
    ub = np.ones(dim, dtype=np.float64) * 4
    
    print("\n" + "="*60)
    print("Fuzzy PID Rule Optimization with GBO (Numba Optimized)")
    print("="*60)
    print(f"Population: {nP}, Iterations: {MaxIt}")
    print(f"Optimizing: 7x7 dKp Rule Table (49 parameters)")
    print(f"Fixed: dKd Rule Table")
    print(f"Bounds: Rule indices 1-4 (ZO, S, M, L)")
    print(f"Simulation time: {sim_config['n_steps'] * sim_config['SAMPLE_TIME']:.1f}s")
    print(f"Estimated time per iteration: {np.mean(times) * nP * 2:.1f}s")
    print(f"CPU cores available: {os.cpu_count()}")
    print("="*60)
    
    user_input = input("\nProceed with optimization? (y/n): ").strip().lower()
    if user_input != 'y':
        print("Optimization cancelled.")
        exit()
    
    start_time = time.perf_counter()
    
    Best_Cost, Best_Params, Convergence_curve = GBO(
        nP, MaxIt, lb, ub, dim,
        simulation_cost_FuzzyPID, sim_config,
        use_parallel=True,
        n_workers=min(nP, os.cpu_count())
    )
    
    end_time = time.perf_counter()
    
    # Decode best rules
    Best_Rules_Table = decode_rules_from_params(Best_Params)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
    print(f"Best Cost: {Best_Cost:.6f}")
    print(f"Improvement from default: {(default_cost - Best_Cost) / default_cost * 100:.2f}%")
    
    print_rule_table(Best_Rules_Table, "Optimized dKp Rule Table")
    print_rule_table(FIXED_RULE_TABLE_DKD, "Fixed dKd Rule Table")
    
    # Save results
    np.savez('fuzzy_pid_optimization_results.npz',
             Best_Cost=Best_Cost,
             Best_Params=Best_Params,
             Best_Rules_Table=np.array(Best_Rules_Table),
             Fixed_dKd_Table=np.array(FIXED_RULE_TABLE_DKD),
             Convergence_curve=Convergence_curve,
             default_cost=default_cost,
             default_params=default_params)
    
    # Generate optimized lookup tables and save
    e_range = sim_config['e_range']
    de_range = sim_config['de_range']
    
    # Create output directory if needed
    os.makedirs("FuzzyLogicOut", exist_ok=True)
    
    # Save theta 1 lookup table
    fs1 = controller.FuzzySugeno(e_range, de_range, Best_Rules_Table, FIXED_RULE_TABLE_DKD)
    fs1.save_npz(theta_N=1, fileNamePath="FuzzyLogicOut/FuzzySugeno_Optimized_Theta_1.npz", nE=101, nDE=101)
    
    # Save theta 2 lookup table
    fs2 = controller.FuzzySugeno(e_range, de_range, Best_Rules_Table, FIXED_RULE_TABLE_DKD)
    fs2.save_npz(theta_N=2, fileNamePath="FuzzyLogicOut/FuzzySugeno_Optimized_Theta_2.npz", nE=101, nDE=101)
    
    print("\nOptimized lookup tables saved to FuzzyLogicOut/")
    
    # Plot convergence
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].plot(range(1, MaxIt + 1), Convergence_curve, 'b-', linewidth=2)
        axes[0].axhline(y=default_cost, color='r', linestyle='--', label=f'Default: {default_cost:.4f}')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Best Cost')
        axes[0].set_title('GBO Convergence - Fuzzy PID Rule Optimization')
        axes[0].grid(True)
        axes[0].legend()
        
        axes[1].semilogy(range(1, MaxIt + 1), Convergence_curve, 'b-', linewidth=2)
        axes[1].axhline(y=default_cost, color='r', linestyle='--', label=f'Default: {default_cost:.4f}')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Best Cost (log scale)')
        axes[1].set_title('GBO Convergence (Log Scale)')
        axes[1].grid(True)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('fuzzy_convergence_plot.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nConvergence plot saved to 'fuzzy_convergence_plot.png'")
        
    except Exception as e:
        print(f"Could not create plot: {e}")
    
    # Generate code snippet
    print("\n" + "="*60)
    print("Copy these rule tables to your fuzzy controller:")
    print("="*60)
    print(f"""
# Optimized dKp Rule Table from GBO
ruleTable_dKp_optimized = {Best_Rules_Table}

# Fixed dKd Rule Table
ruleTable_dKd = {FIXED_RULE_TABLE_DKD}
""")