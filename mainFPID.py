"""
Fuzzy PID Rule Table Optimization using GBO (Gradient-Based Optimizer)

Optimizes: dKp rule table (7x7 = 49 parameters)
Fixed: dKd rule table (symmetric pattern)

Cost Function Components:
- ISE (Integral Square Error): Penalizes large errors heavily, good for fast response
- ITAE (Integral Time-weighted Absolute Error): Penalizes persistent errors, good for settling
- IAE (Integral Absolute Error): Balanced error penalty
- ISCO (Integral Square Control Output): Penalizes aggressive control, smoother response

Higher ISE weight  -> Faster response, but may overshoot
Higher ITAE weight -> Better settling, slower initial response  
Higher IAE weight  -> Balanced tracking performance
Higher ISCO weight -> Smoother control, less aggressive, better noise rejection
"""

import numpy as np
import time
import os
import sys
from numba import njit

# Import existing modules
import PoE.lieTheory as lie
import PoE.kinematics as kine
import PoE.helper as helper
import PoE.dynamics as dyna
import PoE.trajectoryGen as trajGen
import PoE.controller as controller
from PoE.optimizer import GBO

# =============================================================================
# FIXED RULE TABLES AND CONSTANTS
# =============================================================================

# Output labels: index 1->ZO, 2->S, 3->M, 4->L
LABELS_OUT = ["ZO", "S", "M", "L"]
MAP_VAL = {"ZO": 0.00, "S": 0.33, "M": 0.66, "L": 1.00}

# =============================================================================
# HELPER FUNCTIONS FOR RULE TABLE ENCODING/DECODING
# =============================================================================

def decode_rules_from_params(params: np.ndarray) -> tuple:
    """Convert 98 optimization parameters to two 7x7 rule tables (dKp, dKd)."""
    params_int = np.round(params).astype(int)
    params_int = np.clip(params_int, 1, 4)
    
    # First 49 params -> dKp
    dkp_indices = params_int[:49].reshape(7, 7)
    dkp_table = []
    for i in range(7):
        row = [LABELS_OUT[dkp_indices[i, j] - 1] for j in range(7)]
        dkp_table.append(row)
    
    # Next 49 params -> dKd
    dkd_indices = params_int[49:98].reshape(7, 7)
    dkd_table = []
    for i in range(7):
        row = [LABELS_OUT[dkd_indices[i, j] - 1] for j in range(7)]
        dkd_table.append(row)
    
    return dkp_table, dkd_table

def encode_rules_to_params(dkp_table: list, dkd_table: list) -> np.ndarray:
    """Convert two 7x7 rule tables to 98 parameters."""
    label_to_idx = {"ZO": 1, "S": 2, "M": 3, "L": 4}
    params = np.zeros(98, dtype=np.float64)
    
    # Encode dKp (0-48)
    for i in range(7):
        for j in range(7):
            params[i * 7 + j] = label_to_idx[dkp_table[i][j]]
    
    # Encode dKd (49-97)
    for i in range(7):
        for j in range(7):
            params[49 + i * 7 + j] = label_to_idx[dkd_table[i][j]]
    
    return params

@njit(cache=True)
def params_to_const_matrices(params: np.ndarray) -> tuple:
    """Convert 98 parameters to two numeric 7x7 constant matrices (dKp, dKd)."""
    val_map = np.array([0.00, 0.33, 0.66, 1.00], dtype=np.float64)
    
    kp_out = np.zeros((7, 7), dtype=np.float64)
    kd_out = np.zeros((7, 7), dtype=np.float64)
    
    # dKp (first 49)
    for i in range(49):
        idx = int(np.round(params[i]))
        idx = max(1, min(4, idx)) - 1
        kp_out[i // 7, i % 7] = val_map[idx]
    
    # dKd (next 49)
    for i in range(49):
        idx = int(np.round(params[49 + i]))
        idx = max(1, min(4, idx)) - 1
        kd_out[i // 7, i % 7] = val_map[idx]
    
    return kp_out, kd_out

def print_rule_table(rule_table: list, title: str = "Rule Table"):
    """Pretty print a 7x7 rule table."""
    labels_in = ["NL", "NM", "NS", "ZO", "PS", "PM", "PL"]
    print(f"\n{title}:")
    print("-" * 60)
    header = "e\\de  " + "  ".join([f"{l:>3}" for l in labels_in])
    print(header)
    print("-" * 60)
    for i, row in enumerate(rule_table):
        print(f"{labels_in[i]:>4}  " + "  ".join([f"{r:>3}" for r in row]))
    print("-" * 60)


# =============================================================================
# NUMBA-OPTIMIZED SIMULATION LOOP
# =============================================================================

@njit(cache=True)
def simulation_loop_fuzzy(
    # Robot configuration
    S: np.ndarray, M: np.ndarray, MList: np.ndarray, GList: np.ndarray,
    g: np.ndarray, Ftip: np.ndarray, theta_Pose: np.ndarray, theta_D: np.ndarray,
    tauLim: np.ndarray, tauInit: np.ndarray, wLim: np.ndarray,
    # Simulation params
    SAMPLE_TIME: float, n_steps: int,
    # Trajectory params
    random_pairs: np.ndarray, seed:int, start_time: float, step_time_xy: float,
    step_time_z: float, z_init: float, z_reach: float,
    # Fuzzy lookup tables (pre-computed)
    e_vec: np.ndarray, de_vec: np.ndarray, Y_kp: np.ndarray, Y_kd: np.ndarray,
    # Base PID gains
    Kp_base_1: float, Kd_base_1: float, Kp_base_2: float, Kd_base_2: float,
    # Noise parameters
    noise_used: bool, theta_std: float, theta_dot_std: float, torque_std: float,
) -> tuple:
    """
    Numba-optimized simulation loop with fuzzy PID control.
    Returns: (ISE_x, ISE_y, ITAE_x, ITAE_y, IAE_x, IAE_y, ISCO)
    """
    TRAJ_START = 2.0
    if noise_used:
        np.random.seed(seed)
    
    # Initialize states
    n_joint = theta_Pose.size
    thetaRun_IK = theta_Pose.copy()
    thetaRun_Actual = theta_Pose.copy()
    thetaDotRun_Actual = np.zeros(4, dtype=np.float64)

    T_d = kine.PoE_transform(S, M, theta_D)
    Tsb = kine.PoE_transform(S, M, theta_Pose)

    NOISE_thetaRun_Actual = thetaRun_Actual.copy()
    # PID states
    IK_WzXY_prev_error = np.zeros(3, dtype=np.float64)
    IK_Z_prev_error = 0.0
    Torque_I = tauInit.copy()
    
    # Cost accumulators
    ISE_x, ISE_y = 0.0, 0.0
    ITAE_x, ITAE_y = 0.0, 0.0
    IAE_x, IAE_y = 0.0, 0.0
    ISCO = 0.0  # Integral Square Control Output
    
    # Fixed PID gains
    Kp_IK_WzXY, Kd_IK_WzXY = 80.0, 15.0
    Kp_IK_Z, Kd_IK_Z = 45.0, 5.0
    Kp_3, Kd_3 = 60.0, 5.0
    Kp_4, Ki_4, Kd_4 = 40.0, 20.0, 30.0
    
    Vs_PID = np.zeros(6, dtype=np.float64)
    torqueEffort = np.zeros(4, dtype=np.float64)
    
    for i in range(n_steps):
        t = i * SAMPLE_TIME
        
        # --- Trajectory Generation ---
        if t < start_time:
            pos_traj = M[:3, 3].copy()
        else:
            x_traj, y_traj = trajGen.tic_tac_toe_gen(random_pairs, start_time, step_time_xy, t)
            z_traj = trajGen.zAxisUpDown(z_init, z_reach, start_time, step_time_z, t)
            pos_traj = np.array([x_traj, y_traj, z_traj], dtype=np.float64)

        if noise_used:
            NOISE_thetaRun_Actual = thetaRun_Actual + np.random.normal(0.0, theta_std, size = n_joint)
            NOISE_thetaDotRun_Actual = thetaDotRun_Actual + np.random.normal(0.0, theta_dot_std, size = n_joint) 
            NOISE_Tsb = kine.PoE_transform(S, M, NOISE_thetaRun_Actual)
    
        T_traj = helper.RpTo_TransMat(T_d[:3, :3], pos_traj)
        Tsb = kine.PoE_transform(S, M, thetaRun_Actual)
        
        Vs = kine.twist_Error(NOISE_Tsb, T_traj) if noise_used else kine.twist_Error(Tsb, T_traj)

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
        
        Vs_PID[0], Vs_PID[1] = Vs[0], Vs[1]
        Vs_PID[2], Vs_PID[3], Vs_PID[4] = Vs_PID_WzXY[0], Vs_PID_WzXY[1], Vs_PID_WzXY[2]
        Vs_PID[5] = Vs_PID_Z
        
        # Jacobian and velocity IK
        Js = kine.jacobian_Space(S, NOISE_thetaRun_Actual) if  noise_used else kine.jacobian_Space(S, thetaRun_Actual)
        if not np.all(np.isfinite(Js)):
            return 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10
        JsInv = helper.dls_inverse(Js, 1e-3)
        thetaDotRun_IK = JsInv @ Vs_PID
        
        # Apply velocity limits
        for j in range(4):
            thetaDotRun_IK[j] = max(-wLim[j], min(wLim[j], thetaDotRun_IK[j]))
        
        # Integrate IK
        thetaRun_IK = helper.discrete_Integrator(thetaRun_IK, thetaDotRun_IK, SAMPLE_TIME)
        
        # --- Fuzzy PID Torque Control ---
        error_theta = thetaRun_IK - NOISE_thetaRun_Actual if noise_used else (thetaRun_IK - thetaRun_Actual)
        errorDot_theta = thetaDotRun_IK - NOISE_thetaDotRun_Actual if noise_used else (thetaDotRun_IK - thetaDotRun_Actual)
 
        # Fuzzy lookup (bilinear interpolation)
        dKp_1_norm = interp2d_scalar(error_theta[0], errorDot_theta[0] , e_vec, de_vec, Y_kp)
        dKd_1_norm = interp2d_scalar(error_theta[0], errorDot_theta[0], e_vec, de_vec, Y_kd)
        dKp_2_norm = interp2d_scalar(error_theta[1], errorDot_theta[1], e_vec, de_vec, Y_kp)
        dKd_2_norm = interp2d_scalar(error_theta[1], errorDot_theta[1], e_vec, de_vec, Y_kd)
        
        # Compute actual gains
        Kp_1 = Kp_base_1 * dKp_1_norm 
        Kd_1 = Kd_base_1 * dKd_1_norm
        Kp_2 = Kp_base_2 * dKp_2_norm 
        Kd_2 = Kd_base_2 * dKd_2_norm  

        # Theta 1 (Fuzzy)
        u1 = Kp_1 * error_theta[0] + Kd_1 * errorDot_theta[0]
        torqueEffort[0] = max(-tauLim[0], min(tauLim[0], u1))
        
        # Theta 2 (Fuzzy)
        u2 = Kp_2 * error_theta[1] + Kd_2 * errorDot_theta[1]
        torqueEffort[1] = max(-tauLim[1], min(tauLim[1], u2))
        
        # Theta 3 (Fixed PD)
        u3 = Kp_3 * error_theta[2] + Kd_3 * errorDot_theta[2]
        torqueEffort[2] = max(-tauLim[2], min(tauLim[2], u3))
        
        # Theta 4 (Fixed PID)
        Torque_I[3] = Torque_I[3] + error_theta[3] * SAMPLE_TIME
        u4 = Kp_4 * error_theta[3] + Ki_4 * Torque_I[3] + Kd_4 * errorDot_theta[3]
        torqueEffort[3] = max(-tauLim[3], min(tauLim[3], u4))

        torqueNoise = np.random.normal(0.0, torque_std, size = n_joint)
        torqueEffort = np.clip(torqueEffort + torqueNoise, -tauLim, tauLim) if noise_used else torqueEffort 
 
        # --- Forward Dynamics ---
        thetaDotDotRun = dyna.forward_Dynamics(
            S, MList, GList, thetaRun_Actual, thetaDotRun_Actual, torqueEffort, g, Ftip
        )
        
        # Check for invalid dynamics
        valid = True
        for j in range(4):
            if not np.isfinite(thetaDotDotRun[j]):
                valid = False
                break
        if not valid:
            return 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10
        
        # Integrate dynamics
        thetaDotRun_Actual = thetaDotRun_Actual + thetaDotDotRun * SAMPLE_TIME
        thetaRun_Actual = thetaRun_Actual + thetaDotRun_Actual * SAMPLE_TIME
        
        # --- Compute Position Errors ---
        Tsb_new = kine.PoE_transform(S, M, thetaRun_Actual)
        error_x = pos_traj[0] - Tsb_new[0, 3]
        error_y = pos_traj[1] - Tsb_new[1, 3]
        
        # Accumulate costs after trajectory starts
        if t >= TRAJ_START:
            t_offset = t - TRAJ_START
            
            # ISE: Good for fast response, penalizes large errors
            ISE_x += error_x * error_x * SAMPLE_TIME
            ISE_y += error_y * error_y * SAMPLE_TIME
            
            # ITAE: Good for settling time, penalizes persistent errors
            ITAE_x += t_offset * abs(error_x) * SAMPLE_TIME
            ITAE_y += t_offset * abs(error_y) * SAMPLE_TIME
            
            # IAE: Balanced error penalty
            IAE_x += abs(error_x) * SAMPLE_TIME
            IAE_y += abs(error_y) * SAMPLE_TIME
            
            # ISCO: Control effort penalty (smoother response, noise rejection)
            ISCO += (torqueEffort[0]**2 + torqueEffort[1]**2) * SAMPLE_TIME
    
    return ISE_x, ISE_y, ITAE_x, ITAE_y, IAE_x, IAE_y, ISCO


@njit(cache=True)
def interp2d_scalar(x: float, y: float, x_vec: np.ndarray, y_vec: np.ndarray, Z: np.ndarray) -> float:
    """Bilinear interpolation for scalar (x, y) input with clipping."""
    nx, ny = x_vec.shape[0], y_vec.shape[0]
    
    # Clip to bounds
    x_c = max(x_vec[0], min(x, x_vec[nx - 1]))
    y_c = max(y_vec[0], min(y, y_vec[ny - 1]))
    
    # Find indices
    ix = 0
    for i in range(nx - 1):
        if x_vec[i + 1] >= x_c:
            ix = i
            break
    else:
        ix = nx - 2
    
    iy = 0
    for i in range(ny - 1):
        if y_vec[i + 1] >= y_c:
            iy = i
            break
    else:
        iy = ny - 2
    
    # Interpolation weights
    dx = x_vec[ix + 1] - x_vec[ix]
    dy = y_vec[iy + 1] - y_vec[iy]
    tx = (x_c - x_vec[ix]) / dx if dx > 1e-12 else 0.0
    ty = (y_c - y_vec[iy]) / dy if dy > 1e-12 else 0.0
    
    # Bilinear interpolation
    z00, z01 = Z[iy, ix], Z[iy, ix + 1]
    z10, z11 = Z[iy + 1, ix], Z[iy + 1, ix + 1]
    z0 = z00 * (1.0 - tx) + z01 * tx
    z1 = z10 * (1.0 - tx) + z11 * tx
    
    return z0 * (1.0 - ty) + z1 * ty


# =============================================================================
# COST FUNCTION WRAPPER
# =============================================================================

def simulation_cost_FuzzyPID(rule_params: np.ndarray, sim_config: dict) -> float:
    """
    Cost function for GBO optimization.
    
    Cost = w_ISE*(ISE_x + ISE_y) + w_ITAE*(ITAE_x + ITAE_y) 
         + w_IAE*(IAE_x + IAE_y) + w_ISCO*ISCO
    
    Weights affect behavior:
    - Higher w_ISE:  Faster response, may overshoot
    - Higher w_ITAE: Better settling, slower initial response
    - Higher w_IAE:  Balanced tracking
    - Higher w_ISCO: Smoother control, better noise rejection
    """
    kp_const, kd_const = params_to_const_matrices(rule_params)
    
    # Get config
    e_range = sim_config['e_range']
    de_range = sim_config['de_range']
    
    # Build lookup tables using existing controller functions
    mfs_e = controller.gen7tri(e_range[0], e_range[1])
    mfs_de = controller.gen7tri(de_range[0], de_range[1])
    
    nE, nDE = 201, 201  # Reduced resolution for speed
    e_vec = np.linspace(e_range[0], e_range[1], nE)
    de_vec = np.linspace(de_range[0], de_range[1], nDE)
    
    Y_kp, Y_kd = controller.eval_grid(e_vec, de_vec, mfs_e, mfs_de, kp_const, kd_const)
    
    # Run simulation
    ISE_x, ISE_y, ITAE_x, ITAE_y, IAE_x, IAE_y, ISCO = simulation_loop_fuzzy(
        sim_config['S'], sim_config['M'], sim_config['MList'], sim_config['GList'],
        sim_config['g'], sim_config['Ftip'], sim_config['theta_Pose'], sim_config['theta_D'], 
        sim_config['tauLim'], sim_config['tauInit'], sim_config['wLim'],
        sim_config['SAMPLE_TIME'], sim_config['n_steps'],
        sim_config['random_pairs'], sim_config['seed'], 2.0, 3.0, 1.5,
        sim_config['z_init'], sim_config['z_reach'],
        e_vec, de_vec, Y_kp, Y_kd,
        sim_config['Kp_base_1'], sim_config['Kd_base_1'],
        sim_config['Kp_base_2'], sim_config['Kd_base_2'],
        sim_config['NOISE']['used'], sim_config['NOISE']['theta_std'],
        sim_config['NOISE']['theta_dot_std'], sim_config['NOISE']['torque_std'],
    )
    
    # Weighted cost combination
    w_ISE = sim_config.get('w_ISE', 1.0)
    w_ITAE = sim_config.get('w_ITAE', 0.1)
    w_IAE = sim_config.get('w_IAE', 0.0)
    w_ISCO = sim_config.get('w_ISCO', 0.001)  # Small weight for control smoothness
    
    cost = (w_ISE * (ISE_x + ISE_y) + 
            w_ITAE * (ITAE_x + ITAE_y) + 
            w_IAE * (IAE_x + IAE_y) + 
            w_ISCO * ISCO)
    
    return cost


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

def setup_simulation_config():
    """Setup robot and simulation configuration."""
    # Robot dimensions (meters)
    l1, l2 = 0.35, 0.45
    d1, d2, d3 = 0.284, 0.1, 0.334
    a1, b1, a2, b2 = 0.035, 0.095, 0.02, 0.08
    r3, c1, c2 = 0.033, 0.05, 0.08

    theta_Pose = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    theta_D= np.array([np.deg2rad(0),
                    np.deg2rad(0),
                    np.deg2rad(0),
                                0.4], dtype=np.float64)
   
    # Screw axis setup
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
    
    S = lie.compute_ScrewMat(w, q, 3, -3)
    
        
    # Dynamics
    m1, m2, m3, m4 = 3.0, 2.0, 1.5, 1.5
    g = np.array([0, 0, -9.8], dtype=np.float64)
    tauInit = np.array([0, 0, 0, m4 * g[2]], dtype=np.float64)
    Ftip = np.zeros(6, dtype=np.float64)
    
    MList = np.zeros((4, 4, 5), dtype=np.float64)
    for i in range(5):
        MList[:, :, i] = np.eye(4)
    MList[0, 3, 1], MList[2, 3, 1] = l1, d1
    MList[0, 3, 2], MList[2, 3, 2] = l2, d2
    
    I1 = (m1/12) * np.diag([a1**2 + b1**2, l1**2 + a1**2, b1**2 + l1**2])
    I2 = (m2/12) * np.diag([a2**2 + b2**2, l2**2 + a2**2, b2**2 + l2**2])
    I3 = m3 * np.diag([(1/12)*(d3**2 + 3*r3**2), (1/12)*(d3**2 + 3*r3**2), (1/2)*r3**2])
    
    GList = np.zeros((6, 6, 4), dtype=np.float64)
    GList[:, :, 0] = helper.mcI(m1, np.array([l1/2-c1, 0, 0]), I1)
    GList[:, :, 1] = helper.mcI(m2, np.array([l2/2-c2, 0, 0]), I2)
    GList[:, :, 2] = helper.mcI(m3, np.zeros(3), I3)
    GList[:, :, 3] = helper.mcI(m4, np.zeros(3), I3.copy())
    
    # Simulation parameters
    SAMPLE_TIME = 0.001
    TIME_STOP = 20.0
    n_steps = int(TIME_STOP / SAMPLE_TIME)
    
    # Pre-compute trajectory
    seed = 3
    random_pairs = trajGen.random_Index_Pair(n=1000, seed=seed)

    NOISE = {
        "used": False,
        "theta_std": np.deg2rad(0.005),        # joint angle sensor noise [rad]
        "theta_dot_std": np.deg2rad(0.005),
        "torque_std": 0.2
    }
    
    return {
        'S': S, 'M': M, 'MList': MList, 'GList': GList,
        'g': g, 'Ftip': Ftip, 'theta_Pose': theta_Pose, 'theta_D': theta_D,
        'tauLim': np.array([20.0, 15.0, 5.0, 20.0]),
        'tauInit': tauInit,
        'wLim': np.array([1.0, 2.0, 2.0, 0.2]),
        'SAMPLE_TIME': SAMPLE_TIME,
        'n_steps': n_steps,
        'random_pairs': random_pairs,
        'seed': seed,
        'z_init': d1 + d2,
        'z_reach': d1 + d2 - 0.1,
        # Fuzzy parameters
        'e_range': (-0.1, 0.1),
        'de_range': (-1.0, 1.0),
        # Base PID gains
        'Kp_base_1': 489.76, 'Kd_base_1': 293.38,
        'Kp_base_2': 880.28, 'Kd_base_2': 43.74,
        # Cost weights
        'w_ISE': 0.3, 'w_ITAE': 0.7, 'w_IAE': 0.8, 'w_ISCO': 0.01,
        'NOISE': NOISE
    }


def get_default_rules():
    """Default rule tables for comparison."""
    dkp = [
        ["L","L","M","M","S","ZO","ZO"],
        ["L","M","M","S","ZO","S","ZO"],
        ["M","M","S","ZO","S","M","M"],
        ["M","S","ZO","ZO","ZO","S","M"],
        ["M","S","ZO","S","M","M","L"],
        ["ZO","S","M","S","M","L","L"],
        ["ZO","ZO","M","M","L","L","L"],
    ]
    
    dkd = [
        ["L","M","M","S","M","M","L"],
        ["M","S","S","ZO","S","S","M"],
        ["M","S","ZO","ZO","ZO","S","M"],
        ["S","ZO","ZO","ZO","ZO","ZO","S"],
        ["M","S","ZO","ZO","ZO","S","M"],
        ["M","S","S","ZO","S","S","M"],
        ["L","M","M","S","M","M","L"],
    ]
    
    return dkp, dkd


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def warmup_jit(sim_config):
    """Warm up JIT compilation."""
    print("=" * 70)
    print("WARMING UP JIT COMPILATION...")
    print("=" * 70)
    
    warmup_config = sim_config.copy()
    warmup_config['n_steps'] = 100
    
    default_dkp, default_dkd = get_default_rules()
    default_params = encode_rules_to_params(default_dkp, default_dkd)
    
    start = time.perf_counter()
    _ = simulation_cost_FuzzyPID(default_params, warmup_config)
    elapsed = time.perf_counter() - start
    
    print(f"JIT warmup complete in {elapsed:.2f}s")
    print("=" * 70 + "\n")

def test_evaluation(rule_params, sim_config, label="Test"):
    """Test single cost evaluation with detailed output."""
    print(f"\n{'='*70}")
    print(f"EVALUATION: {label}")
    print("=" * 70)
    
    dkp_table, dkd_table = decode_rules_from_params(rule_params)
    print_rule_table(dkp_table, "dKp Rule Table")
    print_rule_table(dkd_table, "dKd Rule Table")
    
    start = time.perf_counter()
    cost = simulation_cost_FuzzyPID(rule_params, sim_config)
    elapsed = time.perf_counter() - start
    
    print(f"\nCost: {cost:.6f}")
    print(f"Evaluation time: {elapsed:.3f}s")
    print("=" * 70)
    
    return cost, elapsed


if __name__ == "__main__":
    os.makedirs("FuzzyLogicOut", exist_ok=True)
    log_file = "FuzzyLogicOut/optimization_log.txt"
    sys.stdout = helper.DualLogger(log_file)
    sys.stderr = sys.stdout

    print("\n" + "=" * 70)
    print("FUZZY PID RULE TABLE OPTIMIZATION USING GBO")
    print("=" * 70)
    print("Optimizing: dKp and dKd rules tables (with each rules table is a 7x7 = 49 parameters)")
    print("=" * 70 + "\n")
    
    # Setup
    sim_config = setup_simulation_config()
    warmup_jit(sim_config)
    
    # Test default
    default_dkp, default_dkd = get_default_rules()
    default_params = encode_rules_to_params(default_dkp, default_dkd)
    default_cost, eval_time = test_evaluation(default_params, sim_config, "Default Rules")
    
    # GBO parameters
    nP = 20
    MaxIt = 25000
    dim = 98
    lb = np.ones(dim, dtype=np.float64)
    ub = np.ones(dim, dtype=np.float64) * 4
    
    print("\n" + "=" * 70)
    print("GBO OPTIMIZATION SETTINGS")
    print("=" * 70)
    print(f"Population size: {nP}")
    print(f"Max iterations: {MaxIt}")
    print(f"Dimensions: {dim} (7x7 rule table)")
    print(f"Bounds: [1, 4] (ZO=1, S=2, M=3, L=4)")
    print(f"Simulation time: {sim_config['n_steps'] * sim_config['SAMPLE_TIME']:.1f}s")
    print(f"Est. time per evaluation: {eval_time:.2f}s")
    print(f"Est. time per iteration: {eval_time * nP * 2:.1f}s")
    print(f"Est. total time: {eval_time * nP * 2 * MaxIt / 60:.1f} minutes")
    print(f"Cost weights: ISE={sim_config['w_ISE']}, ITAE={sim_config['w_ITAE']}, "
          f"IAE={sim_config['w_IAE']}, ISCO={sim_config['w_ISCO']}")
    print("=" * 70)
    
    # Run GBO
    print("\n" + "=" * 70)
    print("STARTING GBO OPTIMIZATION...")
    print("=" * 70 + "\n")
    
    start_time = time.perf_counter()
    
    Best_Cost, Best_Params, Convergence_curve = GBO(
        nP, MaxIt, lb, ub, dim,
        simulation_cost_FuzzyPID, sim_config,
        use_parallel=True,
        n_workers=min(nP, os.cpu_count())
    )
    
    total_time = time.perf_counter() - start_time
    
    # Results
    Best_dKp, Best_dKd = decode_rules_from_params(Best_Params)

    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Best cost: {Best_Cost:.6f}")
    print(f"Improvement: {(default_cost - Best_Cost) / default_cost * 100:.2f}%")
    
    print_rule_table(Best_dKp, "OPTIMIZED dKp Rule Table")
    print_rule_table(Best_dKd, "OPTIMIZED dKd Rule Table")
    
    # Save results
    os.makedirs("FuzzyLogicOut", exist_ok=True)
    
    np.savez('FuzzyLogicOut/fuzzy_optimization_results.npz',
             Best_Cost=Best_Cost,
             Best_Params=Best_Params,
             Best_dKp=np.array(Best_dKp),
             Best_dKd=np.array(Best_dKd),
             Convergence=Convergence_curve,
             default_cost=default_cost)
    
    # Save lookup tables
    e_range, de_range = sim_config['e_range'], sim_config['de_range']
    fs = controller.FuzzySugeno(e_range, de_range, Best_dKp, Best_dKd)
    fs.save_npz(theta_N=1, fileNamePath="FuzzyLogicOut/FuzzySugeno_Optimized_Theta_1.npz", nE=201, nDE=201)
    fs.save_npz(theta_N=2, fileNamePath="FuzzyLogicOut/FuzzySugeno_Optimized_Theta_2.npz", nE=201, nDE=201)
    
    print("\nResults saved to FuzzyLogicOut/")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(Convergence_curve, 'b-', lw=2)
        ax[0].axhline(default_cost, color='r', ls='--', label=f'Default: {default_cost:.4f}')
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Best Cost')
        ax[0].set_title('GBO Convergence')
        ax[0].legend()
        ax[0].grid(True)
        
        ax[1].semilogy(Convergence_curve, 'b-', lw=2)
        ax[1].axhline(default_cost, color='r', ls='--')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Best Cost (log)')
        ax[1].set_title('Convergence (Log Scale)')
        ax[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('FuzzyLogicOut/convergence.png', dpi=150)
        plt.show()
    except Exception as e:
        print(f"Plot error: {e}")
    
    # Output code
    print("\n" + "=" * 70)
    print("COPY THIS TO YOUR CODE:")
    print("=" * 70)
    print(f"ruleTable_dKp_optimized = {Best_dKp}")
    print(f"\nruleTable_dKd_optimized = {Best_dKd}")

    # Restore stdout before closing
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    try:
        sys.stdout.write("✓ Log saved to FuzzyLogicOut/optimization_log.txt\n")
        sys.stdout.close()
    except:
        pass