"""Fuzzy PID Rule Table Optimization using GBO (Gradient-Based Optimizer)

Optimizes: dKp rule table (7x7 = 49 parameters)
Fixed: dKd rule table (symmetric pattern)

Cost Function Components:
- ISE (Integral Square Error): Penalizes large errors heavily, good for fast response
- ITAE (Integral Time-weighted Absolute Error): Penalizes persistent errors, good for settling
- IAE (Integral Absolute Error): Balanced error penalty

Higher ISE weight  -> Faster response, but may overshoot
Higher ITAE weight -> Better settling, slower initial response  
Higher IAE weight  -> Balanced tracking performance
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
from scipy.interpolate import RegularGridInterpolator
# =============================================================================
# FIXED RULE TABLES AND CONSTANTS
# =============================================================================

LABELS_OUT = ["NL", "NM", "NS" ,"ZO", "PS", "PM", "PL"]
MAP_VAL = {"NL": -1.00, "NM": -.66, "NS": -0.33, "ZO": 0.00, "PS": 0.33, "PM": 0.66, "PL": 1.00}

# =============================================================================
# HELPER FUNCTIONS FOR RULE TABLE ENCODING/DECODING
# =============================================================================

def decode_rules_from_params(params: np.ndarray) -> tuple:
    """Convert 98 optimization parameters to two 7x7 rule table strings (dKp, dKd)."""
    params_int = np.round(params).astype(int)
    params_int = np.clip(params_int, 1, 7)
    
    # Split into 49 and 49
    kp_indices = params_int[:49].reshape(7, 7)
    kd_indices = params_int[49:].reshape(7, 7)
    
    def to_table(indices):
        return [[LABELS_OUT[indices[i, j] - 1] for j in range(7)] for i in range(7)]

    return to_table(kp_indices), to_table(kd_indices)

def encode_rules_to_params(kp_table: list, kd_table: list) -> np.ndarray:
    """Convert two 7x7 rule tables to 98 parameters."""
    label_to_idx = {
    "NL": 1, "NM": 2, "NS": 3,
    "ZO": 4,
    "PS": 5, "PM": 6, "PL": 7
    }
    params = np.zeros(98, dtype=np.float64)
    for i in range(7):
        for j in range(7):
            params[i * 7 + j] = label_to_idx[kp_table[i][j]]
            params[49 + i * 7 + j] = label_to_idx[kd_table[i][j]]
    return params
    
@njit(cache=True)
def params_to_const_matrices(params: np.ndarray) -> tuple:
    """Convert 98 parameters to two numeric 7x7 matrices (Numba compatible)."""
    val_map = np.array([-1.00, -.66, -.33, 0.00, .33, .66, 1.00], dtype=np.float64)
    kp_mat = np.zeros((7, 7), dtype=np.float64)
    kd_mat = np.zeros((7, 7), dtype=np.float64)
    
    for i in range(49):
        # dKp part
        idx_p = int(np.round(params[i]))
        idx_p = max(1, min(7, idx_p)) - 1
        kp_mat[i // 7, i % 7] = val_map[idx_p]
        
        # dKd part
        idx_d = int(np.round(params[i + 49]))
        idx_d = max(1, min(4, idx_d)) - 1
        kd_mat[i // 7, i % 7] = val_map[idx_d]
        
    return kp_mat, kd_mat

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
    g: np.ndarray, Ftip: np.ndarray, theta_Pose: np.ndarray, T_d: np.ndarray,
    tauLim: np.ndarray, tauInit: np.ndarray, wLim: np.ndarray,
    # Simulation params
    SAMPLE_TIME: float, STOP_TIME: float, startTime: float, stepTimeXY: float, stepTimeZ: float, randomPairs: np.ndarray,
    # Base PID gain
    Kp_base_1: float, Kd_base_1: float, Kp_base_2: float, Kd_base_2: float,
    Kp_IK_WzXY: float, Kd_IK_WzXY: float, Kp_IK_Z: float, Kd_IK_Z: float, 
    Kp_Torque_3: float, Kd_Torque_3: float, Kp_Torque_4: float, Ki_Torque_4: float, Kd_Torque_4: float,
    alpha: float, beta: float,
    # FPID params
    e_vec: np.ndarray, de_vec: np.ndarray, Y_kp: np.ndarray, Y_kd: np.ndarray,
    # Noise parameters
    used_Noise: bool, NOISE_Gen_thetaRun_Actual: np.ndarray, NOISE_Gen_thetaRunDot_Actual: np.ndarray
) -> tuple:
    """
    Numba-optimized simulation loop with fuzzy PID control.
    Returns: (ISE_x, ISE_y, ITAE_x, ITAE_y, IAE_x, IAE_y)
    """
    # Initialize states
    n_steps = int(STOP_TIME / SAMPLE_TIME)
    n_joint = theta_Pose.size
    thetaRun_IK = theta_Pose.copy()
    thetaRun_Actual = theta_Pose.copy()
    thetaDotRun_Actual = np.zeros_like(thetaRun_Actual)

    Tsb = kine.PoE_transform(S, M, theta_Pose)

    NOISE_thetaRun_Actual = thetaRun_Actual.copy()

    # PID controllers
    PID_IK_prevError_WzXY = np.zeros(3, dtype=np.float64)
    PID_IK_prevError_Z = 0.0

    Vs_PID = np.zeros(6, dtype=np.float64)
    torqueEffort = np.zeros_like(tauInit)
    PID_Ki_torque = tauInit[3]
    if Ki_Torque_4 > 0.0 and Kd_Torque_4 > 0.0:
        Kb_Torque_4 = 1.0 / np.sqrt(Ki_Torque_4 * Kd_Torque_4)
    else:
        Kb_Torque_4 = 1.0


    posInput_log = np.zeros((n_steps, 3), dtype=np.float64)
    posOutput_log = np.zeros((n_steps, 3), dtype=np.float64)
    torque_log = np.zeros((n_steps, n_joint), dtype=np.float64)
    tVec = np.zeros(n_steps, dtype=np.float64)
    
    for i in range(n_steps):
        t = i * SAMPLE_TIME

        if t < startTime:
            R_traj = M[:3,:3].copy()
            pos_traj = M[:3, 3].copy()
        else:
            R_traj = T_d[:3, :3]
            x_traj, y_traj = trajGen.tic_tac_toe_gen(randomPairs, startTime, stepTimeXY, t)
            z_traj = trajGen.zAxisUpDown(M[2,3], T_d[2,3], startTime, stepTimeZ, t)
            pos_traj = np.array([x_traj, y_traj, z_traj], dtype=np.float64)

        if used_Noise:
            NOISE_thetaRun_Actual = thetaRun_Actual + NOISE_Gen_thetaRun_Actual 
            NOISE_thetaDotRun_Actual = thetaDotRun_Actual + NOISE_Gen_thetaRunDot_Actual 
            NOISE_Tsb = kine.PoE_transform(S, M, NOISE_thetaRun_Actual)

        T_traj = helper.RpTo_TransMat(R_traj, pos_traj)

        Tsb = kine.PoE_transform(S, M, thetaRun_Actual)

        Vs = kine.twist_Error(NOISE_Tsb, T_traj) if used_Noise else kine.twist_Error(Tsb, T_traj)
        
        Vs_WzXY = Vs[2:5].copy()
        Vs_Z = Vs[5]

        Vs_PID_WzXY = Kp_IK_WzXY * Vs_WzXY + Kd_IK_WzXY * ((Vs_WzXY - PID_IK_prevError_WzXY) / SAMPLE_TIME)
        Vs_PID_Z =  Kp_IK_Z * Vs_Z + Kd_IK_Z * ((Vs_Z - PID_IK_prevError_Z) / SAMPLE_TIME)
        PID_IK_prevError_WzXY = Vs_WzXY.copy()
        PID_IK_prevError_Z = Vs_Z

        Vs_PID[0] = Vs[0]
        Vs_PID[1] = Vs[1]
        Vs_PID[2] = Vs_PID_WzXY[0]
        Vs_PID[3] = Vs_PID_WzXY[1]
        Vs_PID[4] = Vs_PID_WzXY[2]
        Vs_PID[5] = Vs_PID_Z
        
        # Jacobian and velocity IK
        Js = kine.jacobian_Space(S, NOISE_thetaRun_Actual) if used_Noise else kine.jacobian_Space(S, thetaRun_Actual)

        JsInv = helper.dls_inverse(Js, 1e-3)
        
        thetaDotRun_IK = JsInv @ Vs_PID
        
        # Apply velocity limits
        for j in range(4):
            thetaDotRun_IK[j] = max(-wLim[j], min(wLim[j], thetaDotRun_IK[j]))
        
        # Integrate IK
        thetaRun_IK = helper.discrete_Integrator(thetaRun_IK, thetaDotRun_IK, SAMPLE_TIME)
        
        # --- Fuzzy PID Torque Control ---
        error_theta = thetaRun_IK - NOISE_thetaRun_Actual if used_Noise else (thetaRun_IK - thetaRun_Actual)
        errorDot_theta = thetaDotRun_IK - NOISE_thetaDotRun_Actual if used_Noise else (thetaDotRun_IK - thetaDotRun_Actual)
 
        # Fuzzy lookup (bilinear interpolation)
        dKp_theta_1, dKd_theta_1 = controller.LookUp2D_Nearest(
            errorDot_theta[0],
            error_theta[0],
            e_vec,
            de_vec,
            Y_kp,
            Y_kd
        )

        dKp_theta_2, dKd_theta_2 = controller.LookUp2D_Nearest(
            errorDot_theta[1],
            error_theta[1],
            e_vec,
            de_vec,
            Y_kp,
            Y_kd
        )

        FPID_Kp_1, FPID_Kd_1 = Kp_base_1 * (1 + alpha * dKp_theta_1), Kd_base_1 * (1 + beta * dKd_theta_1)
        FPID_Kp_2, FPID_Kd_2 = Kp_base_2 * (1 + alpha * dKp_theta_2), Kd_base_2 * (1 + beta * dKd_theta_2)

        u1 = FPID_Kp_1 * error_theta[0] + FPID_Kd_1 * errorDot_theta[0]
        torqueEffort[0] = helper.clip_scalar(u1, -tauLim[0], tauLim[0])

        u2 = FPID_Kp_2 * error_theta[1] + FPID_Kd_2 * errorDot_theta[1]
        torqueEffort[1] = helper.clip_scalar(u2, -tauLim[1], tauLim[1])

        u3 = Kp_Torque_3 * error_theta[2] + Kd_Torque_3 * errorDot_theta[2] 
        torqueEffort[2] = helper.clip_scalar(u3, -tauLim[2], tauLim[2])

        PID_Ki_torque = PID_Ki_torque + error_theta[3] * SAMPLE_TIME
        u4 = Kp_Torque_4 * error_theta[3] + Ki_Torque_4 * PID_Ki_torque + Kd_Torque_4 * errorDot_theta[3]
        u4_sat = helper.clip_scalar(u4, -tauLim[3], tauLim[3])
        PID_Ki_torque = PID_Ki_torque + Kb_Torque_4 * (u4_sat - u4)
        torqueEffort[3] = u4_sat

        thetaDotDotRun_Actual = dyna.forward_Dynamics(
            S, MList, GList, thetaRun_Actual, thetaDotRun_Actual, torqueEffort, g, Ftip
        )
        
        # Integrate dynamics
        thetaDotRun_Actual = helper.discrete_Integrator(thetaDotRun_Actual, thetaDotDotRun_Actual, SAMPLE_TIME)
        thetaRun_Actual= helper.discrete_Integrator(thetaRun_Actual, thetaDotRun_Actual, SAMPLE_TIME)
        
        # --- Compute Position Errors ---
        Tsb = kine.PoE_transform(S, M, thetaRun_Actual)

        posInput_log[i, :] = pos_traj 
        posOutput_log[i, :] = Tsb[:3, 3]
        torque_log[i, :] = torqueEffort
        tVec[i] = t

    return posInput_log, posOutput_log, torque_log, tVec


# =============================================================================
# COST FUNCTION WRAPPER
# =============================================================================
def simulation_cost_FuzzyPID(rule_params: np.ndarray, sim_config: dict) -> float:
    """
    Cost function for GBO optimization.
    
    Cost = w_ISE*(ISE_x + ISE_y) + w_ITAE*(ITAE_x + ITAE_y) 
         + w_IAE*(IAE_x + IAE_y) 
    
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
    
    nE, nDE = 301, 301  # Reduced resolution for speed
    e_vec = np.linspace(e_range[0], e_range[1], nE)
    de_vec = np.linspace(de_range[0], de_range[1], nDE)
    
    Y_kp, Y_kd = controller.eval_grid(e_vec, de_vec, mfs_e, mfs_de, kp_const, kd_const)
    
    # Run simulation
    posInput_log, posOutput_log, torque_log, tVec = simulation_loop_fuzzy(
        # Robot configuration
        S=sim_config['S'], M=sim_config['M'], MList=sim_config['MList'], GList=sim_config['GList'],
        g=sim_config['g'], Ftip=sim_config['Ftip'], theta_Pose=sim_config['theta_Pose'], T_d=sim_config['T_d'], 
        tauLim=sim_config['tauLim'], tauInit=sim_config['tauInit'], wLim=sim_config['wLim'],
        # Simulation params
        SAMPLE_TIME=sim_config['SAMPLE_TIME'], STOP_TIME=sim_config['STOP_TIME'], 
        startTime=sim_config['startTime'], stepTimeXY=sim_config['stepTimeXY'], 
        stepTimeZ=sim_config['stepTimeZ'], randomPairs=sim_config['randomPairs'],
        # IK PID gain
        Kp_IK_WzXY=sim_config['Kp_IK_WzXY'], Kd_IK_WzXY=sim_config['Kd_IK_WzXY'], 
        Kp_IK_Z=sim_config['Kp_IK_Z'], Kd_IK_Z=sim_config['Kd_IK_Z'],
        # Base PID gain
        Kp_base_1=sim_config['Kp_base_1'], Kd_base_1=sim_config['Kd_base_1'],
        Kp_base_2=sim_config['Kp_base_2'], Kd_base_2=sim_config['Kd_base_2'],
        alpha=sim_config['alpha'], beta=sim_config['beta'],
        # FPID params
        Kp_Torque_3=sim_config['Kp_Torque_3'], Kd_Torque_3=sim_config['Kd_Torque_3'], 
        Kp_Torque_4=sim_config['Kp_Torque_4'], Ki_Torque_4=sim_config['Ki_Torque_4'], Kd_Torque_4=sim_config['Kd_Torque_4'],
        e_vec=e_vec, de_vec=de_vec, Y_kp=Y_kp, Y_kd=Y_kd,
        # Noise parameters
        used_Noise=sim_config['used_Noise'], NOISE_Gen_thetaRun_Actual=sim_config['NOISE_Gen_thetaRun_Actual'], 
        NOISE_Gen_thetaRunDot_Actual=sim_config['NOISE_Gen_thetaRunDot_Actual']
    )

    # Weighted cost combination
    w_ISE = sim_config['w_ISE']
    w_ITAE = sim_config['w_ITAE']
    w_IAE = sim_config['w_IAE']
    w_U = sim_config['w_U']

    SAMPLE_TIME = sim_config['SAMPLE_TIME']

    # Position error (x, y only)
    error = posInput_log[:, :2] - posOutput_log[:, :2]   # shape (N, 2)
    torque_U = np.sum(torque_log**2) * SAMPLE_TIME

    # ISE
    ISE_x = np.sum(error[:, 0] ** 2) * SAMPLE_TIME
    ISE_y = np.sum(error[:, 1] ** 2) * SAMPLE_TIME

    # IAE
    IAE_x = np.sum(np.abs(error[:, 0])) * SAMPLE_TIME
    IAE_y = np.sum(np.abs(error[:, 1])) * SAMPLE_TIME

    # ITAE (absolute time)
    ITAE_x = np.sum(tVec * np.abs(error[:, 0])) * SAMPLE_TIME
    ITAE_y = np.sum(tVec * np.abs(error[:, 1])) * SAMPLE_TIME

    cost = (w_ISE * (ISE_x + ISE_y) + 
            w_ITAE * (ITAE_x + ITAE_y) + 
            w_IAE * (IAE_x + IAE_y) + 
            w_U * torque_U
            )
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
    n_joint = theta_Pose.size

    S = lie.compute_ScrewMat(w, q, 3, -3)
    T_d = kine.PoE_transform(S, M, theta_D)

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
    
    # I1 CoM
    I1CoM = (m1/12) * np.array([
        [a1**2 + b1**2,      0,               0],
        [0,                  l1**2 + a1**2,   0],
        [0,                  0,               b1**2 + l1**2]
    ], dtype=np.float64)

    # I2 CoM
    I2CoM = (m2/12) * np.array([
        [a2**2 + b2**2,      0,               0],
        [0,                  l2**2 + a2**2,   0],
        [0,                  0,               b2**2 + l2**2]
    ], dtype=np.float64)

    # I3 CoM  (cylindrical / prismatic)
    I3CoM = m3 * np.array([
        [(1/12)*(d3**2 + 3*r3**2),    0,                              0],
        [0,                          (1/12)*(d3**2 + 3*r3**2),       0],
            [0,                           0,                              (1/2)*r3**2]
    ], dtype=np.float64)

    # I4 CoM (same as I3)
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
    STOP_TIME = 20.0
    
    # Pre-compute trajectory
    seed = 10
    startTime=2.0
    stepTimeXY=2.0
    stepTimeZ=5.0
    randomPairs = trajGen.random_Index_Pair(n=1000, seed=seed)

    NOISE = {
        "used": True,
        "theta_std": np.deg2rad(0.01),        # joint angle sensor noise [rad]
        "theta_dot_std": np.deg2rad(0.005),
    }

    NOISE_Gen_thetaRun_Actual = np.random.normal(0.0, NOISE["theta_std"], size = n_joint)
    NOISE_Gen_thetaRunDot_Actual  = np.random.normal(0.0, NOISE["theta_dot_std"], size = n_joint) 
    return {
        'S': S, 'M': M, 'MList': MList, 'GList': GList,
        'g': g, 'Ftip': Ftip, 'theta_Pose':theta_Pose, 'T_d': T_d, 
        'tauLim': np.array([20.0, 15.0, 5.0, 20.0]),
        'tauInit': tauInit,
        'wLim': np.array([1.0, 2.0, 2.0, 0.2]),
        'SAMPLE_TIME': SAMPLE_TIME,
        'STOP_TIME': STOP_TIME,
        'startTime':startTime,
        'stepTimeXY':stepTimeXY,
        'stepTimeZ':stepTimeZ,
        'randomPairs': randomPairs,

        # PID gains for IK
        'Kp_IK_WzXY': 60.0, 'Kd_IK_WzXY': 15.0, 
        'Kp_IK_Z': 70.0, 'Kd_IK_Z': 5.0, 

        # PID gains for torque control
        'Kp_base_1': 80.0, 'Kd_base_1': 60.0,
        'Kp_base_2': 90.0, 'Kd_base_2': 43.74,

        'Kp_Torque_3': 90.0, 'Kd_Torque_3': 5.0, 
        'Kp_Torque_4': 40.0, 'Ki_Torque_4': 20.0, 'Kd_Torque_4': 30.0,

        'alpha': 0.55, 'beta': 0.3,

        # Fuzzy parameters
        'e_range': (-0.1, 0.1),
        'de_range': (-1.0, 1.0),

        # Cost weights
        'w_ISE': 0.3, 'w_ITAE': 0.7, 'w_IAE': 0.8, 'w_U': 1e-4,
        'used_Noise': NOISE["used"], 'NOISE_Gen_thetaRun_Actual': NOISE_Gen_thetaRun_Actual, 'NOISE_Gen_thetaRunDot_Actual':NOISE_Gen_thetaRunDot_Actual
    }

def get_default_rules():
    """Default rule tables for comparison."""
    dkp = [
        ["PL", "PL", "PM", "PM", "PS", "ZO", "ZO"],
        ["PL", "PM", "PM", "PS", "ZO", "NS", "ZO"],
        ["PM", "PM", "PS", "ZO", "NS", "NS", "NM"],
        ["PM", "PS", "ZO", "NL", "ZO", "PS", "PM"],
        ["NM", "NS", "NS", "ZO", "PS", "PM", "PM"],
        ["ZO", "NS", "ZO", "PS", "PM", "PM", "PL"],
        ["ZO", "ZO", "PS", "PM", "PM", "PL", "PL"],
    ]
    
    dkd = [
        ["PM", "PM", "PM", "PS", "PM", "PM", "PM"],
        ["PM", "PS", "PS", "ZO", "PS", "PS", "PM"],
        ["PS", "ZO", "NS", "NL", "NS", "ZO", "PS"],
        ["ZO", "NL", "NL", "NL", "NL", "NL", "ZO"],
        ["PS", "ZO", "NS", "NL", "NS", "ZO", "PS"],
        ["PM", "PS", "PS", "ZO", "PS", "PS", "PM"],
        ["PM", "PM", "PM", "PS", "PM", "PM", "PM"],
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
    nP = 16
    MaxIt = 250
    dim = 98
    lb = np.ones(dim, dtype=np.float64)
    ub = np.ones(dim, dtype=np.float64) * 7
    
    print("\n" + "=" * 70)
    print("GBO OPTIMIZATION SETTINGS")
    print("=" * 70)
    print(f"Population size: {nP}")
    print(f"Max iterations: {MaxIt}")
    print(f"Dimensions: {dim} (7x7 rule table)")
    print(f"Bounds: [1, 7] {LABELS_OUT}")
    print(f"Est. time per evaluation: {eval_time:.2f}s")
    print(f"Est. time per iteration: {eval_time * nP * 2:.1f}s")
    print(f"Est. total time: {eval_time * nP * 2 * MaxIt / 60:.1f} minutes")
    print(f"Cost weights: ISE={sim_config['w_ISE']}, ITAE={sim_config['w_ITAE']}, "
          f"IAE={sim_config['w_IAE']}")
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