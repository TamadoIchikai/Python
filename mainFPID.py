import numpy as np
import time
import os
import sys
from numba import njit

# Import existing modules
import PoE.kinematics as kine
import PoE.helper as helper
import PoE.dynamics as dyna
import PoE.trajectoryGen as trajGen
import PoE.controller as controller
import PoE.optimizer as optimizer
from main import setup_simulation_config

# =============================================================================
# NUMBA-OPTIMIZED SIMULATION LOOP
# =============================================================================
def simulation_loop_fuzzy(sim_config, fuzzyParams=None, fuzzyParamsPath=None, used_InOptimizer=False):
    if used_InOptimizer and fuzzyParams is not None:
        e_vec=fuzzyParams['e_vec'], de_vec=fuzzyParams['de_vec'], Y_kp=fuzzyParams['Y_kp'], Y_kd=fuzzyParams['Y_kd']
    elif used_InOptimizer and fuzzyParams is None:
        raise ValueError("Optimizer flag on but no fuzzy params provided")
    elif not used_InOptimizer and fuzzyParams is not None:
        e_vec=fuzzyParams['e_vec'], de_vec=fuzzyParams['de_vec'], Y_kp=fuzzyParams['Y_kp'], Y_kd=fuzzyParams['Y_kd']
    elif not used_InOptimizer and fuzzyParams is None and fuzzyParamsPath is None:
        raise ValueError("Optimizer flag off but no fuzzy params and path provided")
    else: 
        data = np.load(fuzzyParamsPath)

        keys = list(data.keys())

        # Detect key naming convention
        try:
            evec_key = next(k for k in keys if 'evec' in k.lower())
            devec_key = next(k for k in keys if 'devec' in k.lower())
            dkp_key = next(k for k in keys if 'dkp' in k.lower())
            dkd_key = next(k for k in keys if 'dkd' in k.lower())
        except StopIteration:
            raise ValueError("NPZ file is missing required lookup table keys")

        e_vec = data[evec_key]
        de_vec = data[devec_key]
        Y_kp = data[dkp_key]
        Y_kd = data[dkd_key]

    fuzzyParams = {
        'e_vec': e_vec,
        'de_vec': de_vec,
        'Y_kp': Y_kp,
        'Y_kd': Y_kd
    }
    # Unpack sim_config (Python), pass only arrays/scalars to Numba core
    S  = sim_config['S'];   M  = sim_config['M']
    MList = sim_config['MList']; GList = sim_config['GList']
    g  = sim_config['g'];   Ftip = sim_config['Ftip']
    theta_Pose = sim_config['theta_Pose']; T_d = sim_config['T_d']
    tauLim = sim_config['tauLim']; tauInit = sim_config['tauInit']; wLim = sim_config['wLim']

    SAMPLE_TIME = sim_config['SAMPLE_TIME']; STOP_TIME = sim_config['STOP_TIME']
    startTime   = sim_config['startTime'];   stepTimeXY = sim_config['stepTimeXY']
    stepTimeZ   = sim_config['stepTimeZ'];   randomPairs = sim_config['randomPairs']

    Kp_IK_WzXY = sim_config['Kp_IK_WzXY']; Kd_IK_WzXY = sim_config['Kd_IK_WzXY']
    Kp_IK_Z    = sim_config['Kp_IK_Z'];    Kd_IK_Z    = sim_config['Kd_IK_Z']

    Kp_base_1 = sim_config['Kp_base_1']; Kd_base_1 = sim_config['Kd_base_1']
    Kp_base_2 = sim_config['Kp_base_2']; Kd_base_2 = sim_config['Kd_base_2']
    alpha = sim_config['alpha']; beta = sim_config['beta']

    Kp_Torque_3 = sim_config['Kp_Torque_3']; Kd_Torque_3 = sim_config['Kd_Torque_3']
    Kp_Torque_4 = sim_config['Kp_Torque_4']; Ki_Torque_4 = sim_config['Ki_Torque_4']; Kd_Torque_4 = sim_config['Kd_Torque_4']

    used_Noise = sim_config['used_Noise']
    NOISE_Gen_thetaRun_Actual    = sim_config['NOISE_Gen_thetaRun_Actual']
    NOISE_Gen_thetaRunDot_Actual = sim_config['NOISE_Gen_thetaRunDot_Actual']
    return _simulation_loop_fuzzy(
        S=S, M=M, MList=MList, GList=GList,
        g=g, Ftip=Ftip,
        theta_Pose=theta_Pose, T_d=T_d,
        tauLim=tauLim, tauInit=tauInit, wLim=wLim,
        SAMPLE_TIME=SAMPLE_TIME, STOP_TIME=STOP_TIME,
        startTime=startTime, stepTimeXY=stepTimeXY, stepTimeZ=stepTimeZ, randomPairs=randomPairs,
        Kp_IK_WzXY=Kp_IK_WzXY, Kd_IK_WzXY=Kd_IK_WzXY, Kp_IK_Z=Kp_IK_Z, Kd_IK_Z=Kd_IK_Z,
        Kp_base_1=Kp_base_1, Kd_base_1=Kd_base_1, Kp_base_2=Kp_base_2, Kd_base_2=Kd_base_2, alpha=alpha, beta=beta,
        Kp_Torque_3=Kp_Torque_3, Kd_Torque_3=Kd_Torque_3, 
        Kp_Torque_4=Kp_Torque_4, Ki_Torque_4=Ki_Torque_4, Kd_Torque_4=Kd_Torque_4,
        e_vec=e_vec, de_vec=de_vec, Y_kp=Y_kp, Y_kd=Y_kd,
        used_Noise=used_Noise, NOISE_Gen_thetaRun_Actual=NOISE_Gen_thetaRun_Actual, NOISE_Gen_thetaRunDot_Actual=NOISE_Gen_thetaRunDot_Actual
    )
    

@njit(cache=True)
def _simulation_loop_fuzzy(# Robot configuration
    S: np.ndarray, M: np.ndarray, MList: np.ndarray, GList: np.ndarray,
    g: np.ndarray, Ftip: np.ndarray, theta_Pose: np.ndarray, T_d: np.ndarray,
    tauLim: np.ndarray, tauInit: np.ndarray, wLim: np.ndarray,
    # Simulation params
    SAMPLE_TIME: float, STOP_TIME: float, startTime: float, stepTimeXY: float, stepTimeZ: float, randomPairs: np.ndarray,
    # Base PID gain
    Kp_IK_WzXY: float, Kd_IK_WzXY: float, Kp_IK_Z: float, Kd_IK_Z: float, 
    Kp_base_1: float, Kd_base_1: float, Kp_base_2: float, Kd_base_2: float,
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
    Vs_log = np.zeros((n_steps, 6), dtype=np.float64)

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
        Vs_log[i, :] = Vs
        
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
    return posInput_log, posOutput_log, torque_log, tVec, Vs_log


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

if __name__ == "__main__":
    os.makedirs("FuzzyLogicOut", exist_ok=True)
    log_file = "FuzzyLogicOut/optimization_log.txt"
    sys.stdout = helper.DualLogger(log_file)
    sys.stderr = sys.stdout

    # Setup
    sim_config = setup_simulation_config()
    fuzzyParamsPath = "FuzzyLogicOut/FuzzySugeno_e_Theta_1.npz"
    
    start_time = time.perf_counter()
    print("Begin simulation...")
    posInput_log, posOutput_log, torque_log, tVec, Vs_log = simulation_loop_fuzzy(sim_config, fuzzyParamsPath=fuzzyParamsPath)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time    
    print("Simulation complete.")
    print(f"elapsed: {elapsed_time:.2f} seconds")

    helper.plot_Continuous(posInput_log, posOutput_log, tVec, extras={'Vs':Vs_log})

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    try:
        sys.stdout.write("✓ Log saved to FuzzyLogicOut/optimization_log.txt\n")
        sys.stdout.close()
    except:
        pass