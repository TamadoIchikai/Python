import PoE.lieTheory as lie
import PoE.kinematics as kine
import PoE.helper as helper
import PoE.dynamics as dyna
import PoE.trajectoryGen as trajGen
import PoE.controller as controller

import numpy as np
import time
import matplotlib.pyplot as plt

def run_simulation(use_fpid=True, label="Default", seed=1):
    """
    Run simulation with optional Fuzzy PID.
    
    Parameters
    ----------
    use_fpid : bool
        If True, use fuzzy adaptive PID. If False, use standard PID.
    label : str
        Label for this simulation
    seed : int
        Random seed for trajectory
    
    Returns
    -------
    results : dict
        Dictionary containing time vector and logged data
    """
    print(f"\n{'='*60}")
    print(f"Running simulation: {label}")
    print(f"Fuzzy PID: {use_fpid}")
    print(f"{'='*60}")
    
    # Robotic configs
    l1 = .35
    l2 = .45
    d1 = .284
    d2 = .1
    d3 = .334
    
    a1 = .035
    b1 = .095
    a2 = .02
    b2 = .08
    r3 = 0.033
    c1 = .05
    c2 = .08
    
    theta_Pose = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0], dtype=np.float64)
    theta_D = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0.4], dtype=np.float64)
    
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
    
    # Dynamics
    Ftip = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
    m1, m2, m3, m4 = 3.0, 2.0, 1.5, 1.5
    g = np.array([0, 0, -9.8], dtype=np.float64)
    tauInit = np.array([0, 0, 0, m4 * g[2]], dtype=np.float64)
    
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
    STOP_TIME = 30.0
    seed = 3
    rng = np.random.default_rng(seed)

    NOISE = {
        "used": True,
        "theta_std": np.deg2rad(0.03),        # joint angle sensor noise [rad]
        "theta_dot_std": np.deg2rad(0.02),
        "torque_std": 0.2
    }

    # Initialize states
    Tsb = kine.PoE_transform(S, M, theta_Pose)
    T_d = kine.PoE_transform(S, M, theta_D)
    
    thetaRun_IK = theta_Pose.copy()
    thetaDotRun_IK = np.zeros_like(thetaRun_IK)
    
    thetaRun_Actual = theta_Pose.copy()
    thetaDotRun_Actual = np.zeros_like(thetaRun_Actual)
    thetaDotDotRun_Actual = np.zeros_like(thetaDotRun_Actual)

    NOISE_thetaRun_Actual = thetaRun_Actual.copy()
    
    wLim = np.array([1, 2, 2, .2], dtype=np.float64)
    tauLim = np.array([20, 15, 5, 20], dtype=np.float64)
    
    # PID controllers
    PID_IK_WzXY = controller.PID_Discrete(Kp=100.0, Ki=0.0, Kd=15.0, Ts=SAMPLE_TIME)
    PID_IK_Z = controller.PID_Discrete(Kp=70.0, Ki=0.0, Kd=5.0, Ts=SAMPLE_TIME)
    
    PID_Torque_theta_1 = controller.PID_Discrete(Kp=80.0, Ki=0.0, Kd=60.0, Ts=SAMPLE_TIME, outputLimit=(-tauLim[0], tauLim[0]), initial_integral=tauInit[0])
    PID_Torque_theta_2 = controller.PID_Discrete(Kp=150.0, Ki=0.0, Kd=43.74, Ts=SAMPLE_TIME, outputLimit=(-tauLim[1], tauLim[1]), initial_integral=tauInit[1])
    PID_Torque_theta_3 = controller.PID_Discrete(Kp=90.0, Ki=0.0, Kd=5.0, Ts=SAMPLE_TIME, outputLimit=(-tauLim[2], tauLim[2]), initial_integral=tauInit[2])
    PID_Torque_theta_4 = controller.PID_Discrete(Kp=40.0, Ki=20.0, Kd=30.0, Ts=SAMPLE_TIME, outputLimit=(-tauLim[3], tauLim[3]), initial_integral=tauInit[3])

    
    # Load fuzzy controllers if using FPID
    fuzzy_Theta_1 = None
    fuzzy_Theta_2 = None
    KpBase_theta_1 = PID_Torque_theta_1.Kp
    KdBase_theta_1 = PID_Torque_theta_1.Kd
    KpBase_theta_2 = PID_Torque_theta_2.Kp
    KdBase_theta_2 = PID_Torque_theta_2.Kd
    
    if use_fpid:
        try:
            fuzzy_Theta_1 = controller.FuzzyLookupController("FuzzyLogicOut/FuzzySugeno_e_Theta_1.npz")
            fuzzy_Theta_2 = controller.FuzzyLookupController("FuzzyLogicOut/FuzzySugeno_e_Theta_2.npz")
            print("Fuzzy controllers loaded successfully")
        except FileNotFoundError as e:
            print(f"Warning: Could not load fuzzy controllers: {e}")
            print("Falling back to standard PID")
            use_fpid = False
    
    Vs_PID = np.zeros(6, dtype=np.float64)
    torqueEffort = np.zeros_like(thetaRun_IK)
    
    n_steps = int(STOP_TIME / SAMPLE_TIME)
    posInput_log = np.zeros((n_steps, 3), dtype=np.float64)
    posOutput_log = np.zeros((n_steps, 3), dtype=np.float64)
    torque_log = np.zeros((n_steps, thetaRun_IK.size), dtype=np.float64)
    kp_adaptive_log = np.zeros((n_steps, 2), dtype=np.float64)
    kd_adaptive_log = np.zeros((n_steps, 2), dtype=np.float64)
    tVec = np.zeros(n_steps, dtype=np.float64)
    
    traj = trajGen.TrajectoryGen(
        M=M,
        T_d=T_d,
        startTime=2,
        stepTimeXY=2,
        stepTimeZ=5,
        stopTime=STOP_TIME,
        seed=seed
    )
    
    # Main simulation loop
    start_time = time.perf_counter()
    
    for i in range(n_steps):
        t = i * SAMPLE_TIME

        if NOISE['used']:
            NOISE_thetaRun_Actual = thetaRun_Actual + rng.normal(0.0, NOISE['theta_std'], size = n_joint)
            NOISE_thetaDotRun_Actual = thetaDotRun_Actual + rng.normal(0.0, NOISE['theta_dot_std'], size = n_joint) 
            NOISE_Tsb = kine.PoE_transform(S, M, NOISE_thetaRun_Actual)

        R_traj, pos_traj = traj.generate(t)
        T_traj = helper.RpTo_TransMat(R_traj, pos_traj)
        
        Vs = kine.twist_Error(NOISE_Tsb, T_traj) if NOISE['used'] else kine.twist_Error(Tsb, T_traj)
        
        Vs_WzXY = Vs[2:5].copy()
        Vs_Z = Vs[5]
        
        Vs_PID_WzXY = PID_IK_WzXY.update(Vs_WzXY)
        Vs_PID_Z = PID_IK_Z.update(Vs_Z)
        
        Vs_PID[0] = Vs[0]
        Vs_PID[1] = Vs[1]
        Vs_PID[2] = Vs_PID_WzXY[0]
        Vs_PID[3] = Vs_PID_WzXY[1]
        Vs_PID[4] = Vs_PID_WzXY[2]
        Vs_PID[5] = Vs_PID_Z
        
        Js = kine.jacobian_Space(S, NOISE_thetaRun_Actual) if  NOISE['used'] else kine.jacobian_Space(S, thetaRun_Actual)

        if not np.all(np.isfinite(Js)):
            print(f"Bad Jacobian at step {i} — aborting")
            break
        
        JsInv = helper.dls_inverse(Js, 1e-3)
        thetaDotRun_IK = JsInv @ Vs_PID
        
        for j in range(thetaDotRun_IK.size):
            if thetaDotRun_IK[j] < -wLim[j]:
                thetaDotRun_IK[j] = -wLim[j]
            elif thetaDotRun_IK[j] > wLim[j]:
                thetaDotRun_IK[j] = wLim[j]
        
        thetaRun_IK = helper.discrete_Integrator(thetaRun_IK, thetaDotRun_IK, SAMPLE_TIME)
        
        error_theta = thetaRun_IK - NOISE_thetaRun_Actual if NOISE['used'] else (thetaRun_IK - thetaRun_Actual)
        errorDot_theta = thetaDotRun_IK - NOISE_thetaDotRun_Actual if NOISE['used'] else (thetaDotRun_IK - thetaDotRun_Actual)
        
        # Apply fuzzy adaptation if enabled
        if use_fpid and fuzzy_Theta_1 is not None and fuzzy_Theta_2 is not None:
            dKp_theta_1, dKd_theta_1 = fuzzy_Theta_1.get_gains(error_theta[0], errorDot_theta[0])
            dKp_theta_2, dKd_theta_2 = fuzzy_Theta_2.get_gains(error_theta[1], errorDot_theta[1])
            
            PID_Torque_theta_1.Kp = KpBase_theta_1 * dKp_theta_1
            PID_Torque_theta_1.Kd = KdBase_theta_1 * dKd_theta_1
            PID_Torque_theta_2.Kp = KpBase_theta_2 * dKp_theta_2
            PID_Torque_theta_2.Kd = KdBase_theta_2 * dKd_theta_2
            
            kp_adaptive_log[i, 0] = PID_Torque_theta_1.Kp
            kp_adaptive_log[i, 1] = PID_Torque_theta_2.Kp
            kd_adaptive_log[i, 0] = PID_Torque_theta_1.Kd
            kd_adaptive_log[i, 1] = PID_Torque_theta_2.Kd
        
        torqueEffort[0] = PID_Torque_theta_1.update(error_theta[0])
        torqueEffort[1] = PID_Torque_theta_2.update(error_theta[1])
        torqueEffort[2] = PID_Torque_theta_3.update(error_theta[2])
        torqueEffort[3] = PID_Torque_theta_4.update(error_theta[3])
        
        thetaDotDotRun_Actual = dyna.forward_Dynamics(
            S, MList, GList, thetaRun_Actual, thetaDotRun_Actual, torqueEffort, g, Ftip
        )
        
        thetaDotRun_Actual = helper.discrete_Integrator(thetaDotRun_Actual, thetaDotDotRun_Actual, SAMPLE_TIME)
        thetaRun_Actual = helper.discrete_Integrator(thetaRun_Actual, thetaDotRun_Actual, SAMPLE_TIME)
        
        Tsb = kine.PoE_transform(S, M, thetaRun_Actual)
        
        if (i % 5000) == 0:
            print(f"  Progress: {i}/{n_steps} ({i/n_steps*100:.1f}%)")
        
        posInput_log[i, :] = pos_traj
        posOutput_log[i, :] = Tsb[:3, 3]
        torque_log[i, :] = torqueEffort
        tVec[i] = t
    
    elapsed = time.perf_counter() - start_time
    print(f"Simulation completed in {elapsed:.2f}s")
    
    return {
        'tVec': tVec,
        'posInput_log': posInput_log,
        'posOutput_log': posOutput_log,
        'torque_log': torque_log,
        'kp_adaptive_log': kp_adaptive_log,
        'kd_adaptive_log': kd_adaptive_log,
        'label': label,
        'use_fpid': use_fpid
    }


def compute_metrics(ref, output):
    """Compute error metrics."""
    error = ref - output
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    return mae, rmse, max_error


def plot_comparison(results_pid, results_fpid):
    """Create comprehensive comparison plots."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    t = results_pid['tVec']
    ref_x = results_pid['posInput_log'][:, 0]
    ref_y = results_pid['posInput_log'][:, 1]
    
    pid_x = results_pid['posOutput_log'][:, 0]
    pid_y = results_pid['posOutput_log'][:, 1]
    
    fpid_x = results_fpid['posOutput_log'][:, 0]
    fpid_y = results_fpid['posOutput_log'][:, 1]
    
    # 1. X Position Tracking
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, ref_x, 'k--', linewidth=2, label='Reference', alpha=0.7)
    ax1.plot(t, pid_x, 'b-', linewidth=1.5, label='Standard PID', alpha=0.8)
    ax1.plot(t, fpid_x, 'r-', linewidth=1.5, label='Fuzzy PID', alpha=0.8)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('X Position (m)', fontsize=11)
    ax1.set_title('X Position Tracking', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Y Position Tracking
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, ref_y, 'k--', linewidth=2, label='Reference', alpha=0.7)
    ax2.plot(t, pid_y, 'b-', linewidth=1.5, label='Standard PID', alpha=0.8)
    ax2.plot(t, fpid_y, 'r-', linewidth=1.5, label='Fuzzy PID', alpha=0.8)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Y Position (m)', fontsize=11)
    ax2.set_title('Y Position Tracking', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. XY Trajectory Plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ref_x, ref_y, 'k--', linewidth=2, label='Reference', alpha=0.7)
    ax3.plot(pid_x, pid_y, 'b-', linewidth=1.5, label='Standard PID', alpha=0.6)
    ax3.plot(fpid_x, fpid_y, 'r-', linewidth=1.5, label='Fuzzy PID', alpha=0.6)
    ax3.set_xlabel('X Position (m)', fontsize=11)
    ax3.set_ylabel('Y Position (m)', fontsize=11)
    ax3.set_title('XY Trajectory', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 4. X Error (PID)
    error_pid_x = ref_x - pid_x
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, error_pid_x * 1000, 'b-', linewidth=1.5)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('X Error (mm)', fontsize=11)
    ax4.set_title('X Tracking Error - Standard PID', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    mae, rmse, max_err = compute_metrics(ref_x, pid_x)
    ax4.text(0.02, 0.98, f'MAE: {mae*1000:.3f}mm\nRMSE: {rmse*1000:.3f}mm\nMax: {max_err*1000:.3f}mm',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 5. Y Error (PID)
    error_pid_y = ref_y - pid_y
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, error_pid_y * 1000, 'b-', linewidth=1.5)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('Y Error (mm)', fontsize=11)
    ax5.set_title('Y Tracking Error - Standard PID', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    mae, rmse, max_err = compute_metrics(ref_y, pid_y)
    ax5.text(0.02, 0.98, f'MAE: {mae*1000:.3f}mm\nRMSE: {rmse*1000:.3f}mm\nMax: {max_err*1000:.3f}mm',
             transform=ax5.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 6. Combined Error (PID)
    error_pid_mag = np.sqrt(error_pid_x**2 + error_pid_y**2)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, error_pid_mag * 1000, 'b-', linewidth=1.5)
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Position Error (mm)', fontsize=11)
    ax6.set_title('Combined Position Error - Standard PID', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. X Error (FPID)
    error_fpid_x = ref_x - fpid_x
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t, error_fpid_x * 1000, 'r-', linewidth=1.5)
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Time (s)', fontsize=11)
    ax7.set_ylabel('X Error (mm)', fontsize=11)
    ax7.set_title('X Tracking Error - Fuzzy PID', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    mae, rmse, max_err = compute_metrics(ref_x, fpid_x)
    ax7.text(0.02, 0.98, f'MAE: {mae*1000:.3f}mm\nRMSE: {rmse*1000:.3f}mm\nMax: {max_err*1000:.3f}mm',
             transform=ax7.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # 8. Y Error (FPID)
    error_fpid_y = ref_y - fpid_y
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, error_fpid_y * 1000, 'r-', linewidth=1.5)
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Time (s)', fontsize=11)
    ax8.set_ylabel('Y Error (mm)', fontsize=11)
    ax8.set_title('Y Tracking Error - Fuzzy PID', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    mae, rmse, max_err = compute_metrics(ref_y, fpid_y)
    ax8.text(0.02, 0.98, f'MAE: {mae*1000:.3f}mm\nRMSE: {rmse*1000:.3f}mm\nMax: {max_err*1000:.3f}mm',
             transform=ax8.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # 9. Combined Error (FPID)
    error_fpid_mag = np.sqrt(error_fpid_x**2 + error_fpid_y**2)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(t, error_fpid_mag * 1000, 'r-', linewidth=1.5)
    ax9.set_xlabel('Time (s)', fontsize=11)
    ax9.set_ylabel('Position Error (mm)', fontsize=11)
    ax9.set_title('Combined Position Error - Fuzzy PID', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    # 10. Error Comparison (PID vs FPID)
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(t, error_pid_mag * 1000, 'b-', linewidth=1.5, label='Standard PID', alpha=0.7)
    ax10.plot(t, error_fpid_mag * 1000, 'r-', linewidth=1.5, label='Fuzzy PID', alpha=0.7)
    ax10.set_xlabel('Time (s)', fontsize=11)
    ax10.set_ylabel('Position Error (mm)', fontsize=11)
    ax10.set_title('Error Comparison', fontsize=12, fontweight='bold')
    ax10.legend(fontsize=9)
    ax10.grid(True, alpha=0.3)
    
    # 11. Torque Comparison - Joint 1
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(t, results_pid['torque_log'][:, 0], 'b-', linewidth=1, label='Standard PID', alpha=0.7)
    ax11.plot(t, results_fpid['torque_log'][:, 0], 'r-', linewidth=1, label='Fuzzy PID', alpha=0.7)
    ax11.set_xlabel('Time (s)', fontsize=11)
    ax11.set_ylabel('Torque (Nm)', fontsize=11)
    ax11.set_title('Joint 1 Torque', fontsize=12, fontweight='bold')
    ax11.legend(fontsize=9)
    ax11.grid(True, alpha=0.3)
    
    # 12. Torque Comparison - Joint 2
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(t, results_pid['torque_log'][:, 1], 'b-', linewidth=1, label='Standard PID', alpha=0.7)
    ax12.plot(t, results_fpid['torque_log'][:, 1], 'r-', linewidth=1, label='Fuzzy PID', alpha=0.7)
    ax12.set_xlabel('Time (s)', fontsize=11)
    ax12.set_ylabel('Torque (Nm)', fontsize=11)
    ax12.set_title('Joint 2 Torque', fontsize=12, fontweight='bold')
    ax12.legend(fontsize=9)
    ax12.grid(True, alpha=0.3)
    
    plt.show()


def plot_adaptive_gains(results_fpid):
    """Plot adaptive gains over time."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    t = results_fpid['tVec']
    
    # Kp Joint 1
    axes[0, 0].plot(t, results_fpid['kp_adaptive_log'][:, 0], 'r-', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (s)', fontsize=11)
    axes[0, 0].set_ylabel('Kp (adaptive)', fontsize=11)
    axes[0, 0].set_title('Adaptive Kp - Joint 1', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Kd Joint 1
    axes[0, 1].plot(t, results_fpid['kd_adaptive_log'][:, 0], 'b-', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)', fontsize=11)
    axes[0, 1].set_ylabel('Kd (adaptive)', fontsize=11)
    axes[0, 1].set_title('Adaptive Kd - Joint 1', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Kp Joint 2
    axes[1, 0].plot(t, results_fpid['kp_adaptive_log'][:, 1], 'r-', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)', fontsize=11)
    axes[1, 0].set_ylabel('Kp (adaptive)', fontsize=11)
    axes[1, 0].set_title('Adaptive Kp - Joint 2', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Kd Joint 2
    axes[1, 1].plot(t, results_fpid['kd_adaptive_log'][:, 1], 'b-', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)', fontsize=11)
    axes[1, 1].set_ylabel('Kd (adaptive)', fontsize=11)
    axes[1, 1].set_title('Adaptive Kd - Joint 2', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_summary_statistics(results_pid, results_fpid):
    """Print summary statistics comparing both methods."""
    ref_x = results_pid['posInput_log'][:, 0]
    ref_y = results_pid['posInput_log'][:, 1]
    
    pid_x = results_pid['posOutput_log'][:, 0]
    pid_y = results_pid['posOutput_log'][:, 1]
    
    fpid_x = results_fpid['posOutput_log'][:, 0]
    fpid_y = results_fpid['posOutput_log'][:, 1]
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS: Standard PID vs Fuzzy PID")
    print("="*80)
    
    # Standard PID statistics
    mae_x_pid, rmse_x_pid, max_x_pid = compute_metrics(ref_x, pid_x)
    mae_y_pid, rmse_y_pid, max_y_pid = compute_metrics(ref_y, pid_y)
    
    # Fuzzy PID statistics
    mae_x_fpid, rmse_x_fpid, max_x_fpid = compute_metrics(ref_x, fpid_x)
    mae_y_fpid, rmse_y_fpid, max_y_fpid = compute_metrics(ref_y, fpid_y)
    
    print("\n--- STANDARD PID ---")
    print(f"X-axis: MAE={mae_x_pid*1000:.4f}mm, RMSE={rmse_x_pid*1000:.4f}mm, Max={max_x_pid*1000:.4f}mm")
    print(f"Y-axis: MAE={mae_y_pid*1000:.4f}mm, RMSE={rmse_y_pid*1000:.4f}mm, Max={max_y_pid*1000:.4f}mm")
    
    print("\n--- FUZZY PID ---")
    print(f"X-axis: MAE={mae_x_fpid*1000:.4f}mm, RMSE={rmse_x_fpid*1000:.4f}mm, Max={max_x_fpid*1000:.4f}mm")
    print(f"Y-axis: MAE={mae_y_fpid*1000:.4f}mm, RMSE={rmse_y_fpid*1000:.4f}mm, Max={max_y_fpid*1000:.4f}mm")
    
    print("\n--- IMPROVEMENT (Fuzzy PID vs Standard PID) ---")
    improvement_x_mae = (mae_x_pid - mae_x_fpid) / mae_x_pid * 100 if mae_x_pid > 0 else 0
    improvement_y_mae = (mae_y_pid - mae_y_fpid) / mae_y_pid * 100 if mae_y_pid > 0 else 0
    improvement_x_rmse = (rmse_x_pid - rmse_x_fpid) / rmse_x_pid * 100 if rmse_x_pid > 0 else 0
    improvement_y_rmse = (rmse_y_pid - rmse_y_fpid) / rmse_y_pid * 100 if rmse_y_pid > 0 else 0
    
    print(f"X-axis MAE improvement: {improvement_x_mae:+.2f}%")
    print(f"Y-axis MAE improvement: {improvement_y_mae:+.2f}%")
    print(f"X-axis RMSE improvement: {improvement_x_rmse:+.2f}%")
    print(f"Y-axis RMSE improvement: {improvement_y_rmse:+.2f}%")
    
    print("="*80)


if __name__ == "__main__":
    # Run with standard PID
    seed = 3
    results_pid = run_simulation(
        use_fpid=False,
        label="Standard PID",
        seed=seed
    )
    
    # Run with Fuzzy PID
    results_fpid = run_simulation(
        use_fpid=True,
        label="Fuzzy PID",
        seed=seed
    )
    
    # Print summary statistics
    print_summary_statistics(results_pid, results_fpid)
    
    # Create comparison plots
    plot_comparison(results_pid, results_fpid)
    
    # Plot adaptive gains
    if results_fpid['use_fpid']:
        plot_adaptive_gains(results_fpid)