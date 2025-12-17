import PoE.lieTheory as lie
import PoE.kinematics as kine
import PoE.helper as helper
import PoE.dynamics as dyna
import PoE.trajectoryGen as trajGen
import PoE.controller as controller

import numpy as np
import time
import matplotlib.pyplot as plt


def run_simulation(Kp1, Kd1, Kp2, Kd2, label="Default", seed=1):
    """
    Run simulation with specified PID parameters.
    
    Returns:
    --------
    results : dict
        Dictionary containing time vector and logged data
    """
    print(f"\n{'='*60}")
    print(f"Running simulation: {label}")
    print(f"PID_theta_1: Kp={Kp1:.2f}, Kd={Kd1:.2f}")
    print(f"PID_theta_2: Kp={Kp2:.2f}, Kd={Kd2:.2f}")
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
    STOP_TIME = 60.0
    
    N = 100
    all_points = np.array([[x, y] for x in range(3) for y in range(3)], dtype=np.int64)
    randomList = []
    np.random.seed(42)  # Fixed seed for reproducibility
    while len(randomList) < N:
        shuffled = all_points[np.random.permutation(9)]
        randomList.append(shuffled)
    randomList = np.vstack(randomList)[:N].astype(np.int64)
    
    # Initialize states
    Tsb = kine.PoE_transform(S, M, theta_Pose)
    T_d = kine.PoE_transform(S, M, theta_D)
    
    thetaRun_IK = theta_Pose.copy()
    thetaDotRun_IK = np.zeros_like(thetaRun_IK)
    
    thetaRun_Actual = theta_Pose.copy()
    thetaDotRun_Actual = np.zeros_like(thetaRun_Actual)
    thetaDotDotRun_Actual = np.zeros_like(thetaDotRun_Actual)
    
    wLim = np.array([1, 2, 2, .2], dtype=np.float64)
    tauLim = np.array([20, 15, 5, 20], dtype=np.float64)
    
    # PID controllers
    PID_IK_WzXY = controller.PID_Discrete(Kp=80.0, Ki=0.0, Kd=15.0, Ts=SAMPLE_TIME)
    PID_IK_Z = controller.PID_Discrete(Kp=45.0, Ki=0.0, Kd=5.0, Ts=SAMPLE_TIME)
    
    # Use provided PID parameters
    PID_Torque_theta_1 = controller.PID_Discrete(
        Kp=Kp1, Ki=0.0, Kd=Kd1, Ts=SAMPLE_TIME,
        outputLimit=(-tauLim[0], tauLim[0]), initial_integral=tauInit[0]
    )
    PID_Torque_theta_2 = controller.PID_Discrete(
        Kp=Kp2, Ki=0.0, Kd=Kd2, Ts=SAMPLE_TIME,
        outputLimit=(-tauLim[1], tauLim[1]), initial_integral=tauInit[1]
    )
    PID_Torque_theta_3 = controller.PID_Discrete(
        Kp=60.0, Ki=0.0, Kd=5.0, Ts=SAMPLE_TIME,
        outputLimit=(-tauLim[2], tauLim[2]), initial_integral=tauInit[2]
    )
    PID_Torque_theta_4 = controller.PID_Discrete(
        Kp=40.0, Ki=20.0, Kd=30.0, Ts=SAMPLE_TIME,
        outputLimit=(-tauLim[3], tauLim[3]), initial_integral=tauInit[3]
    )
    
    Vs_PID = np.zeros(6, dtype=np.float64)
    torqueEffort = np.zeros_like(thetaRun_IK)
    
    n_steps = int(STOP_TIME / SAMPLE_TIME)
    posInput_log = np.zeros((n_steps, 3), dtype=np.float64)
    posOutput_log = np.zeros((n_steps, 3), dtype=np.float64)
    torque_log = np.zeros((n_steps, thetaRun_IK.size), dtype=np.float64)
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
        
        R_traj, pos_traj =traj.generate(t)
        T_traj = helper.RpTo_TransMat(R_traj, pos_traj)
        
        Vs = kine.twist_Error(Tsb, T_traj)
        
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
        
        Js = kine.jacobian_Space(S, thetaRun_Actual)
        
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
        
        error_theta = thetaRun_IK - thetaRun_Actual
        
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
        
        if (i % 10000) == 0:
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
        'label': label
    }


def compute_metrics(ref, output):
    """Compute error metrics."""
    error = ref - output
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))
    return mae, rmse, max_error


def plot_comparison(results_default, results_optimized):
    """
    Create comprehensive comparison plots.
    """
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    t = results_default['tVec']
    ref_x = results_default['posInput_log'][:, 0]
    ref_y = results_default['posInput_log'][:, 1]
    
    default_x = results_default['posOutput_log'][:, 0]
    default_y = results_default['posOutput_log'][:, 1]
    
    optimized_x = results_optimized['posOutput_log'][:, 0]
    optimized_y = results_optimized['posOutput_log'][:, 1]
    
    # 1. X Position Tracking
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, ref_x, 'k--', linewidth=2, label='Reference', alpha=0.7)
    ax1.plot(t, default_x, 'b-', linewidth=1.5, label='Default PID', alpha=0.8)
    ax1.plot(t, optimized_x, 'r-', linewidth=1.5, label='Optimized PID', alpha=0.8)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('X Position (m)', fontsize=11)
    ax1.set_title('X Position Tracking', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Y Position Tracking
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, ref_y, 'k--', linewidth=2, label='Reference', alpha=0.7)
    ax2.plot(t, default_y, 'b-', linewidth=1.5, label='Default PID', alpha=0.8)
    ax2.plot(t, optimized_y, 'r-', linewidth=1.5, label='Optimized PID', alpha=0.8)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Y Position (m)', fontsize=11)
    ax2.set_title('Y Position Tracking', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. XY Trajectory Plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ref_x, ref_y, 'k--', linewidth=2, label='Reference', alpha=0.7)
    ax3.plot(default_x, default_y, 'b-', linewidth=1.5, label='Default PID', alpha=0.6)
    ax3.plot(optimized_x, optimized_y, 'r-', linewidth=1.5, label='Optimized PID', alpha=0.6)
    ax3.set_xlabel('X Position (m)', fontsize=11)
    ax3.set_ylabel('Y Position (m)', fontsize=11)
    ax3.set_title('XY Trajectory', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # 4. X Error (Reference vs Default)
    error_default_x = ref_x - default_x
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t, error_default_x * 1000, 'b-', linewidth=1.5, label='Default PID')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('X Error (mm)', fontsize=11)
    ax4.set_title('X Tracking Error - Default PID', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    mae, rmse, max_err = compute_metrics(ref_x, default_x)
    ax4.text(0.02, 0.98, f'MAE: {mae*1000:.3f}mm\nRMSE: {rmse*1000:.3f}mm\nMax: {max_err*1000:.3f}mm',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. Y Error (Reference vs Default)
    error_default_y = ref_y - default_y
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t, error_default_y * 1000, 'b-', linewidth=1.5, label='Default PID')
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_ylabel('Y Error (mm)', fontsize=11)
    ax5.set_title('Y Tracking Error - Default PID', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    mae, rmse, max_err = compute_metrics(ref_y, default_y)
    ax5.text(0.02, 0.98, f'MAE: {mae*1000:.3f}mm\nRMSE: {rmse*1000:.3f}mm\nMax: {max_err*1000:.3f}mm',
             transform=ax5.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Combined Error (Default)
    error_default_mag = np.sqrt(error_default_x**2 + error_default_y**2)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(t, error_default_mag * 1000, 'b-', linewidth=1.5)
    ax6.set_xlabel('Time (s)', fontsize=11)
    ax6.set_ylabel('Position Error (mm)', fontsize=11)
    ax6.set_title('Combined Position Error - Default PID', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. X Error (Reference vs Optimized)
    error_optimized_x = ref_x - optimized_x
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.plot(t, error_optimized_x * 1000, 'r-', linewidth=1.5, label='Optimized PID')
    ax7.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Time (s)', fontsize=11)
    ax7.set_ylabel('X Error (mm)', fontsize=11)
    ax7.set_title('X Tracking Error - Optimized PID', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    mae, rmse, max_err = compute_metrics(ref_x, optimized_x)
    ax7.text(0.02, 0.98, f'MAE: {mae*1000:.3f}mm\nRMSE: {rmse*1000:.3f}mm\nMax: {max_err*1000:.3f}mm',
             transform=ax7.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 8. Y Error (Reference vs Optimized)
    error_optimized_y = ref_y - optimized_y
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.plot(t, error_optimized_y * 1000, 'r-', linewidth=1.5, label='Optimized PID')
    ax8.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Time (s)', fontsize=11)
    ax8.set_ylabel('Y Error (mm)', fontsize=11)
    ax8.set_title('Y Tracking Error - Optimized PID', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    mae, rmse, max_err = compute_metrics(ref_y, optimized_y)
    ax8.text(0.02, 0.98, f'MAE: {mae*1000:.3f}mm\nRMSE: {rmse*1000:.3f}mm\nMax: {max_err*1000:.3f}mm',
             transform=ax8.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 9. Combined Error (Optimized)
    error_optimized_mag = np.sqrt(error_optimized_x**2 + error_optimized_y**2)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(t, error_optimized_mag * 1000, 'r-', linewidth=1.5)
    ax9.set_xlabel('Time (s)', fontsize=11)
    ax9.set_ylabel('Position Error (mm)', fontsize=11)
    ax9.set_title('Combined Position Error - Optimized PID', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    # 10. Difference in X output (Default vs Optimized)
    diff_x = default_x - optimized_x
    ax10 = fig.add_subplot(gs[3, 0])
    ax10.plot(t, diff_x * 1000, 'g-', linewidth=1.5)
    ax10.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax10.set_xlabel('Time (s)', fontsize=11)
    ax10.set_ylabel('X Difference (mm)', fontsize=11)
    ax10.set_title('Output Difference: Default - Optimized (X)', fontsize=12, fontweight='bold')
    ax10.grid(True, alpha=0.3)
    mae_diff = np.mean(np.abs(diff_x))
    ax10.text(0.02, 0.98, f'MAE: {mae_diff*1000:.3f}mm',
             transform=ax10.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 11. Difference in Y output (Default vs Optimized)
    diff_y = default_y - optimized_y
    ax11 = fig.add_subplot(gs[3, 1])
    ax11.plot(t, diff_y * 1000, 'g-', linewidth=1.5)
    ax11.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax11.set_xlabel('Time (s)', fontsize=11)
    ax11.set_ylabel('Y Difference (mm)', fontsize=11)
    ax11.set_title('Output Difference: Default - Optimized (Y)', fontsize=12, fontweight='bold')
    ax11.grid(True, alpha=0.3)
    mae_diff = np.mean(np.abs(diff_y))
    ax11.text(0.02, 0.98, f'MAE: {mae_diff*1000:.3f}mm',
             transform=ax11.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 12. Error Comparison (Default vs Optimized)
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.plot(t, error_default_mag * 1000, 'b-', linewidth=1.5, label='Default PID', alpha=0.7)
    ax12.plot(t, error_optimized_mag * 1000, 'r-', linewidth=1.5, label='Optimized PID', alpha=0.7)
    ax12.set_xlabel('Time (s)', fontsize=11)
    ax12.set_ylabel('Position Error (mm)', fontsize=11)
    ax12.set_title('Error Comparison', fontsize=12, fontweight='bold')
    ax12.legend(fontsize=9)
    ax12.grid(True, alpha=0.3)
    
    plt.savefig('pid_comparison_complete.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to 'pid_comparison_complete.png'")
    plt.show()


def print_summary_statistics(results_default, results_optimized):
    """Print summary statistics comparing both methods."""
    ref_x = results_default['posInput_log'][:, 0]
    ref_y = results_default['posInput_log'][:, 1]
    
    default_x = results_default['posOutput_log'][:, 0]
    default_y = results_default['posOutput_log'][:, 1]
    
    optimized_x = results_optimized['posOutput_log'][:, 0]
    optimized_y = results_optimized['posOutput_log'][:, 1]
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Default PID statistics
    mae_x_def, rmse_x_def, max_x_def = compute_metrics(ref_x, default_x)
    mae_y_def, rmse_y_def, max_y_def = compute_metrics(ref_y, default_y)
    
    # Optimized PID statistics
    mae_x_opt, rmse_x_opt, max_x_opt = compute_metrics(ref_x, optimized_x)
    mae_y_opt, rmse_y_opt, max_y_opt = compute_metrics(ref_y, optimized_y)
    
    print("\n--- DEFAULT PID ---")
    print(f"X-axis: MAE={mae_x_def*1000:.4f}mm, RMSE={rmse_x_def*1000:.4f}mm, Max={max_x_def*1000:.4f}mm")
    print(f"Y-axis: MAE={mae_y_def*1000:.4f}mm, RMSE={rmse_y_def*1000:.4f}mm, Max={max_y_def*1000:.4f}mm")
    
    print("\n--- OPTIMIZED PID ---")
    print(f"X-axis: MAE={mae_x_opt*1000:.4f}mm, RMSE={rmse_x_opt*1000:.4f}mm, Max={max_x_opt*1000:.4f}mm")
    print(f"Y-axis: MAE={mae_y_opt*1000:.4f}mm, RMSE={rmse_y_opt*1000:.4f}mm, Max={max_y_opt*1000:.4f}mm")
    
    print("\n--- IMPROVEMENT ---")
    improvement_x_mae = (mae_x_def - mae_x_opt) / mae_x_def * 100
    improvement_y_mae = (mae_y_def - mae_y_opt) / mae_y_def * 100
    improvement_x_rmse = (rmse_x_def - rmse_x_opt) / rmse_x_def * 100
    improvement_y_rmse = (rmse_y_def - rmse_y_opt) / rmse_y_def * 100
    
    print(f"X-axis MAE improvement: {improvement_x_mae:+.2f}%")
    print(f"Y-axis MAE improvement: {improvement_y_mae:+.2f}%")
    print(f"X-axis RMSE improvement: {improvement_x_rmse:+.2f}%")
    print(f"Y-axis RMSE improvement: {improvement_y_rmse:+.2f}%")
    
    print("="*80)


if __name__ == "__main__":
    # Run with default PID parameters
    results_default = run_simulation(
        Kp1=100.0, Kd1=30.0,
        Kp2=80.0, Kd2=40.0,
        label="Default PID",
        seed=3
    )
    
    # Run with optimized PID parameters
    results_optimized = run_simulation(
        Kp1=489.76, Kd1=293.38,
        Kp2=880.28, Kd2=43.74,
        label="Optimized PID",
        seed=3
    )
    
    # Print summary statistics
    print_summary_statistics(results_default, results_optimized)
    
    # Create comparison plots
    plot_comparison(results_default, results_optimized)
    
    # Save results to file
    np.savez('pid_comparison_results.npz',
             tVec=results_default['tVec'],
             ref_x=results_default['posInput_log'][:, 0],
             ref_y=results_default['posInput_log'][:, 1],
             default_x=results_default['posOutput_log'][:, 0],
             default_y=results_default['posOutput_log'][:, 1],
             optimized_x=results_optimized['posOutput_log'][:, 0],
             optimized_y=results_optimized['posOutput_log'][:, 1])
    
    print("\nResults saved to 'pid_comparison_results.npz'")