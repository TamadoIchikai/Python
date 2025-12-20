import os
import time
import numpy as np
import matplotlib.pyplot as plt  # added

# Reuse existing functions/constants from mainFPID.py
from mainFPID import (
    setup_simulation_config,
    simulation_cost_FuzzyPID,
    encode_rules_to_params,
    get_default_dKp_rules,
    print_rule_table,
    FIXED_RULE_TABLE_DKD,
    params_to_const_matrix,
    interp2d_scalar,
    FIXED_DKD_MATRIX,
)

# Extra imports for logging simulation
import PoE.controller as controller
import PoE.trajectoryGen as trajGen
import PoE.helper as helper
import PoE.kinematics as kine
import PoE.dynamics as dyna

RESULTS_NPZ = "FuzzyLogicOut/fuzzy_optimization_results.npz"

save_dir = os.path.dirname(RESULTS_NPZ)
def simulate_and_log(rule_params: np.ndarray, sim_config: dict, log_stride: int = 5):
    """Run FPID sim and log t, pos_ref, pos_out, err."""
    # Fuzzy LUTs
    kp_const = params_to_const_matrix(rule_params)
    kd_const = FIXED_DKD_MATRIX
    e_range, de_range = sim_config['e_range'], sim_config['de_range']
    mfs_e = controller.gen7tri(e_range[0], e_range[1])
    mfs_de = controller.gen7tri(de_range[0], de_range[1])
    nE = nDE = 201
    e_vec = np.linspace(e_range[0], e_range[1], nE)
    de_vec = np.linspace(de_range[0], de_range[1], nDE)
    Y_kp, Y_kd = controller.eval_grid(e_vec, de_vec, mfs_e, mfs_de, kp_const, kd_const)

    # Unpack config
    S = sim_config['S']; M = sim_config['M']; MList = sim_config['MList']; GList = sim_config['GList']
    g = sim_config['g']; Ftip = sim_config['Ftip']; theta_Pose = sim_config['theta_Pose']
    tauLim = sim_config['tauLim']; tauInit = sim_config['tauInit']; wLim = sim_config['wLim']
    SAMPLE_TIME = sim_config['SAMPLE_TIME']; n_steps = sim_config['n_steps']
    random_pairs = sim_config['random_pairs']; z_init = sim_config['z_init']; z_reach = sim_config['z_reach']
    Kp_base_1 = sim_config['Kp_base_1']; Kd_base_1 = sim_config['Kd_base_1']
    Kp_base_2 = sim_config['Kp_base_2']; Kd_base_2 = sim_config['Kd_base_2']
    dKp_scale_1 = sim_config['dKp_scale_1']; dKd_scale_1 = sim_config['dKd_scale_1']
    dKp_scale_2 = sim_config['dKp_scale_2']; dKd_scale_2 = sim_config['dKd_scale_2']
    # reuse trajectory/timing from config instead of re-declaring
    TRAJ_START = sim_config.get('traj_start', 2.0)
    step_time_xy = sim_config.get('step_time_xy', 3.0)
    step_time_z = sim_config.get('step_time_z', 1.5)

    # States
    thetaRun_IK = theta_Pose.copy()
    thetaRun_Actual = theta_Pose.copy()
    thetaDotRun_Actual = np.zeros(4, dtype=np.float64)
    IK_WzXY_prev_error = np.zeros(3, dtype=np.float64)
    IK_Z_prev_error = 0.0
    Torque_prev_error = np.zeros(4, dtype=np.float64)
    Torque_I = tauInit.copy()
    prev_error_theta_1 = 0.0
    prev_error_theta_2 = 0.0

    # Fixed gains
    Kp_IK_WzXY, Kd_IK_WzXY = 80.0, 15.0
    Kp_IK_Z,   Kd_IK_Z   = 45.0, 5.0
    Kp_3, Kd_3 = 60.0, 5.0
    Kp_4, Ki_4, Kd_4 = 40.0, 20.0, 30.0

    t_log, pos_ref_log, pos_out_log, err_log = [], [], [], []

    for i in range(n_steps):
        t = i * SAMPLE_TIME

        # Trajectory
        if t < TRAJ_START:
            pos_traj = M[:3, 3].copy()
        else:
            x_traj, y_traj = trajGen.tic_tac_toe_gen(random_pairs, TRAJ_START, step_time_xy, t)
            z_traj = trajGen.zAxisUpDown(z_init, z_reach, TRAJ_START, step_time_z, t)
            pos_traj = np.array([x_traj, y_traj, z_traj], dtype=np.float64)

        T_traj = np.eye(4, dtype=np.float64)
        T_traj[:3, 3] = pos_traj

        # Current pose
        Tsb = kine.PoE_transform(S, M, thetaRun_Actual)

        # Twist error
        Vs = kine.twist_Error(Tsb, T_traj)

        # IK PID
        Vs_WzXY = Vs[2:5].copy()
        d_err_WzXY = (Vs_WzXY - IK_WzXY_prev_error) / SAMPLE_TIME
        Vs_PID_WzXY = Kp_IK_WzXY * Vs_WzXY + Kd_IK_WzXY * d_err_WzXY
        IK_WzXY_prev_error = Vs_WzXY.copy()

        Vs_Z = Vs[5]
        d_err_Z = (Vs_Z - IK_Z_prev_error) / SAMPLE_TIME
        Vs_PID_Z = Kp_IK_Z * Vs_Z + Kd_IK_Z * d_err_Z
        IK_Z_prev_error = Vs_Z

        Vs_PID = np.zeros(6, dtype=np.float64)
        Vs_PID[0:2] = Vs[0:2]
        Vs_PID[2:5] = Vs_PID_WzXY
        Vs_PID[5] = Vs_PID_Z

        Js = kine.jacobian_Space(S, thetaRun_Actual)
        JsInv = helper.dls_inverse(Js, 1e-3)
        thetaDotRun_IK = JsInv @ Vs_PID
        thetaDotRun_IK = np.clip(thetaDotRun_IK, -wLim, wLim)
        thetaRun_IK = thetaRun_IK + thetaDotRun_IK * SAMPLE_TIME

        # Fuzzy PID torque
        error_theta = thetaRun_IK - thetaRun_Actual
        de_theta_1 = (error_theta[0] - prev_error_theta_1) / SAMPLE_TIME
        de_theta_2 = (error_theta[1] - prev_error_theta_2) / SAMPLE_TIME

        dKp_1_norm = interp2d_scalar(error_theta[0], de_theta_1, e_vec, de_vec, Y_kp)
        dKd_1_norm = interp2d_scalar(error_theta[0], de_theta_1, e_vec, de_vec, Y_kd)
        dKp_2_norm = interp2d_scalar(error_theta[1], de_theta_2, e_vec, de_vec, Y_kp)
        dKd_2_norm = interp2d_scalar(error_theta[1], de_theta_2, e_vec, de_vec, Y_kd)

        Kp_1 = Kp_base_1 + dKp_1_norm * dKp_scale_1
        Kd_1 = Kd_base_1 + dKd_1_norm * dKd_scale_1
        Kp_2 = Kp_base_2 + dKp_2_norm * dKp_scale_2
        Kd_2 = Kd_base_2 + dKd_2_norm * dKd_scale_2

        d_err_torque = (error_theta - Torque_prev_error) / SAMPLE_TIME
        u1 = Kp_1 * error_theta[0] + Kd_1 * d_err_torque[0]
        u2 = Kp_2 * error_theta[1] + Kd_2 * d_err_torque[1]
        u3 = Kp_3 * error_theta[2] + Kd_3 * d_err_torque[2]
        Torque_I[3] += error_theta[3] * SAMPLE_TIME
        u4 = Kp_4 * error_theta[3] + Ki_4 * Torque_I[3] + Kd_4 * d_err_torque[3]

        torqueEffort = np.clip(np.array([u1, u2, u3, u4]), -tauLim, tauLim)
        Torque_prev_error = error_theta.copy()
        prev_error_theta_1 = error_theta[0]
        prev_error_theta_2 = error_theta[1]

        # Forward dynamics
        thetaDotDotRun = dyna.forward_Dynamics(S, MList, GList, thetaRun_Actual, thetaDotRun_Actual, torqueEffort, g, Ftip)
        if not np.all(np.isfinite(thetaDotDotRun)):
            break
        thetaDotRun_Actual = thetaDotRun_Actual + thetaDotDotRun * SAMPLE_TIME
        thetaRun_Actual = thetaRun_Actual + thetaDotRun_Actual * SAMPLE_TIME

        # Pose after integration
        Tsb_new = kine.PoE_transform(S, M, thetaRun_Actual)
        pos_out = Tsb_new[:3, 3].copy()
        err_xyz = pos_traj - pos_out

        if i % log_stride == 0:
            t_log.append(t)
            pos_ref_log.append(pos_traj)
            pos_out_log.append(pos_out)
            err_log.append(err_xyz)

    return {
        "t": np.array(t_log),
        "pos_ref": np.array(pos_ref_log),
        "pos_out": np.array(pos_out_log),
        "err": np.array(err_log),
    }

def compute_mae_mse(err: np.ndarray):
    mae = np.mean(np.abs(err), axis=0)
    mse = np.mean(err ** 2, axis=0)
    mae_all = np.mean(np.linalg.norm(err, axis=1))
    mse_all = np.mean(np.sum(err * err, axis=1))
    return {"mae": mae, "mse": mse, "mae_all": mae_all, "mse_all": mse_all}

def print_metrics(name, err):
    m = compute_mae_mse(err)
    axes = ["x", "y", "z"]
    print(f"\nMetrics for {name}:")
    for i, ax in enumerate(axes):
        print(f"  {ax}: MAE={m['mae'][i]:.6f}, MSE={m['mse'][i]:.6f}")
    print(f"  overall: MAE={m['mae_all']:.6f}, MSE={m['mse_all']:.6f}")
    return m

def compare_metrics(base, other, label_other):
    def pct_improve(a, b):
        return 0.0 if a <= 0 else (a - b) / a * 100.0
    print(f"\nImprovement vs DEFAULT for {label_other}:")
    axes = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        print(f"  {ax}: MAE {pct_improve(base['mae'][i], other['mae'][i]):+.2f}%, "
              f"MSE {pct_improve(base['mse'][i], other['mse'][i]):+.2f}%")
    print(f"  overall: MAE {pct_improve(base['mae_all'], other['mae_all']):+.2f}%, "
          f"MSE {pct_improve(base['mse_all'], other['mse_all']):+.2f}%")

def plot_tracking_and_error(log_def, log_opt, save_dir=save_dir):
    os.makedirs(save_dir, exist_ok=True)
    t = log_def["t"]; ref = log_def["pos_ref"]; out_d = log_def["pos_out"]; err_d = log_def["err"]
    has_opt = log_opt is not None
    if has_opt:
        t_o = log_opt["t"]; out_o = log_opt["pos_out"]; err_o = log_opt["err"]

    labels = ["X", "Y", "Z"]

    # Tracking
    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, ax in enumerate(axs1):
        ax.plot(t, ref[:, i], "k--", label="ref")
        ax.plot(t, out_d[:, i], "r-", label="default")
        if has_opt:
            ax.plot(t_o, out_o[:, i], "b-", label="optimized")
        ax.set_ylabel(f"{labels[i]} (m)")
        ax.grid(True)
        if i == 0:
            ax.legend()
    axs1[-1].set_xlabel("Time (s)")
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, "COMPARE_tracking_ref_vs_out.png"), dpi=150)

    # Errors
    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, ax in enumerate(axs2):
        ax.plot(t, err_d[:, i], "r-", label="default err")
        if has_opt:
            ax.plot(t_o, err_o[:, i], "b-", label="optimized err")
        ax.set_ylabel(f"{labels[i]} error (m)")
        ax.grid(True)
        if i == 0:
            ax.legend()
    axs2[-1].set_xlabel("Time (s)")
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, "COMPARE_errors_default_vs_optimized.png"), dpi=150)

    # Error difference (default - optimized)
    if has_opt:
        fig3, axs3 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        # Align lengths
        n = min(len(err_d), len(err_o))
        td = t[:n]
        diff = err_d[:n] - err_o[:n]
        for i, ax in enumerate(axs3):
            ax.plot(td, diff[:, i], "m-", label="err_default - err_optimized")
            ax.set_ylabel(f"{labels[i]} Δerr (m)")
            ax.grid(True)
            if i == 0:
                ax.legend()
        axs3[-1].set_xlabel("Time (s)")
        fig3.tight_layout()
        fig3.savefig(os.path.join(save_dir, "COMPARE_error_diff_default_minus_optimized.png"), dpi=150)
        plt.close(fig3)

    plt.close(fig1)
    plt.close(fig2)

def main():
    # Setup simulation config
    sim_config = setup_simulation_config()

    # Default rules and params
    default_dkp = get_default_dKp_rules()
    default_params = encode_rules_to_params(default_dkp)

    # Try to load optimized rules from the last run
    optimized_dkp = None
    optimized_cost = None
    if os.path.exists(RESULTS_NPZ):
        data = np.load(RESULTS_NPZ, allow_pickle=True)
        if "Best_Rules" in data:
            optimized_dkp = data["Best_Rules"].tolist()
        if "Best_Cost" in data:
            optimized_cost = float(data["Best_Cost"])
    if optimized_dkp is None:
        print("\nOptimized table not available; skipping trajectory comparison.")
        print("\nDone.")
        return

    print("\nCollecting trajectories for DEFAULT and OPTIMIZED (with logging)...")
    log_default = simulate_and_log(encode_rules_to_params(default_dkp), sim_config, log_stride=5)
    log_opt = simulate_and_log(encode_rules_to_params(optimized_dkp), sim_config, log_stride=5)

    metrics_def = print_metrics("DEFAULT", log_default["err"])
    metrics_opt = print_metrics("OPTIMIZED", log_opt["err"])
    compare_metrics(metrics_def, metrics_opt, "OPTIMIZED")

    plot_tracking_and_error(log_default, log_opt, save_dir=save_dir)
    print(f"\nPlots saved to {save_dir}:")
    print("  COMPARE_tracking_ref_vs_out.png")
    print("  COMPARE_errors_default_vs_optimized.png")
    print("  COMPARE_error_diff_default_minus_optimized.png")
    print("\nDone.")

if __name__ == "__main__":
    main()