import os
import numpy as np
import matplotlib.pyplot as plt

from mainFPID import (
    setup_simulation_config,
    get_default_rules,
    encode_rules_to_params,
    params_to_const_matrices,
    print_rule_table,
    simulation_loop_fuzzy,
)
import PoE.controller as controller

RESULTS_NPZ = "FuzzyLogicOut/fuzzy_optimization_results.npz"
SAVE_DIR = "FuzzyLogicOut"


def run_with_logging(rule_params: np.ndarray, sim_config: dict):
    """Run the numba simulation loop and return logs for plotting."""
    # Build fuzzy LUTs
    kp_const, kd_const = params_to_const_matrices(rule_params)
    e_range, de_range = sim_config["e_range"], sim_config["de_range"]
    mfs_e = controller.gen7tri(e_range[0], e_range[1])
    mfs_de = controller.gen7tri(de_range[0], de_range[1])
    nE = nDE = 301  # match mainFPID resolution
    e_vec = np.linspace(e_range[0], e_range[1], nE)
    de_vec = np.linspace(de_range[0], de_range[1], nDE)
    Y_kp, Y_kd = controller.eval_grid(e_vec, de_vec, mfs_e, mfs_de, kp_const, kd_const)

    pos_in, pos_out, torque_out, t_vec = simulation_loop_fuzzy(
        # Robot configuration
        S=sim_config["S"],
        M=sim_config["M"],
        MList=sim_config["MList"],
        GList=sim_config["GList"],
        g=sim_config["g"],
        Ftip=sim_config["Ftip"],
        theta_Pose=sim_config["theta_Pose"],
        T_d=sim_config["T_d"],
        tauLim=sim_config["tauLim"],
        tauInit=sim_config["tauInit"],
        wLim=sim_config["wLim"],
        # Simulation params
        SAMPLE_TIME=sim_config["SAMPLE_TIME"],
        STOP_TIME=sim_config["STOP_TIME"],
        startTime=sim_config["startTime"],
        stepTimeXY=sim_config["stepTimeXY"],
        stepTimeZ=sim_config["stepTimeZ"],
        randomPairs=sim_config["randomPairs"],
        # Base PID gain
        Kp_base_1=sim_config["Kp_base_1"],
        Kd_base_1=sim_config["Kd_base_1"],
        Kp_base_2=sim_config["Kp_base_2"],
        Kd_base_2=sim_config["Kd_base_2"],
        # IK PID gain
        Kp_IK_WzXY=sim_config["Kp_IK_WzXY"],
        Kd_IK_WzXY=sim_config["Kd_IK_WzXY"],
        Kp_IK_Z=sim_config["Kp_IK_Z"],
        Kd_IK_Z=sim_config["Kd_IK_Z"],
        # Torque PID gain
        Kp_Torque_3=sim_config["Kp_Torque_3"],
        Kd_Torque_3=sim_config["Kd_Torque_3"],
        Kp_Torque_4=sim_config["Kp_Torque_4"],
        Ki_Torque_4=sim_config["Ki_Torque_4"],
        Kd_Torque_4=sim_config["Kd_Torque_4"],
        alpha=sim_config["alpha"],
        beta=sim_config["beta"],
        # FPID params
        e_vec=e_vec,
        de_vec=de_vec,
        Y_kp=Y_kp,
        Y_kd=Y_kd,
        # Noise
        used_Noise=sim_config["used_Noise"],
        NOISE_Gen_thetaRun_Actual=sim_config["NOISE_Gen_thetaRun_Actual"],
        NOISE_Gen_thetaRunDot_Actual=sim_config["NOISE_Gen_thetaRunDot_Actual"],
    )

    err = pos_in - pos_out
    return {
        "t": t_vec,
        "pos_ref": pos_in,
        "pos_out": pos_out,
        "torque_out": torque_out,
        "err": err,
    }


def compute_metrics(err: np.ndarray):
    mae = np.mean(np.abs(err), axis=0)
    mse = np.mean(err ** 2, axis=0)
    mae_all = np.mean(np.linalg.norm(err, axis=1))
    mse_all = np.mean(np.sum(err * err, axis=1))
    return {"mae": mae, "mse": mse, "mae_all": mae_all, "mse_all": mse_all}


def print_metrics(name, err):
    m = compute_metrics(err)
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
        print(
            f"  {ax}: MAE {pct_improve(base['mae'][i], other['mae'][i]):+.2f}%, "
            f"MSE {pct_improve(base['mse'][i], other['mse'][i]):+.2f}%"
        )
    print(
        f"  overall: MAE {pct_improve(base['mae_all'], other['mae_all']):+.2f}%, "
        f"MSE {pct_improve(base['mse_all'], other['mse_all']):+.2f}%"
    )


def plot_tracking(log_def, log_opt=None, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)
    t = log_def["t"]
    ref = log_def["pos_ref"]
    out_d = log_def["pos_out"]
    err_d = log_def["err"]
    tau_d = log_def["torque_out"]
    labels = ["X", "Y", "Z"]

    has_opt = log_opt is not None
    if has_opt:
        t_o = log_opt["t"]
        out_o = log_opt["pos_out"]
        err_o = log_opt["err"]

        # ============================================
    # Tracking (left) + Error (right), side by side
    # ============================================
    fig12, axs = plt.subplots(
        3, 2, figsize=(14, 9), sharex="col"
    )

    for i in range(3):
        # ---- Tracking (LEFT column) ----
        ax_tr = axs[i, 0]
        ax_tr.plot(t, ref[:, i], "k--", label="ref")
        ax_tr.plot(t, out_d[:, i], "r-", label="default")
        if has_opt:
            ax_tr.plot(t_o, out_o[:, i], "b-", label="optimized")
        ax_tr.set_ylabel(f"{labels[i]} (m)")
        ax_tr.set_title("Tracking" if i == 0 else "")
        ax_tr.grid(True)
        if i == 0:
            ax_tr.legend()

        # ---- Error (RIGHT column) ----
        ax_er = axs[i, 1]
        ax_er.plot(t, err_d[:, i], "r-", label="default err")
        if has_opt:
            ax_er.plot(t_o, err_o[:, i], "b-", label="optimized err")
        ax_er.set_ylabel(f"{labels[i]} error (m)")
        ax_er.set_title("Error" if i == 0 else "")
        ax_er.grid(True)
        if i == 0:
            ax_er.legend()

    axs[-1, 0].set_xlabel("Time (s)")
    axs[-1, 1].set_xlabel("Time (s)")

    fig12.tight_layout()
    fig12.savefig(
        os.path.join(save_dir, "COMPARE_tracking_and_error_side_by_side.png"),
        dpi=150,
    )

    # Error difference
    if has_opt:
        fig3, axs3 = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
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

    has_opt_tau = has_opt and ("torque_out" in log_opt)
    if has_opt_tau:
        tau_o = log_opt["torque_out"]

    joint_labels = ["J1", "J2", "J3", "J4"]

    fig4, axs4 = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for j, ax in enumerate(axs4):
        ax.plot(t, tau_d[:, j], "r-", label="default")
        if has_opt_tau:
            ax.plot(t_o, tau_o[:, j], "b-", label="optimized")
        ax.set_ylabel(f"τ{joint_labels[j]} (Nm)")
        ax.grid(True)
        if j == 0:
            ax.legend()

    axs4[-1].set_xlabel("Time (s)")
    fig4.tight_layout()
    fig4.savefig(
        os.path.join(save_dir, "COMPARE_torque_default_vs_optimized.png"),
        dpi=150,
    )
    plt.close(fig4)
    plt.close(fig12)


def load_optimized_tables():
    if not os.path.exists(RESULTS_NPZ):
        return None, None, None
    data = np.load(RESULTS_NPZ, allow_pickle=True)
    dkp = data["Best_dKp"].tolist() if "Best_dKp" in data else None
    dkd = data["Best_dKd"].tolist() if "Best_dKd" in data else None
    cost = float(data["Best_Cost"]) if "Best_Cost" in data else None
    return dkp, dkd, cost


def main():
    sim_config = setup_simulation_config()

    # Default tables
    default_dkp, default_dkd = get_default_rules()
    print_rule_table(default_dkp, "DEFAULT dKp")
    print_rule_table(default_dkd, "DEFAULT dKd")
    default_params = encode_rules_to_params(default_dkp, default_dkd)

    # Load optimized
    opt_dkp, opt_dkd, opt_cost = load_optimized_tables()
    if opt_dkp and opt_dkd:
        print_rule_table(opt_dkp, "OPTIMIZED dKp")
        print_rule_table(opt_dkd, "OPTIMIZED dKd")
    else:
        print("\nOptimized tables not found; will run DEFAULT only.")

    # Run sims
    print("\nRunning DEFAULT...")
    log_def = run_with_logging(default_params, sim_config)

    if opt_dkp and opt_dkd:
        print("Running OPTIMIZED...")
        log_opt = run_with_logging(encode_rules_to_params(opt_dkp, opt_dkd), sim_config)
    else:
        log_opt = None

    # Metrics
    metrics_def = print_metrics("DEFAULT", log_def["err"])
    if log_opt:
        metrics_opt = print_metrics("OPTIMIZED", log_opt["err"])
        compare_metrics(metrics_def, metrics_opt, "OPTIMIZED")

    # Plots
    plot_tracking_and_error = plot_tracking  # alias
    plot_tracking_and_error(log_def, log_opt, save_dir=SAVE_DIR)

    print(f"\nPlots saved to {SAVE_DIR}:")
    print("  COMPARE_tracking_ref_vs_out.png")
    print("  COMPARE_errors_default_vs_optimized.png")
    if log_opt:
        print("  COMPARE_error_diff_default_minus_optimized.png")
    print("\nDone.")


if __name__ == "__main__":
    main()