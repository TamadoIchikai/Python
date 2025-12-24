#%% 
import numpy as np
import matplotlib.pyplot as plt
# ===== PoE modules (same as your project) =====
import PoE.lieTheory as lie
import PoE.kinematics as kine
import PoE.helper as helper
import PoE.trajectoryGen as trajGen

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

# ============================================================
# SIMULATION PARAMETERS
# ============================================================
SAMPLE_TIME = 0.001
STOP_TIME   = 10.0
n_steps     = int(STOP_TIME / SAMPLE_TIME)

# ============================================================
# ROBOT SETUP (copied structurally from your config)
# ============================================================
l1, l2 = 0.35, 0.45
d1, d2 = 0.284, 0.015

theta_init = np.zeros(4)

q = np.array([
    [0, l1, l1 + l2, l1 + l2],
    [0, 0, 0, 0],
    [0, d1, d1 + d2, d1 + d2]
], dtype=np.float64)

w = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 0]
], dtype=np.float64)

M = np.array([
    [1, 0, 0, l1 + l2],
    [0, 1, 0, 0],
    [0, 0, 1, d1 + d2],
    [0, 0, 0, 1]
], dtype=np.float64)

S = lie.compute_ScrewMat(w, q, 3, -3)
# Desired pose

theta_d = np.array([np.deg2rad(45.0), np.deg2rad(90.0), np.deg2rad(30.0), 0.3])
T_d = kine.PoE_transform(S, M, theta_d)
#%% 
# ============================================================
# TRAJECTORY SETUP
# ============================================================
startTime   = 3.0
stepTimeXY = 2.0
stepTimeZ  = 3.0

randomPairs = trajGen.random_Index_Pair(n=1000, seed=10)

# ============================================================
# LOGS
# ============================================================
theta_log     = np.zeros((n_steps, 4))
pos_input_log = np.zeros((n_steps, 3))
pos_out_log   = np.zeros((n_steps, 3))
t_log         = np.zeros(n_steps)

# ============================================================
# IK LOOP
# ============================================================
theta = theta_init.copy()

for i in range(n_steps):
    t = i * SAMPLE_TIME

    # Desired trajectory
    if t < startTime:
        R_traj = M[:3, :3]
        p_traj = M[:3, 3]
    else:
        R_traj = T_d[:3, :3]
        p_traj = T_d[:3, 3]

    T_traj = helper.RpTo_TransMat(R_traj, p_traj)

    # Current FK
    Tsb = kine.PoE_transform(S, M, theta)

    # Twist error (SE(3))
    Vs = kine.twist_Error(Tsb, T_traj)

    # Jacobian (space)
    Js = kine.jacobian_Space(S, theta)

    # Damped least squares inverse
    JsInv = helper.dls_inverse(Js, 1e-3)

    # Velocity IK
    theta_dot = JsInv @ Vs

    # Integrate joint angles
    theta = helper.discrete_Integrator(theta, theta_dot, SAMPLE_TIME)

    # Logging
    theta_log[i]     = theta
    pos_input_log[i] = p_traj
    pos_out_log[i]   = Tsb[:3, 3]
    t_log[i]         = t

# ============================================================
# PRINT RESULTS
# ============================================================
print("=" * 60)
print("INVERSE KINEMATICS ONLY – RESULT")
print("=" * 60)

print("\nFinal joint configuration:")
print(theta)

print("\nFinal desired position:")
print(pos_input_log[-1])

print("\nFinal end-effector position:")
print(pos_out_log[-1])

print("\nFinal position error:")
print(pos_input_log[-1] - pos_out_log[-1])

fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

labels = ["X", "Y", "Z"]

for i in range(3):
    axs1[i].plot(t_log, pos_input_log[:, i], "k--", label="Reference")
    axs1[i].plot(t_log, pos_out_log[:, i], label="Output")
    axs1[i].set_ylabel(f"{labels[i]} [m]")
    axs1[i].grid(True)
    axs1[i].legend()

axs1[-1].set_xlabel("Time [s]")
fig1.suptitle("End-Effector Position: Reference vs Output")
plt.tight_layout()
plt.show()

# ============================================================
# POST-PROCESSING: SEPARATE AXIS PLOTS (NO ERROR)
# ============================================================
labels = ["X", "Y", "Z"]

for i, axis in enumerate(labels):

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    # Reference vs Output
    ax.plot(
        t_log,
        pos_input_log[:, i],
        "k--",
        linewidth=2,
        label="Reference"
    )
    ax.plot(
        t_log,
        pos_out_log[:, i],
        linewidth=2,
        label="Output"
    )

    ax.set_title(f"End-Effector {axis}-Axis Tracking")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{axis} position [m]")
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()

    # Save figure
    filename = f"Word_Related/Section_3_2_InverseKinematics/IK_{axis}_tracking.png"
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.show()

# #%% 
# import numpy as np
# import PoE.lieTheory as lie

# def print_individual_exponentials(S, theta):
#     """
#     Compute and print exp([xi_i] * theta_i) for each joint.
    
#     S: 6xn screw axis matrix (space frame)
#     theta: nx1 joint variable vector
#     """
#     n = S.shape[1]

#     print("=" * 70)
#     print("INDIVIDUAL EXPONENTIAL MAPS exp([xi_i] * theta_i)")
#     print("=" * 70)

#     for i in range(n):
#         xi_theta = S[:, i] * theta[i]          # 6x1
#         se3_i = lie.twistTo_se3(xi_theta)      # 4x4 se(3)
#         T_i = lie.exp_se3(se3_i)               # 4x4 SE(3)


#         # print("\nse(3) matrix:")
#         # print(se3_i)

#         print("\nexp(se(3)) = T_{i}:")
#         print(T_i)
# # %%
