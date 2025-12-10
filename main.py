# %% Libs
import PoE.lieTheory as lie
import PoE.kinematics as kine
import PoE.helper as helper
import PoE.dynamics as dyna
import PoE.trajectoryGen as trajGen
import PoE.controller as controller

import numpy as np
import time
# %% Robotic configs
# Dimension in m
# link 1 and link 2 length (along x axis)
l1 = .35
l2 = .45
# link 1 and link 2 height (along z axis)
d1 = .284
d2 = .1
d3 = .334 # prismatic joint max range

# link 1 and link 2 thickness
a1 = .035
b1 = .095

a2 = .02
b2 = .08

# link 3 and link 4 radius
r3 = 0.033

# displacement of links compare to it's axis of rotation
c1 = .05
c2 = .08

# Example angle
theta_Pose = np.array([np.deg2rad(0),
                        np.deg2rad(0),
                        np.deg2rad(0),
                                0])

theta_D= np.array([np.deg2rad(0),
                    np.deg2rad(0),
                    np.deg2rad(0),
                                0])
n = [3, -3] # prismatic joint 4 -z axis

q = np.array([[0,     l1,      l1+l2,      l1+l2],
                [0,     0,       0,          0],
                [0,     d1,      d1+d2,      d1+d2]])

w = np.array([[0,     0,       0,       0],
                [0,     0,       0,       0],
                [1,     1,       1,       0]])

M = np.array([[1,     0,       0,       l1+l2],
                [0,     1,       0,       0],
                [0,     0,       1,       d1+d2],
                [0,     0,       0,       1]])

n_joint = theta_Pose.size
S = lie.compute_ScrewMat(w,q,n)

print(kine.PoE_transform(S, M, theta_Pose))
# helper.validate_joint_inputs(q, w, M, S, theta_Pose)
# %% Dynamics
Ftip = np.array([0,0,0,0,0,0], dtype=np.float32)
m1, m2, m3, m4 = 3, 2, 1.5, 1.5 #kg

g = np.array([0,0,-9.8], dtype=np.float32)
tauInit = np.array([0,0,0,m4 * g[2]], dtype=np.float32)

MList = np.array(np.zeros((4,4,n_joint+1)), dtype=np.float32)
MList[:,:,0] = np.vstack([np.hstack([np.eye(3,3), np.zeros((3,1))]), np.hstack([np.zeros((1,3)), [[1]]])])
MList[:,:,1] = np.vstack([np.hstack([np.eye(3,3), np.array([[l1],[0],[d1]])]), np.hstack([np.zeros((1,3)), [[1]]])])
MList[:,:,2] = np.vstack([np.hstack([np.eye(3,3), np.array([[l2],[0],[d2]])]), np.hstack([np.zeros((1,3)), [[1]]])])
MList[:,:,3] = MList[:,:,0].copy()
MList[:,:,4] = MList[:,:,0].copy()


# I1 CoM
I1CoM = (m1/12) * np.array([
    [a1**2 + b1**2,      0,               0],
    [0,                  l1**2 + a1**2,   0],
    [0,                  0,               b1**2 + l1**2]
], dtype=np.float32)

# I2 CoM
I2CoM = (m2/12) * np.array([
    [a2**2 + b2**2,      0,               0],
    [0,                  l2**2 + a2**2,   0],
    [0,                  0,               b2**2 + l2**2]
],dtype=np.float32)

# I3 CoM  (cylindrical / prismatic)
I3CoM = m3 * np.array([
    [(1/12)*(d3**2 + 3*r3**2),    0,                              0],
    [0,                          (1/12)*(d3**2 + 3*r3**2),       0],
        [0,                           0,                              (1/2)*r3**2]
], dtype=np.float32)

# I4 CoM (same as I3)
I4CoM = I3CoM.copy()

f1 = np.array([l1/2 - c1, 0.0, 0.0],dtype=np.float32)
f2 = np.array([l2/2 - c2, 0.0, 0.0],dtype=np.float32)
f3 = np.array([0.0, 0.0, 0.0],dtype=np.float32)
f4 = np.array([0.0, 0.0, 0.0],dtype=np.float32)

GList = np.asarray(np.zeros((6,6,n_joint)), dtype=np.float32)
GList[:,:,0] = helper.mcI(m1, f1, I1CoM)
GList[:,:,1] = helper.mcI(m2, f2, I2CoM)
GList[:,:,2] = helper.mcI(m3, f3, I3CoM)
GList[:,:,3] = helper.mcI(m4, f4, I4CoM)

# helper.validate_Dynamics_Inputs(Ftip, g, MList, GList, theta_Pose)
# %% testing 
# thetaDotTest = np.array([1, 2, 2, .2])
# thetaDotDotTest = np.array([1, 2, 2, .2])
# FD = dyna.forward_Dynamics(S, MList, GList, theta_Pose, thetaDotTest, tauInit, g, Ftip)
# ID = dyna.inverse_Dynamics(S, MList, GList, theta_Pose, thetaDotTest, FD, g, Ftip)

# print(FD)
# print(ID)
# %% Some inverse kinematic
SAMPLE_TIME = 0.0001
TIME_STOP = 60

N = 100
all_points = np.array([[x, y] for x in range(3) for y in range(3)])   # shape (9,2)
randomList = []
while len(randomList)<N:
    shuffled = all_points[np.random.permutation(9)]   # random perm of 9 rows
    randomList.append(shuffled)

randomList = np.vstack(randomList)[:N]    # trim to exactly N rows

# %% Looping time
Tsb = kine.PoE_transform(S, M, theta_Pose)
T_d = kine.PoE_transform(S, M, theta_D)

thetaRun_IK = theta_Pose.copy()
thetaDotRun_IK = np.zeros_like(thetaRun_IK)

thetaRun_Actual = theta_Pose.copy()
thetaDotRun_Actual = np.zeros_like(thetaRun_Actual)
thetaDotDotRun_Actual = np.zeros_like(thetaDotRun_Actual)

wLim = np.array([1, 2, 2, .2], dtype=np.float32)
tauLim = np.array([20, 15, 5, 20], dtype=np.float32)

PID_IK_WzXY =  controller.PID_Discrete(Kp=80.0, Ki= 0.0, Kd = 15, Ts=SAMPLE_TIME)
PID_IK_Z = controller.PID_Discrete(Kp=45.0, Ki= 0.0, Kd = 5, Ts=SAMPLE_TIME)

PID_Torque_theta_1 = controller.PID_Discrete(Kp=100.0, Ki= 0.0, Kd = 30.0, Ts=SAMPLE_TIME, outputLimit=(-tauLim[0], tauLim[0]), initial_integral=tauInit[0])
PID_Torque_theta_2 = controller.PID_Discrete(Kp=80.0, Ki= 0.0, Kd = 40.0, Ts=SAMPLE_TIME, outputLimit=(-tauLim[1], tauLim[1]), initial_integral=tauInit[1])
PID_Torque_theta_3 = controller.PID_Discrete(Kp=60.0, Ki= 0.0, Kd = 5, Ts=SAMPLE_TIME, outputLimit=(-tauLim[2], tauLim[2]), initial_integral=tauInit[2])
PID_Torque_theta_4 = controller.PID_Discrete(Kp=40.0, Ki= 20.0, Kd = 30.0, Ts=SAMPLE_TIME, outputLimit=(-tauLim[3], tauLim[3]), initial_integral=tauInit[3])

Vs_PID = np.zeros(6, dtype=np.float32) 
torqueEffort = np.zeros_like(thetaRun_IK)

n_steps = int(TIME_STOP / SAMPLE_TIME)
posInput_log = np.zeros((n_steps, 3), dtype=float)
posOutput_log = np.zeros((n_steps, 3), dtype=float)
torque_log = np.zeros((n_steps, thetaRun_IK.size), dtype=float)
thetaDotDot_Log = np.zeros((n_steps, thetaRun_IK.size), dtype=float)

tVec = np.zeros(n_steps, dtype=float)
t = 0

counterFDynamic = []
counterLoop_Start = time.perf_counter()
for i in range(n_steps):
    t = i * SAMPLE_TIME

    R_traj, pos_traj, k = trajGen.tic_tac_toe_gen(M, T_d, randomList, t)
    T_traj = helper.RpTo_TransMat(R_traj, pos_traj)

    Vs = kine.twist_Error(Tsb, T_traj)

    Vs_WzXY = Vs[2:5]
    Vs_Z = Vs[5]

    Vs_PID_WzXY = PID_IK_WzXY.update(Vs_WzXY)
    Vs_PID_Z = PID_IK_Z.update(Vs_Z)
    
    Vs_PID[0] = Vs[0]      # Vx (unchanged)
    Vs_PID[1] = Vs[1]      # Vy (unchanged)
    Vs_PID[2:5] = Vs_PID_WzXY   # Wz, Wx, Wy (PID output)
    Vs_PID[5] = Vs_PID_Z

    Js = kine.jacobian_Space(S, thetaRun_Actual)

    if not np.all(np.isfinite(Js)):
        print("Bad Jacobian (NaN/inf) — aborting step. thetaRun:", thetaRun_IK)
        break

    JsInv = helper.dls_inverse(Js, 1e-3)

    thetaDotRun_IK = JsInv @ Vs_PID

    thetaDotRun_IK = np.clip(thetaDotRun_IK, -wLim, wLim)

    thetaRun_IK = helper.discrete_Integrator(thetaRun_IK, thetaDotRun_IK, SAMPLE_TIME)

    error_theta = thetaRun_IK - thetaRun_Actual

    torqueEffort[0] = PID_Torque_theta_1.update(error_theta[0])
    torqueEffort[1] = PID_Torque_theta_2.update(error_theta[1])
    torqueEffort[2] = PID_Torque_theta_3.update(error_theta[2])
    torqueEffort[3] = PID_Torque_theta_4.update(error_theta[3])
    
    counterFDynamic_Start = time.perf_counter()
    thetaDotDotRun_Actual = dyna.forward_Dynamics(S, MList, GList, thetaRun_Actual, thetaDotRun_Actual, torqueEffort, g, Ftip)
    counterFDynamic_End = time.perf_counter()
    counterFDynamic.append(counterFDynamic_End- counterFDynamic_Start)

    thetaDotRun_Actual = helper.discrete_Integrator(thetaDotRun_Actual, thetaDotDotRun_Actual, SAMPLE_TIME)

    thetaRun_Actual = helper.discrete_Integrator(thetaRun_Actual, thetaDotRun_Actual, SAMPLE_TIME)

    Tsb = kine.PoE_transform(S, M, thetaRun_Actual)

    posInput_log[i] = pos_traj
    posOutput_log[i] = Tsb[:3,3]
    torque_log[i] = torqueEffort  # already clipped
    thetaDotDot_Log[i] = thetaDotDotRun_Actual
    tVec[i] = t

counterLoop_End = time.perf_counter()

if counterFDynamic:
    print("FDynamic timer:", sum(counterFDynamic)/len(counterFDynamic))
print("Elapsed:", counterLoop_End - counterLoop_Start)

posInput_log = np.array(posInput_log)
posOutput_log = np.array(posOutput_log)

# %% 
helper.plot_Continuous(posInput_log, posOutput_log, tVec, extras={'torqueEffort': torque_log, 'thetaDotDot': thetaDotDot_Log}, figsize=(20,14), save_path="plots/output.png")

# run profiler
# import line_profiler as profiler
# lp = profiler.LineProfiler()
# lp.add_function(run_simulation)
# lp.add_function(kine.jacobian_Space)
# lp.add_function(kine.twist_Error)
# lp.add_function(dyna.forward_Dynamics)
# lp.add_function(helper.dls_inverse)

# lp.run('run_simulation()')
# lp.print_stats()

# %% plotting
# import importlib
# importlib.reload(helper)

# %% Scatter plot
# # --- compute reference-change indices (like MATLAB refChange) ---
# diff_x = np.diff(xin)
# diff_y = np.diff(yin)
# # indices where either x or y changed; add first index 0, convert to python 0-based
# changed_idx = np.nonzero((diff_x != 0) | (diff_y != 0))[0] + 1
# ref_change = np.concatenate(([0], changed_idx))   # like [1; ...] in MATLAB but zero-based

# # --- offset distance ---
# x_range = np.max(xin) - np.min(xin)
# y_range = np.max(yin) - np.min(yin)
# xOffset = 0.05 * max(x_range, y_range)

# # --- bookkeeping for duplicate labels at same coord ---
# label_groups = {}   # key -> count

# plt.figure(figsize=(6,6))
# plt.scatter(xin, yin, s=8, color='0.3', alpha=0.6)  # optional trace points

# n_labels = len(ref_change)
# for k_idx, i in enumerate(ref_change, start=1):  # k_idx = 1..n_labels
#     x_i = float(xin[i])
#     y_i = float(yin[i])
#     key = f"{x_i:.6f}_{y_i:.6f}"

#     # update group count
#     cnt = label_groups.get(key, 0) + 1
#     label_groups[key] = cnt

#     # shift: (count-1) * xOffset, MATLAB used (labelGroups(key)-1) * xOffset
#     shift = (cnt - 1) * xOffset

#     # color logic
#     if k_idx == 1:
#         txtColor = 'g'
#     elif k_idx == n_labels:
#         txtColor = 'r'
#     else:
#         txtColor = 'k'

#     plt.text(x_i + shift, y_i,
#              f"{k_idx}",
#              fontsize=9, fontweight='bold', color=txtColor,
#              ha='center', va='center',
#              bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

# plt.xlabel("x position")
# plt.ylabel("y position")
# plt.title("Appearance Order (refChange labels)")
# plt.grid(True)
# plt.axis("equal")
# plt.show()