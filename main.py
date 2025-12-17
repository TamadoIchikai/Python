import PoE.kinematics as kine
import PoE.helper as helper
import PoE.dynamics as dyna
import PoE.trajectoryGen as trajGen
import PoE.controller as controller
import PoE.lieTheory as lie

import numpy as np
import time
from numba import prange
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
                                0], dtype=np.float64)

theta_D= np.array([np.deg2rad(0),
                    np.deg2rad(0),
                    np.deg2rad(0),
                                0.4], dtype=np.float64)
n = (3, -3) # prismatic joint 4 -z axis

q = np.array([[0,     l1,      l1+l2,      l1+l2],
                [0,     0,       0,          0],
                [0,     d1,      d1+d2,      d1+d2]], dtype=np.float64)

w = np.array([[0,     0,       0,       0],
                [0,     0,       0,       0],
                [1,     1,       1,       0]], dtype=np.float64)

M = np.array([[1,     0,       0,       l1+l2],
                [0,     1,       0,       0],
                [0,     0,       1,       d1+d2],
                [0,     0,       0,       1]], dtype=np.float64)

n_joint = theta_Pose.size
# Unpack the tuple
n_prismatic_joint, n_prismatic_axis = n  # (3, -3)
S = lie.compute_ScrewMat(w, q, n_prismatic_joint, n_prismatic_axis)

print(kine.PoE_transform(S, M, theta_Pose))
# helper.validate_joint_inputs(q, w, M, S, theta_Pose)
# %% Dynamics
Ftip = np.array([0,0,0,0,0,0], dtype=np.float64)
m1, m2, m3, m4 = 3.0, 2.0, 1.5, 1.5 #kg

g = np.array([0,0,-9.8], dtype=np.float64)
tauInit = np.array([0,0,0,m4 * g[2]], dtype=np.float64)

MList = np.zeros((4, 4, n_joint+1), dtype=np.float64)
MList[:,:,0] = np.eye(4, dtype=np.float64)
MList[:,:,1] = np.eye(4, dtype=np.float64)
MList[0,3,1] = l1
MList[1,3,1] = 0.0
MList[2,3,1] = d1
MList[:,:,2] = np.eye(4, dtype=np.float64)
MList[0,3,2] = l2
MList[1,3,2] = 0.0
MList[2,3,2] = d2
MList[:,:,3] = np.eye(4, dtype=np.float64)
MList[:,:,4] = np.eye(4, dtype=np.float64)


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

# %% Some inverse kinematic
SAMPLE_TIME = 0.001
STOP_TIME = 30.0


# %% Looping time
Tsb = kine.PoE_transform(S, M, theta_Pose)
T_d = kine.PoE_transform(S, M, theta_D)

thetaRun_IK = theta_Pose.copy()
thetaDotRun_IK = np.zeros_like(thetaRun_IK)

thetaRun_Actual = theta_Pose.copy()
thetaDotRun_Actual = np.zeros_like(thetaRun_Actual)
thetaDotDotRun_Actual = np.zeros_like(thetaDotRun_Actual)

wLim = np.array([1, 2, 2, .2], dtype=np.float64)
tauLim = np.array([20, 15, 5, 20], dtype=np.float64)

# PID controllers (still class-based, called from Python)
PID_IK_WzXY = controller.PID_Discrete(Kp=80.0, Ki=0.0, Kd=15.0, Ts=SAMPLE_TIME)
PID_IK_Z = controller.PID_Discrete(Kp=45.0, Ki=0.0, Kd=5.0, Ts=SAMPLE_TIME)

PID_Torque_theta_1 = controller.PID_Discrete(Kp=100.0, Ki=0.0, Kd=30.0, Ts=SAMPLE_TIME, outputLimit=(-tauLim[0], tauLim[0]), initial_integral=tauInit[0])
PID_Torque_theta_2 = controller.PID_Discrete(Kp=80.0, Ki=0.0, Kd=40.0, Ts=SAMPLE_TIME, outputLimit=(-tauLim[1], tauLim[1]), initial_integral=tauInit[1])
PID_Torque_theta_3 = controller.PID_Discrete(Kp=60.0, Ki=0.0, Kd=5.0, Ts=SAMPLE_TIME, outputLimit=(-tauLim[2], tauLim[2]), initial_integral=tauInit[2])
PID_Torque_theta_4 = controller.PID_Discrete(Kp=40.0, Ki=20.0, Kd=30.0, Ts=SAMPLE_TIME, outputLimit=(-tauLim[3], tauLim[3]), initial_integral=tauInit[3])

fuzzy_Theta_1 = controller.FuzzyLookupController("FuzzyLogicOut/FuzzySugeno_e_Theta_1.npz")
KpBase_theta_1, KdBase_theta_1  = PID_Torque_theta_1.Kp, PID_Torque_theta_1.Kd

fuzzy_Theta_2 = controller.FuzzyLookupController("FuzzyLogicOut/FuzzySugeno_e_Theta_2.npz")
KpBase_theta_2, KdBase_theta_2  = PID_Torque_theta_2.Kp, PID_Torque_theta_2.Kd

Vs_PID = np.zeros(6, dtype=np.float64)
torqueEffort = np.zeros_like(thetaRun_IK)

n_steps = int(STOP_TIME / SAMPLE_TIME)
posInput_log = np.zeros((n_steps, 3), dtype=np.float64)
posOutput_log = np.zeros((n_steps, 3), dtype=np.float64)
torque_log = np.zeros((n_steps, thetaRun_IK.size), dtype=np.float64)
thetaDotDot_Log = np.zeros((n_steps, thetaRun_IK.size), dtype=np.float64)
tVec = []

# Create trajectory manager
traj = trajGen.TrajectoryGen(
    M=M,
    T_d=T_d,
    startTime=2,
    stepTimeXY=2,
    stepTimeZ=5,
    stopTime=STOP_TIME,
    seed=3
)

counterFDynamic = []
counterLoop_Start = time.perf_counter()

for i in range(n_steps):
    t = i * SAMPLE_TIME

    R_traj, pos_traj = traj.generate(t)

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
        print("Bad Jacobian (NaN/inf) — aborting step. thetaRun:", thetaRun_IK)
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
    errorDot_theta = thetaRun_IK - thetaDotRun_Actual

    dKp_theta_1, dKd_theta_1 = fuzzy_Theta_1.get_gains(error_theta[0], errorDot_theta[0])
    dKp_theta_2, dKd_theta_2 = fuzzy_Theta_2.get_gains(error_theta[1], errorDot_theta[1])

    PID_Torque_theta_1.Kp, PID_Torque_theta_1.Kd = KpBase_theta_1 * dKp_theta_1, KdBase_theta_1  * dKd_theta_1
    PID_Torque_theta_2.Kp, PID_Torque_theta_2.Kd = KpBase_theta_2 * dKp_theta_2, KdBase_theta_2  * dKd_theta_2

    torqueEffort[0] = PID_Torque_theta_1.update(error_theta[0])
    torqueEffort[1] = PID_Torque_theta_2.update(error_theta[1])
    torqueEffort[2] = PID_Torque_theta_3.update(error_theta[2])
    torqueEffort[3] = PID_Torque_theta_4.update(error_theta[3])

    counterFDynamic_Start = time.perf_counter()
    thetaDotDotRun_Actual = dyna.forward_Dynamics(S, MList, GList, thetaRun_Actual, thetaDotRun_Actual, torqueEffort, g, Ftip)
    counterFDynamic_End = time.perf_counter()
    counterFDynamic.append(counterFDynamic_End - counterFDynamic_Start)

    thetaDotRun_Actual = helper.discrete_Integrator(thetaDotRun_Actual, thetaDotDotRun_Actual, SAMPLE_TIME)

    thetaRun_Actual = helper.discrete_Integrator(thetaRun_Actual, thetaDotRun_Actual, SAMPLE_TIME)

    Tsb = kine.PoE_transform(S, M, thetaRun_Actual)
    
    if (i%1000) ==1:
        print("Ite", i)

    posInput_log[i, 0] = pos_traj[0]
    posInput_log[i, 1] = pos_traj[1]
    posInput_log[i, 2] = pos_traj[2]
    posOutput_log[i, 0] = Tsb[0, 3]
    posOutput_log[i, 1] = Tsb[1, 3]
    posOutput_log[i, 2] = Tsb[2, 3]
    torque_log[i, 0] = torqueEffort[0]
    torque_log[i, 1] = torqueEffort[1]
    torque_log[i, 2] = torqueEffort[2]
    torque_log[i, 3] = torqueEffort[3]
    thetaDotDot_Log[i, 0] = thetaDotDotRun_Actual[0]
    thetaDotDot_Log[i, 1] = thetaDotDotRun_Actual[1]
    thetaDotDot_Log[i, 2] = thetaDotDotRun_Actual[2]
    thetaDotDot_Log[i, 3] = thetaDotDotRun_Actual[3]
    tVec.append(t)

counterLoop_End = time.perf_counter()
# After the simulation loop
print("\nFirst 10 trajectory points:")
for i in range(0, min(10000, n_steps), 1000):
    t =tVec[i]
    print(f"t={t:.3f}s: ref=({posInput_log[i,0]:.3f}, {posInput_log[i,1]:.3f}), "
          f"actual=({posOutput_log[i,0]:.3f}, {posOutput_log[i,1]:.3f})")
if counterFDynamic:
    print("FDynamic timer:", sum(counterFDynamic)/len(counterFDynamic))
print("Elapsed:", counterLoop_End - counterLoop_Start)

# %%
helper.plot_Continuous(posInput_log, posOutput_log, tVec, extras={'torqueEffort': torque_log, 'thetaDotDot': thetaDotDot_Log}, figsize=(20,14))