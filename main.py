import PoE.kinematics as kine
import PoE.helper as helper
import PoE.trajectoryGen as trajGen
import PoE.lieTheory as lie

import numpy as np

# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

def setup_simulation_config():
    """Setup robot and simulation configuration."""
    # Robot dimensions (meters)
    l1, l2 = 0.35, 0.45
    d1, d2, d3 = 0.284, 0.015, 0.334
    a1, b1, a2, b2 = 0.035, 0.095, 0.02, 0.08
    r3, c1, c2 = 0.033, 0.05, 0.02

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
        "used": False,
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
        'w_ISE': 0.6, 'w_ITAE': 0.6, 'w_IAE': 0.8, 'w_U': 1e-4,
        'used_Noise': NOISE["used"], 'NOISE_Gen_thetaRun_Actual': NOISE_Gen_thetaRun_Actual, 'NOISE_Gen_thetaRunDot_Actual':NOISE_Gen_thetaRunDot_Actual
    }
