import numpy as np

theta_Pose = np.array([np.deg2rad(0),
                        np.deg2rad(0),
                        np.deg2rad(0),
                                0], dtype=np.float64)

rng = np.random.default_rng(2)

NOISE = {
    "theta_std": np.deg2rad(0.05),        # joint angle sensor noise [rad]
    "theta_dot_std": np.deg2rad(0.10),    # joint velocity sensor noise [rad/s]
    "torque_std": 0.40,                   # actuator torque disturbance [Nm]
    "ext_wrench_std": np.array([          # external wrench at the tool [Nx,Ny,Nz,Tx,Ty,Tz]
        0.0, 0.0, 0.0, 0.05, 0.05, 0.05
    ], dtype=np.float64),
    "use_measured_for_jacobian": True     # use noisy joints for Js as well
}

n_joint = theta_Pose.size

NOISE_thetaRun_Actual = theta_Pose + rng.normal(0.0, NOISE['theta_std'], size = n_joint)

print(rng.normal(0.0, NOISE["torque_std"], size=n_joint))

print(NOISE_thetaRun_Actual)
