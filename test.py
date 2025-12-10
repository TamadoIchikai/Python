import numpy as np
import PoE.lieTheory as lie
import PoE.kinematics as kine
L1 = 10
L2 = 20

S1 = np.array([0,0,1,0,0,0])
S2 = np.array([0,-1,0,0,0,-L1])
S3 = np.array([1,0,0,0,-L2,0])

M = np.array([[0, 0, 1, L1],
              [0, 1, 0, 0],
              [-1, 0, 0, -L2],
              [0, 0, 0, 1]], dtype=float)

S = np.zeros((6,3))
S[:,0] = S1
S[:,1] = S2
S[:,2] = S3
theta_Pose = np.array([np.deg2rad(45),
                        np.deg2rad(90),
                        np.deg2rad(30)])
joint = 1
se3 = lie.twistTo_se3(S[:,joint]) * theta_Pose[joint]
out =  lie.exp_se3(se3) 

# out = kine.PoE_transform(S, M, theta_Pose)
print(np.array2string(out, formatter={'float_kind':lambda x: f"{x: .4f}"}))
