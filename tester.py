# %%
import numpy as np
import PoE.lieTheory as lie
import PoE.kinematics as kine
import PoE.helper as helper
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

n = [3, -3] # prismatic joint 4 -z axis

q = np.array([[0,     l1,      l1+l2,      l1+l2],
                [0,     0,       0,          0],
                [0,     d1,      d1+d2,      d1+d2]])
w = np.array([[0,     0,       0,       0],
            [0,     0,       0,       0],
            [1,     1,       1,       0]])
# %%
T = np.arange(16).reshape(4,4)

R, p = helper.TransMatTo_Rp(T)
print(R)
print(p)

theta_Pose = np.array([np.deg2rad(0),
                        np.deg2rad(0),
                        np.deg2rad(0), 
                                    0])

thetaRun = theta_Pose.copy()
print(thetaRun)
thetaDotRun = np.zeros(thetaRun.size)
print(thetaDotRun)
a = helper.RpTo_TransMat(R, p)
print(a)
