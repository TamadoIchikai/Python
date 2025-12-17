import numpy as np
import matplotlib.pyplot as plt

class TrajectoryGen:
    def __init__(self, M, T_d, startTime, stepTimeXY, stepTimeZ, stopTime, seed=1):
        # 1. Store configuration variables in 'self'
        self.M = M
        self.T_d = T_d
        self.startTime = startTime
        self.stepTimeXY = stepTimeXY
        self.stepTimeZ = stepTimeZ
        self.stopTime = stopTime
        
        # 2. Initialize State
        self.seed = seed
        self.random_pairs = random_Index_Pair(n=1000, seed=self.seed)

    def generate(self, currentTime):
        """
        Calculates the pose based on the stored parameters and the specific currentTime.
        """
        if currentTime < self.startTime:
            R = self.M[:3, :3]
            pos = self.M[:3, 3] 
            return R, pos
        else:
            R = self.T_d[:3, :3]
            
            z_init = self.M[2, 3] 
            z_reach = self.T_d[2, 3]
            
            x, y = tic_tac_toe_gen(self.random_pairs, self.startTime, self.stepTimeXY, currentTime)
            z = zAxisUpDown(z_init, z_reach, self.startTime, self.stepTimeZ, currentTime)
            
            pos = np.array([x, y, z])
            return R, pos


def random_Index_Pair(n=1000, seed=1):
    np.random.seed(seed)
    
    unique_pairs = np.array(np.meshgrid([0, 1, 2], [0, 1, 2])).T.reshape(-1, 2)
    
    n_blocks = (n + 8) // 9
    
    blocks = np.tile(unique_pairs, (n_blocks, 1, 1))
    
    noise = np.random.rand(n_blocks, 9)
    permutations = np.argsort(noise, axis=1)
    
    row_idx = np.arange(n_blocks)[:, None]
    shuffled = blocks[row_idx, permutations]
    
    result = shuffled.reshape(-1, 2)[:n]
    
    return result

def tic_tac_toe_gen(randomPairIndex, startTime, stepTime, currentTime):
    xList = np.array([0.4, 0.5, 0.6])
    yList = np.array([-0.1, 0.0, 0.1])
    
    elapsed = currentTime - startTime
    
    current_idx = int(max(0, elapsed) // stepTime)
    
    if current_idx >= len(randomPairIndex):
        current_idx = len(randomPairIndex) - 1
        
    indices = randomPairIndex[current_idx]
    return xList[indices[0]], yList[indices[1]]

def zAxisUpDown(zInit, zReach, startTime, stepTime, currentTime):
    elapsed = currentTime - startTime
    
    time_in_cycle = elapsed % stepTime
    half_step = stepTime / 2.0

    if time_in_cycle <= half_step:
        ratio = time_in_cycle / half_step
        z = zInit + (zReach - zInit) * ratio
    else:
        time_in_part = time_in_cycle - half_step
        ratio = time_in_part / half_step
        z = zReach - (zReach - zInit) * ratio
        
    return z