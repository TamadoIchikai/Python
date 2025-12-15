import numpy as np
from numba import njit
from numba.experimental import jitclass
from numba import types

# Global state for zAxisDownUpReturn (Numba doesn't support function attributes)
_z_state = {
    'lastStartTime': -1.0,
    'started': False,
    'latchedPos': 0.0
}

@njit(cache=True)
def tic_tac_toe_gen(M, T_d, randomList, time, z_lastStartTime, z_started, z_latchedPos):
    """
    Python version of tic_tac_toe_gen.m
    M, Td: 4x4 numpy arrays
    randomList: Nx2 integer array, each row = [ix, iy] with ix, iy in {0,1,2}
    time: float
    z_lastStartTime, z_started, z_latchedPos: state variables for z-axis function
    
    Returns:
        R : 3x3 rotation matrix
        pos : 3x1 vector
        k : selected index (1-based like MATLAB)
        z_lastStartTime, z_started, z_latchedPos: updated state
    """
    # Candidate sets
    x_list = np.array([0.4, 0.5, 0.6])
    y_list = np.array([0.1, 0.0, -0.1])
    
    startTime = 2.0
    stepTime = 5.0
    currentPos = M[2, 3]
    
    z, z_lastStartTime, z_started, z_latchedPos = zAxisDownUpReturn(
        time, startTime, stepTime, currentPos, z_lastStartTime, z_started, z_latchedPos
    )
    
    # Before 2 seconds: hold initial pose
    if time < 2.0:
        R = M[:3, :3].copy()
        pos = M[:3, 3].copy()
        k = 0
        return R, pos, k, z_lastStartTime, z_started, z_latchedPos
    
    # After 2s: orientation comes from Td
    R = T_d[:3, :3].copy()

    # Determine index k
    if 2.0 <= time < 4.0:
        k = 1
    else:
        k = int((time - 2.0) // 3.0) + 1

    # Get indices
    ix = randomList[k - 1, 0]
    iy = randomList[k - 1, 1]

    # Target position
    pos = np.array([x_list[ix], y_list[iy], z])

    return R, pos, k, z_lastStartTime, z_started, z_latchedPos

@njit(cache=True)
def zAxisDownUpReturn(t, startTime, stepTime, currentPos, lastStartTime, started, latchedPos):
    """
    Generates repeated down->up cycles with amplitude latched at startTime.
    :param t: Current time (float)
    :param startTime: Cycle start time (float)
    :param stepTime: Duration of one full cycle (float)
    :param currentPos: Current Z position (float)
    :param lastStartTime, started, latchedPos: state variables
    :return: z (float), updated state variables
    """
    # If startTime changed, reset latch
    if startTime != lastStartTime:
        started = False
        lastStartTime = startTime

    # Before start: hold current position
    if t < startTime:
        return currentPos, lastStartTime, started, latchedPos

    # On first call after start, latch amplitude
    if not started:
        latchedPos = currentPos
        started = True

    tau = (t - startTime) % stepTime
    halfT = stepTime / 2.0

    if tau <= halfT:
        # Down ramp: latchedPos -> 0
        z = latchedPos * (1.0 - tau / halfT)
    else:
        # Up ramp: 0 -> latchedPos
        z = latchedPos * ((tau - halfT) / halfT)
    
    return z, lastStartTime, started, latchedPos


# Wrapper class to manage state (not JIT compiled, used in main)
class TrajectoryState:
    def __init__(self):
        self.z_lastStartTime = -1.0
        self.z_started = False
        self.z_latchedPos = 0.0
    
    def tic_tac_toe(self, M, T_d, randomList, time):
        R, pos, k, self.z_lastStartTime, self.z_started, self.z_latchedPos = tic_tac_toe_gen(
            M, T_d, randomList, time, self.z_lastStartTime, self.z_started, self.z_latchedPos
        )
        return R, pos, k