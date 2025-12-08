import numpy as np

def tic_tac_toe_gen(M, T_d, randomList, time):
    """
    Python version of tic_tac_toe_gen.m
    M, Td: 4x4 numpy arrays
    random_List: Nx2 integer array, each row = [ix, iy] with ix, iy in {0,1,2}
    time: float
    
    Returns:
        R : 3x3 rotation matrix
        pos : 3x1 vector
        k : selected index (1-based like MATLAB)
    """

    # Candidate sets
    x_list = np.array([0.4, 0.5, 0.6])
    y_list = np.array([0.1, 0.0, -0.1])
    z = zAxisDownUpReturn(time, 2, 5, M[2,3])
    # --- Before 2 seconds: hold initial pose ---
    if time < 2:
        R = M[0:3, 0:3]
        pos = M[0:3, 3]
        k = 0   # same as MATLAB: function exits early
        return R, pos, k
    
    # --- After 2s: orientation comes from Td ---
    R = T_d[0:3, 0:3]

    # Determine index k (MATLAB 1-based logic)
    if 2 <= time < 4:
        k = 1
    else:
        k = int((time - 2) // 3) + 1   # MATLAB floor((time - 2)/3)+1

    # Get indices (MATLAB indexing → Python indexing)
    # MATLAB: idx_x = random_List(k,1)
    # Python: random_List[k-1, 0]
    ix = randomList[k-1, 0]
    iy = randomList[k-1, 1]

    # Target position
    pos = np.array([
        x_list[ix],
        y_list[iy],
        z
    ])

    return R, pos, k

def zAxisDownUpReturn(t, startTime, stepTime, currentPos):
    """
    Generates repeated down->up cycles with amplitude latched at startTime.
    :param t: Current time (float)
    :param startTime: Cycle start time (float)
    :param stepTime: Duration of one full cycle (float)
    :param currentPos: Current Z position (float)
    :return: z (float)
    """
    # Persistent/static variables (attached as function attributes)
    if not hasattr(zAxisDownUpReturn, "lastStartTime"):
        zAxisDownUpReturn.lastStartTime = None
        zAxisDownUpReturn.started = False
        zAxisDownUpReturn.latchedPos = 0.0

    # If startTime changed, reset latch
    if startTime != zAxisDownUpReturn.lastStartTime:
        zAxisDownUpReturn.started = False
        zAxisDownUpReturn.lastStartTime = startTime

    # Before start: hold current position
    if t < startTime:
        return currentPos

    # On first call after start, latch amplitude
    if not zAxisDownUpReturn.started:
        zAxisDownUpReturn.latchedPos = currentPos
        zAxisDownUpReturn.started = True

    tau = (t - startTime) % stepTime
    halfT = stepTime / 2.0

    if tau <= halfT:
        # Down ramp: latchedPos -> 0
        z = zAxisDownUpReturn.latchedPos * (1 - tau / halfT)
    else:
        # Up ramp: 0 -> latchedPos
        z = zAxisDownUpReturn.latchedPos * ((tau - halfT) / halfT)
    return z