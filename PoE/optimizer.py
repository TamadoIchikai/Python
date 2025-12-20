import numpy as np
from numba import njit
import scipy.io as sio
import time


@njit(cache=True)
def initialization(nP: int, dim: int, ub: np.ndarray, lb: np.ndarray) -> np.ndarray:
    """Initialize population with random solutions."""
    X = np.zeros((nP, dim))
    for i in range(dim):
        X[:, i] = np.random.rand(nP) * (ub[i] - lb[i]) + lb[i]
    return X


@njit(cache=True)
def GradientSearchRule(ro1: float, Best_X: np.ndarray, Worst_X: np.ndarray,
                       X: np.ndarray, Xr1: np.ndarray, DM: np.ndarray,
                       eps: float, Xm: np.ndarray, Flag: int) -> np.ndarray:
    """Gradient Search Rule implementation."""
    nV = X.shape[0]
    Delta = 2.0 * np.random.rand() * np.abs(Xm - X)
    Step = ((Best_X - Xr1) + Delta) / 2.0
    DelX = np.random.rand(nV) * np.abs(Step)
    
    GSR = np.random.randn() * ro1 * (2 * DelX * X) / (Best_X - Worst_X + eps)
    
    if Flag == 1:
        Xs = X - GSR + DM
    else:
        Xs = Best_X - GSR + DM
    
    yp = np.random.rand() * (0.5 * (Xs + X) + np.random.rand() * DelX)
    yq = np.random.rand() * (0.5 * (Xs + X) - np.random.rand() * DelX)
    GSR = np.random.randn() * ro1 * (2 * DelX * X) / (yp - yq + eps)
    
    return GSR


@njit(cache=True)
def gbo_iteration(it: int, MaxIt: int, nP: int, nV: int, pr: float,
                  X: np.ndarray, Cost: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                  Best_Rules: np.ndarray, Best_Cost: float,
                  Worst_X: np.ndarray, Worst_Cost: float) -> np.ndarray:
    """
    Single iteration of GBO (generates new candidate solutions).
    Returns Xnew_all: array of new candidate solutions for all particles.
    """
    beta = 0.2 + (1.2 - 0.2) * (1 - (it / MaxIt) ** 3) ** 2
    alpha = np.abs(beta * np.sin(3 * np.pi / 2 + np.sin(3 * np.pi / 2 * beta)))
    
    Xnew_all = np.zeros((nP, nV))
    
    for i in range(nP):
        A1 = (np.random.rand(nP) * nP).astype(np.int64)
        r1 = A1[0]
        r2 = A1[1]
        r3 = A1[2]
        r4 = A1[3]
        
        Xm = (X[r1, :] + X[r2, :] + X[r3, :] + X[r4, :]) / 4.0
        
        ro = alpha * (2 * np.random.rand() - 1)
        ro1 = alpha * (2 * np.random.rand() - 1)
        eps = 5e-3 * np.random.rand()
        
        DM = np.random.rand() * ro * (Best_Rules - X[r1, :])
        Flag = 1
        GSR = GradientSearchRule(ro1, Best_Rules, Worst_X, X[i, :], X[r1, :], DM, eps, Xm, Flag)
        DM = np.random.rand() * ro * (Best_Rules - X[r1, :])
        X1 = X[i, :] - GSR + DM
        
        DM = np.random.rand() * ro * (X[r1, :] - X[r2, :])
        Flag = 2
        GSR = GradientSearchRule(ro1, Best_Rules, Worst_X, X[i, :], X[r1, :], DM, eps, Xm, Flag)
        DM = np.random.rand() * ro * (X[r1, :] - X[r2, :])
        X2 = Best_Rules - GSR + DM
        
        Xnew = np.zeros(nV)
        for j in range(nV):
            ro = alpha * (2 * np.random.rand() - 1)
            X3 = X[i, j] - ro * (X2[j] - X1[j])
            ra = np.random.rand()
            rb = np.random.rand()
            Xnew[j] = ra * (rb * X1[j] + (1 - rb) * X2[j]) + (1 - ra) * X3
        
        if np.random.rand() < pr:
            k = int(np.random.rand() * nP)
            f1 = -1 + 2 * np.random.rand()
            f2 = -1 + 2 * np.random.rand()
            ro = alpha * (2 * np.random.rand() - 1)
            
            Xk = np.zeros(nV)
            for j in range(nV):
                Xk[j] = lb[j] + (ub[j] - lb[j]) * np.random.rand()
            
            L1 = 1.0 if np.random.rand() < 0.5 else 0.0
            u1 = L1 * 2 * np.random.rand() + (1 - L1) * 1
            u2 = L1 * np.random.rand() + (1 - L1) * 1
            u3 = L1 * np.random.rand() + (1 - L1) * 1
            
            L2 = 1.0 if np.random.rand() < 0.5 else 0.0
            Xp = (1 - L2) * X[k, :] + L2 * Xk
            
            if u1 < 0.5:
                Xnew = Xnew + f1 * (u1 * Best_Rules - u2 * Xp) + f2 * ro * (u3 * (X2 - X1) + u2 * (X[r1, :] - X[r2, :])) / 2
            else:
                Xnew = Best_Rules + f1 * (u1 * Best_Rules - u2 * Xp) + f2 * ro * (u3 * (X2 - X1) + u2 * (X[r1, :] - X[r2, :])) / 2
        
        for j in range(nV):
            if Xnew[j] > ub[j]:
                Xnew[j] = ub[j]
            elif Xnew[j] < lb[j]:
                Xnew[j] = lb[j]
        
        Xnew_all[i, :] = Xnew
    
    return Xnew_all


def GBO(nP: int, MaxIt: int, lb: np.ndarray, ub: np.ndarray, dim: int,
        cost_function, *cost_args, use_parallel: bool = True, n_workers: int = None) -> tuple:
    """
    Gradient-Based Optimizer (GBO) Algorithm with timing.
    
    Parameters:
    -----------
    nP : int
        Population size
    MaxIt : int
        Maximum iterations
    lb : np.ndarray
        Lower bounds (1D array of size dim)
    ub : np.ndarray
        Upper bounds (1D array of size dim)
    dim : int
        Number of variables
    cost_function : callable
        Cost function to minimize
    *cost_args : 
        Additional arguments for cost_function
    use_parallel : bool
        Whether to use parallel evaluation
    n_workers : int
        Number of parallel workers (default: CPU count)
    
    Returns:
    --------
    Best_Cost, Best_Rules, Convergence_curve
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    
    if n_workers is None:
        n_workers = os.cpu_count()
    
    nV = dim
    pr = 0.5
    Convergence_curve = np.zeros(MaxIt)
    
    lb = np.asarray(lb, dtype=np.float64)
    ub = np.asarray(ub, dtype=np.float64)
    
    # Initialize population
    print("Initializing population...")
    init_start = time.perf_counter()
    X = initialization(nP, nV, ub, lb)
    
    # Calculate initial costs
    Cost = np.zeros(nP)
    
    if use_parallel and nP > 1:
        print(f"Evaluating initial population with {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(cost_function, X[i, :], *cost_args): i for i in range(nP)}
            for future in as_completed(futures):
                i = futures[future]
                Cost[i] = future.result()
    else:
        for i in range(nP):
            Cost[i] = cost_function(X[i, :], *cost_args)
    
    init_end = time.perf_counter()
    print(f"Initialization done in {init_end - init_start:.2f}s")
    
    # Sort to find best and worst
    Ind = np.argsort(Cost)
    Best_Cost = Cost[Ind[0]]
    Best_Rules = X[Ind[0], :].copy()
    Worst_Cost = Cost[Ind[-1]]
    Worst_X = X[Ind[-1], :].copy()
    
    # Allocate history
    Best_Rules_hist = np.zeros((1, nV, MaxIt))
    Best_Rules_hist[0, :, 0] = Best_Rules
    
    print(f"Initial Best Cost: {Best_Cost:.6f}")
    print("-" * 60)
    
    # Main Loop
    for it in range(1, MaxIt + 1):
        iter_start = time.perf_counter()
        
        # Get new candidate solutions
        Xnew_all = gbo_iteration(it, MaxIt, nP, nV, pr, X, Cost, lb, ub,
                                  Best_Rules, Best_Cost, Worst_X, Worst_Cost)
        
        # Evaluate new solutions
        eval_start = time.perf_counter()
        Xnew_Costs = np.zeros(nP)
        
        if use_parallel and nP > 1:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(cost_function, Xnew_all[i, :], *cost_args): i for i in range(nP)}
                for future in as_completed(futures):
                    i = futures[future]
                    Xnew_Costs[i] = future.result()
        else:
            for i in range(nP):
                Xnew_Costs[i] = cost_function(Xnew_all[i, :], *cost_args)
        
        eval_end = time.perf_counter()
        
        # Update solutions
        for i in range(nP):
            if Xnew_Costs[i] < Cost[i]:
                X[i, :] = Xnew_all[i, :]
                Cost[i] = Xnew_Costs[i]
                if Cost[i] < Best_Cost:
                    Best_Rules = X[i, :].copy()
                    Best_Cost = Cost[i]
            
            if Cost[i] > Worst_Cost:
                Worst_X = X[i, :].copy()
                Worst_Cost = Cost[i]
        
        Convergence_curve[it - 1] = Best_Cost
        Best_Rules_hist[0, :, it - 1] = Best_Rules
        
        iter_end = time.perf_counter()
        
        # Display timing info
        print(f"Iter {it:3d}/{MaxIt} | Best: {Best_Cost:.6f} | "
              f"Eval: {eval_end - eval_start:.2f}s | "
              f"Total: {iter_end - iter_start:.2f}s | "
              f"Params: [{Best_Rules[0]:.1f}, {Best_Rules[1]:.1f}, {Best_Rules[2]:.1f}, {Best_Rules[3]:.1f}]")
    
    # Save final results
    try:
        sio.savemat('FuzzyLogicOut/BestRules_hist.mat', {'Best_Rules_hist': Best_Rules_hist})
    except Exception as e:
        print(f"Warning: Failed to save BestRules_hist.mat: {e}")
    
    return Best_Cost, Best_Rules, Convergence_curve