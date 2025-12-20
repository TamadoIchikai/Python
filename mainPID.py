import numpy as np
import time
import os

import PoE.lieTheory as lie
import PoE.kinematics as kine
import PoE.helper as helper
import PoE.dynamics as dyna
import PoE.trajectoryGen as trajGen
import PoE.controller as controller
from PoE.optimizer import GBO


def simulation_cost_ITAE(pid_params: np.ndarray, sim_config: dict) -> float:
    """
    Cost function for PID optimization using improved ITAE.
    
    Improvements:
    1. Time-offset ITAE: Only count after trajectory starts (t >= 2s)
    2. Step-change tracking: Weight errors after each trajectory step change
    3. ISE component: Better transient response sensitivity
    
    Final cost = ISE_total + weight * ITAE_total
    
    Parameters:
    -----------
    pid_params : np.ndarray
        Array of PID parameters: [Kp1, Kd1, Kp2, Kd2]
    sim_config : dict
        Dictionary containing simulation configuration
    
    Returns:
    --------
    cost : float
        Combined cost value (lower is better)
    """
    # Unpack PID parameters
    Kp1, Kd1, Kp2, Kd2 = pid_params[0], pid_params[1], pid_params[2], pid_params[3]
    
    # Unpack simulation config
    S = sim_config['S']
    M = sim_config['M']
    MList = sim_config['MList']
    GList = sim_config['GList']
    g = sim_config['g']
    Ftip = sim_config['Ftip']
    theta_Pose = sim_config['theta_Pose']
    theta_D = sim_config['theta_D']
    tauLim = sim_config['tauLim']
    tauInit = sim_config['tauInit']
    wLim = sim_config['wLim']
    SAMPLE_TIME = sim_config['SAMPLE_TIME']
    n_steps = sim_config['n_steps']
    randomList = sim_config['randomList']
    
    # Initialize states
    Tsb = kine.PoE_transform(S, M, theta_Pose)
    T_d = kine.PoE_transform(S, M, theta_D)
    
    thetaRun_IK = theta_Pose.copy()
    thetaDotRun_IK = np.zeros_like(thetaRun_IK)
    
    thetaRun_Actual = theta_Pose.copy()
    thetaDotRun_Actual = np.zeros_like(thetaRun_Actual)
    thetaDotDotRun_Actual = np.zeros_like(thetaDotRun_Actual)
    
    # Create PID controllers
    PID_IK_WzXY = controller.PID_Discrete(Kp=80.0, Ki=0.0, Kd=15.0, Ts=SAMPLE_TIME)
    PID_IK_Z = controller.PID_Discrete(Kp=45.0, Ki=0.0, Kd=5.0, Ts=SAMPLE_TIME)
    
    # Torque PIDs with optimized Kp and Kd
    PID_Torque_theta_1 = controller.PID_Discrete(
        Kp=Kp1, Ki=0.0, Kd=Kd1, Ts=SAMPLE_TIME,
        outputLimit=(-tauLim[0], tauLim[0]), initial_integral=tauInit[0]
    )
    PID_Torque_theta_2 = controller.PID_Discrete(
        Kp=Kp2, Ki=0.0, Kd=Kd2, Ts=SAMPLE_TIME,
        outputLimit=(-tauLim[1], tauLim[1]), initial_integral=tauInit[1]
    )
    PID_Torque_theta_3 = controller.PID_Discrete(
        Kp=60.0, Ki=0.0, Kd=5.0, Ts=SAMPLE_TIME,
        outputLimit=(-tauLim[2], tauLim[2]), initial_integral=tauInit[2]
    )
    PID_Torque_theta_4 = controller.PID_Discrete(
        Kp=40.0, Ki=20.0, Kd=30.0, Ts=SAMPLE_TIME,
        outputLimit=(-tauLim[3], tauLim[3]), initial_integral=tauInit[3]
    )
    
    Vs_PID = np.zeros(6, dtype=np.float64)
    torqueEffort = np.zeros_like(thetaRun_IK)
    
    traj_state = trajGen.TrajectoryState()
    
    # Cost accumulators
    ITAE_x = 0.0
    ITAE_y = 0.0
    ISE_x = 0.0
    ISE_y = 0.0
    
    # Step-change tracking
    prev_k = -1
    step_start_time = 2.0  # Trajectory starts at t=2s
    step_ISE_x = 0.0
    step_ISE_y = 0.0
    step_cost = 0.0
    
    # Trajectory start time
    TRAJ_START = 2.0
    
    for i in range(n_steps):
        t = i * SAMPLE_TIME
        
        R_traj, pos_traj, k = traj_state.tic_tac_toe(M, T_d, randomList, t)
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
            return 1e10  # Return high cost for invalid Jacobian
        
        JsInv = helper.dls_inverse(Js, 1e-3)
        thetaDotRun_IK = JsInv @ Vs_PID
        
        for j in range(thetaDotRun_IK.size):
            if thetaDotRun_IK[j] < -wLim[j]:
                thetaDotRun_IK[j] = -wLim[j]
            elif thetaDotRun_IK[j] > wLim[j]:
                thetaDotRun_IK[j] = wLim[j]
        
        thetaRun_IK = helper.discrete_Integrator(thetaRun_IK, thetaDotRun_IK, SAMPLE_TIME)
        
        error_theta = thetaRun_IK - thetaRun_Actual
        
        torqueEffort[0] = PID_Torque_theta_1.update(error_theta[0])
        torqueEffort[1] = PID_Torque_theta_2.update(error_theta[1])
        torqueEffort[2] = PID_Torque_theta_3.update(error_theta[2])
        torqueEffort[3] = PID_Torque_theta_4.update(error_theta[3])
        
        thetaDotDotRun_Actual = dyna.forward_Dynamics(
            S, MList, GList, thetaRun_Actual, thetaDotRun_Actual, torqueEffort, g, Ftip
        )
        
        # Check for instability
        if not np.all(np.isfinite(thetaDotDotRun_Actual)):
            return 1e10  # Unstable system
        
        thetaDotRun_Actual = helper.discrete_Integrator(thetaDotRun_Actual, thetaDotDotRun_Actual, SAMPLE_TIME)
        thetaRun_Actual = helper.discrete_Integrator(thetaRun_Actual, thetaDotRun_Actual, SAMPLE_TIME)
        
        Tsb = kine.PoE_transform(S, M, thetaRun_Actual)
        
        # Compute position errors
        error_x = pos_traj[0] - Tsb[0, 3]
        error_y = pos_traj[1] - Tsb[1, 3]
        
        # Only compute cost after trajectory starts
        if t >= TRAJ_START:
            # Detect step change (new trajectory point)
            if k != prev_k and prev_k >= 0:
                # Add accumulated step error to total
                step_cost += step_ISE_x + step_ISE_y
                step_ISE_x = 0.0
                step_ISE_y = 0.0
                step_start_time = t
            
            # Time since trajectory started (for ITAE)
            t_offset = t - TRAJ_START
            
            # Time since last step change (for step-weighted error)
            t_since_step = t - step_start_time
            
            # ITAE with time offset
            ITAE_x += t_offset * np.abs(error_x) * SAMPLE_TIME
            ITAE_y += t_offset * np.abs(error_y) * SAMPLE_TIME
            
            # ISE for transient response
            ISE_x += error_x**2 * SAMPLE_TIME
            ISE_y += error_y**2 * SAMPLE_TIME
            
            # Step-weighted ISE (emphasizes settling after each step change)
            step_ISE_x += t_since_step * (error_x**2) * SAMPLE_TIME
            step_ISE_y += t_since_step * (error_y**2) * SAMPLE_TIME
        
        prev_k = k
    
    # Add final step error
    step_cost += step_ISE_x + step_ISE_y
    
    # Combined cost function:
    # - ISE: Good for transient response
    # - ITAE: Good for settling time
    # - Step cost: Emphasizes settling after each trajectory change
    w_ise = 1.0       # Weight for ISE
    w_itae = 0.1      # Weight for ITAE (smaller since values are larger)
    w_step = 0.5      # Weight for step-change settling
    
    cost = w_ise * (ISE_x + ISE_y) + w_itae * (ITAE_x + ITAE_y) + w_step * step_cost
    
    return cost


def setup_simulation_config():
    """Setup simulation configuration dictionary."""
    # Dimension in m
    l1 = .35
    l2 = .45
    d1 = .284
    d2 = .1
    d3 = .334
    
    a1 = .035
    b1 = .095
    a2 = .02
    b2 = .08
    r3 = 0.033
    c1 = .05
    c2 = .08
    
    theta_Pose = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0], dtype=np.float64)
    theta_D = np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0], dtype=np.float64)
    
    n = (3, -3)
    
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
    n_prismatic_joint, n_prismatic_axis = n
    S = lie.compute_ScrewMat(w, q, n_prismatic_joint, n_prismatic_axis)
    
    # Dynamics setup
    m1, m2, m3, m4 = 3.0, 2.0, 1.5, 1.5
    g = np.array([0, 0, -9.8], dtype=np.float64)
    tauInit = np.array([0, 0, 0, m4 * g[2]], dtype=np.float64)
    Ftip = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
    
    MList = np.zeros((4, 4, n_joint+1), dtype=np.float64)
    MList[:,:,0] = np.eye(4)
    MList[:,:,1] = np.eye(4)
    MList[0,3,1] = l1
    MList[2,3,1] = d1
    MList[:,:,2] = np.eye(4)
    MList[0,3,2] = l2
    MList[2,3,2] = d2
    MList[:,:,3] = np.eye(4)
    MList[:,:,4] = np.eye(4)
    
    I1CoM = (m1/12) * np.diag([a1**2 + b1**2, l1**2 + a1**2, b1**2 + l1**2])
    I2CoM = (m2/12) * np.diag([a2**2 + b2**2, l2**2 + a2**2, b2**2 + l2**2])
    I3CoM = m3 * np.diag([(1/12)*(d3**2 + 3*r3**2), (1/12)*(d3**2 + 3*r3**2), (1/2)*r3**2])
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
    TIME_STOP = 15.0  # Cover multiple trajectory steps (each step is ~3s after t=2s)
    n_steps = int(TIME_STOP / SAMPLE_TIME)
    
    wLim = np.array([1, 2, 2, .2], dtype=np.float64)
    tauLim = np.array([20, 15, 5, 20], dtype=np.float64)
    
    # Random trajectory - fixed seed for reproducibility
    N = 100
    all_points = np.array([[x, y] for x in range(3) for y in range(3)], dtype=np.int64)
    randomList = []
    np.random.seed(42)
    while len(randomList) < N:
        shuffled = all_points[np.random.permutation(9)]
        randomList.append(shuffled)
    randomList = np.vstack(randomList)[:N].astype(np.int64)
    
    return {
        'S': S,
        'M': M,
        'MList': MList,
        'GList': GList,
        'g': g,
        'Ftip': Ftip,
        'theta_Pose': theta_Pose,
        'theta_D': theta_D,
        'tauLim': tauLim,
        'tauInit': tauInit,
        'wLim': wLim,
        'SAMPLE_TIME': SAMPLE_TIME,
        'n_steps': n_steps,
        'randomList': randomList
    }


def test_single_evaluation(pid_params, sim_config):
    """Test a single cost evaluation and print diagnostic info."""
    print("\n" + "="*60)
    print("DIAGNOSTIC: Single Cost Evaluation")
    print("="*60)
    print(f"PID Params: Kp1={pid_params[0]:.2f}, Kd1={pid_params[1]:.2f}, "
          f"Kp2={pid_params[2]:.2f}, Kd2={pid_params[3]:.2f}")
    
    start = time.perf_counter()
    cost = simulation_cost_ITAE(pid_params, sim_config)
    elapsed = time.perf_counter() - start
    
    print(f"Cost: {cost:.6f}")
    print(f"Evaluation time: {elapsed:.2f}s")
    print("="*60 + "\n")
    return cost


if __name__ == "__main__":
    # Setup simulation config
    sim_config = setup_simulation_config()
    
    # Test with default parameters first
    default_params = np.array([100.0, 30.0, 80.0, 40.0], dtype=np.float64)
    print("Testing default PID parameters...")
    default_cost = test_single_evaluation(default_params, sim_config)
    
    # Test with different parameters to verify cost sensitivity
    test_params = np.array([150.0, 50.0, 120.0, 60.0], dtype=np.float64)
    print("Testing alternative PID parameters...")
    test_cost = test_single_evaluation(test_params, sim_config)
    
    print(f"Cost difference: {abs(default_cost - test_cost):.6f} "
          f"({abs(default_cost - test_cost) / default_cost * 100:.2f}%)")
    
    # GBO parameters
    nP = 16         # Population size
    MaxIt = 500     # Reduced iterations - should converge faster with better cost function
    dim = 4         # Number of variables: [Kp1, Kd1, Kp2, Kd2]
    
    # Tighter bounds based on typical PID values for robotic systems
    lb = np.array([50.0, 10.0, 50.0, 10.0], dtype=np.float64)
    ub = np.array([1000.0, 300.0, 1000.0, 300.0], dtype=np.float64)
    
    print("="*60)
    print("PID Optimization with GBO - Improved Cost Function")
    print("="*60)
    print(f"Population: {nP}, Iterations: {MaxIt}")
    print(f"Optimizing: Kp1, Kd1, Kp2, Kd2")
    print(f"Bounds: lb={lb}, ub={ub}")
    print(f"Cost function: ISE + 0.1*ITAE + 0.5*StepCost")
    print(f"Simulation time: {sim_config['n_steps'] * sim_config['SAMPLE_TIME']:.1f}s")
    print(f"CPU cores available: {os.cpu_count()}")
    print("="*60)
    
    start_time = time.perf_counter()
    
    Best_Cost, Best_Params, Convergence_curve = GBO(
        nP, MaxIt, lb, ub, dim, 
        simulation_cost_ITAE, sim_config,
        use_parallel=True,
        n_workers=min(nP, os.cpu_count())
    )
    
    end_time = time.perf_counter()
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Total elapsed time: {end_time - start_time:.2f} seconds")
    print(f"Best Cost: {Best_Cost:.6f}")
    print(f"Improvement from default: {(default_cost - Best_Cost) / default_cost * 100:.2f}%")
    print(f"\nOptimized PID Parameters:")
    print(f"  PID_Torque_theta_1: Kp={Best_Params[0]:.4f}, Kd={Best_Params[1]:.4f}")
    print(f"  PID_Torque_theta_2: Kp={Best_Params[2]:.4f}, Kd={Best_Params[3]:.4f}")
    print("="*60)
    
    # Save results
    np.savez('pid_optimization_results.npz',
             Best_Cost=Best_Cost,
             Best_Params=Best_Params,
             Convergence_curve=Convergence_curve,
             default_cost=default_cost,
             default_params=default_params)
    
    # Plot convergence
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Convergence plot
        axes[0].plot(range(1, MaxIt + 1), Convergence_curve, 'b-', linewidth=2)
        axes[0].axhline(y=default_cost, color='r', linestyle='--', label=f'Default: {default_cost:.4f}')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Best Cost')
        axes[0].set_title('GBO Convergence - PID Optimization')
        axes[0].grid(True)
        axes[0].legend()
        
        # Convergence plot (log scale for better visibility)
        axes[1].semilogy(range(1, MaxIt + 1), Convergence_curve, 'b-', linewidth=2)
        axes[1].axhline(y=default_cost, color='r', linestyle='--', label=f'Default: {default_cost:.4f}')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Best Cost (log scale)')
        axes[1].set_title('GBO Convergence (Log Scale)')
        axes[1].grid(True)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig('convergence_plot.png', dpi=150, bbox_inches='tight')
        print("\nConvergence plot saved to 'convergence_plot.png'")
        
    except Exception as e:
        print(f"Could not save plot: {e}")
    
    # Generate code snippet for main.py
    print("\n" + "="*60)
    print("Copy these lines to your main.py:")
    print("="*60)
    print(f"""
# Optimized PID parameters from GBO
PID_Torque_theta_1 = controller.PID_Discrete(
    Kp={Best_Params[0]:.4f}, Ki=0.0, Kd={Best_Params[1]:.4f}, 
    Ts=SAMPLE_TIME, outputLimit=(-tauLim[0], tauLim[0]), initial_integral=tauInit[0]
)
PID_Torque_theta_2 = controller.PID_Discrete(
    Kp={Best_Params[2]:.4f}, Ki=0.0, Kd={Best_Params[3]:.4f}, 
    Ts=SAMPLE_TIME, outputLimit=(-tauLim[1], tauLim[1]), initial_integral=tauInit[1]
)
""")