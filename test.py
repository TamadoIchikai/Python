import numpy as np
import sys
import os

class DualLogger:
    """Log output to both console and file."""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        if self.log and not self.log.closed:
            self.log.flush()
    
    def close(self):
        if self.log and not self.log.closed:
            self.log.close()
    
    def __del__(self):
        """Ensure file is closed when object is destroyed."""
        self.close()


if __name__ == "__main__":
    theta_Pose = np.array([np.deg2rad(0),
                            np.deg2rad(0),
                            np.deg2rad(0),
                                    0], dtype=np.float64)

    rng = np.random.default_rng(2)

    NOISE = {
        "theta_std": np.deg2rad(0.05),
        "theta_dot_std": np.deg2rad(0.10),
        "torque_std": 0.40,
        "ext_wrench_std": np.array([
            0.0, 0.0, 0.0, 0.05, 0.05, 0.05
        ], dtype=np.float64),
        "use_measured_for_jacobian": True
    }

    n_joint = theta_Pose.size
    np.random.seed(3)
    NOISE_thetaRun_Actual = theta_Pose + rng.normal(0.0, NOISE['theta_std'], size=n_joint)
    othernoise = theta_Pose + np.random.normal(0.0, NOISE['theta_std'], size=n_joint)

    # Setup logging AFTER initial prints
    os.makedirs("FuzzyLogicOut", exist_ok=True)
    log_file = "FuzzyLogicOut/optimization_log.txt"
    sys.stdout = DualLogger(log_file)
    sys.stderr = sys.stdout
    print("\n" + "=" * 70)
    print("FUZZY PID RULE TABLE OPTIMIZATION USING GBO")
    print("=" * 70)
    print("Optimizing: dKp rule table (7x7 = 49 parameters)")
    print("Fixed: dKd rule table (symmetric pattern)")
    print("=" * 70 + "\n")
    klasjdfklajsldf
    print(f"NOISE_thetaRun_Actual: {NOISE_thetaRun_Actual}")
    print(NOISE_thetaRun_Actual)
    print(othernoise)
    
    # Restore stdout before closing
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    try:
        sys.stdout.write("✓ Log saved to FuzzyLogicOut/optimization_log.txt\n")
        sys.stdout.close()
    except:
        pass