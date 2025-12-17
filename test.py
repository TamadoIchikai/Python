import numpy as np
import matplotlib.pyplot as plt
import PoE.trajectoryGen as trajGen

# Robot configuration
l1, l2, d1, d2 = 0.35, 0.45, 0.284, 0.1
M = np.array([[0, 0, 1, l1+l2],
              [0, 1, 0, 0],
              [-1, 0, 0, d1+d2],
              [0, 0, 0, 1]], dtype=np.float64)
T_d = np.eye(4, dtype=np.float64)

# Simulation parameters
SAMPLE_TIME = 0.001
TIME_STOP = 60.0
n_steps = int(TIME_STOP / SAMPLE_TIME)

# Create trajectory manager
traj = trajGen(
)

# Print the unique 9-pair sequence
traj.print_cycle_sequence(cycle=0)
traj.print_cycle_sequence(cycle=1)

print("\nRunning simulation...")
for i in range(n_steps):
    t = i * SAMPLE_TIME
    R, pos, k = traj.get_trajectory(M, T_d, t)

# Get history
times, positions = traj.get_history()

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# X position
axes[0, 0].plot(times, positions[:, 0], 'b-', linewidth=1.5)
axes[0, 0].axhline(y=0.4, color='g', linestyle='--', alpha=0.5, label='x_list')
axes[0, 0].axhline(y=0.5, color='g', linestyle='--', alpha=0.5)
axes[0, 0].axhline(y=0.6, color='g', linestyle='--', alpha=0.5)
axes[0, 0].axvline(x=traj.startTime, color='r', linestyle=':', label='startTime')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('X Position (m)')
axes[0, 0].set_title('X Position vs Time')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Y position
axes[0, 1].plot(times, positions[:, 1], 'r-', linewidth=1.5)
axes[0, 1].axhline(y=-0.1, color='g', linestyle='--', alpha=0.5, label='y_list')
axes[0, 1].axhline(y=0.0, color='g', linestyle='--', alpha=0.5)
axes[0, 1].axhline(y=0.1, color='g', linestyle='--', alpha=0.5)
axes[0, 1].axvline(x=traj.startTime, color='r', linestyle=':', label='startTime')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Y Position (m)')
axes[0, 1].set_title('Y Position vs Time')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Z position
axes[1, 0].plot(times, positions[:, 2], 'g-', linewidth=1.5)
axes[1, 0].axhline(y=traj.zReach, color='m', linestyle='--', alpha=0.5, label='zReach')
axes[1, 0].axvline(x=traj.startTime, color='r', linestyle=':', label='startTime')
axes[1, 0].set_xlabel('Time (s)')
axes[1, 0].set_ylabel('Z Position (m)')
axes[1, 0].set_title('Z Position vs Time')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# XY trajectory with point labels
axes[1, 1].plot(positions[:, 0], positions[:, 1], 'k-', alpha=0.3, linewidth=0.5)

# Mark all 9 grid positions
for x in [0.4, 0.5, 0.6]:
    for y in [-0.1, 0.0, 0.1]:
        axes[1, 1].plot(x, y, 's', color='lightgray', markersize=20, alpha=0.5)

# Color points by time
scatter = axes[1, 1].scatter(positions[::1000, 0], positions[::1000, 1], 
                              c=times[::1000], cmap='viridis', s=30, alpha=0.8, zorder=5)

# Mark sequence numbers at transition points
pairs = generate_9_unique_pairs(traj.seed)
for i in range(9):
    axes[1, 1].annotate(str(i+1), (pairs[i,0], pairs[i,1]), 
                        fontsize=12, fontweight='bold', ha='center', va='center',
                        color='red')

axes[1, 1].set_xlabel('X Position (m)')
axes[1, 1].set_ylabel('Y Position (m)')
axes[1, 1].set_title('XY Trajectory (Numbers = Visit Order in Cycle 1)')
axes[1, 1].axis('equal')
axes[1, 1].grid(True, alpha=0.3)
plt.colorbar(scatter, ax=axes[1, 1], label='Time (s)')

plt.tight_layout()
plt.savefig('trajectory_unique9.png', dpi=150, bbox_inches='tight')
print("\nPlot saved to 'trajectory_unique9.png'")
plt.show()

# Print summary
print("\n" + "="*60)
print("UNIQUE 9-PAIR TRAJECTORY SUMMARY")
print("="*60)
print(f"Total time: {TIME_STOP}s")
print(f"Step time: {traj.stepTime}s")
print(f"Points per cycle: 9")
print(f"Cycles in simulation: {int((TIME_STOP - traj.startTime) / (9 * traj.stepTime)) + 1}")
print("\nVisited points:")
prev_k = -1
for i, k in enumerate(traj.k_history):
    if k != prev_k:
        t = traj.time_history[i]
        x = traj.x_history[i]
        y = traj.y_history[i]
        cycle = (k - 1) // 9 if k > 0 else 0
        pos_in_cycle = (k - 1) % 9 + 1 if k > 0 else 0
        print(f"  k={k:2d} (cycle {cycle}, pos {pos_in_cycle}): t={t:5.1f}s, x={x:.1f}, y={y:+.1f}")
        prev_k = k