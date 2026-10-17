"""
Following the path with PID -- planner (A* / Potential Fields) + control.

The planners only say WHERE to go (a list of points). A real robot has
inertia and does not "teleport" from point to point. Here a robot with mass
tries to follow each path using two PIDs:
  - SPEED PID:   adjusts the motor force to reach the desired speed.
  - HEADING PID: adjusts the torque to point the robot at the next point.

For each planner the plot shows:
  - WITHOUT PID: the planned path (ideal, as if the robot were a particle with no inertia);
  - WITH PID:    the path the robot actually drove.

Usage: python follow_path.py [seed] [--quiet] [--no-window]
  --quiet     : silences the potential_fields force log
  --no-window : only saves follow_path.png, without opening the Matplotlib window
"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import potential_fields as pf
from compare_planners import (START, GOAL, create_scenario, create_grid,
                              run_pf, run_astar, path_length)
from pid import PID
from astar import OBSTACLE

# ---------------------------------------------------------------
# 1. Robot and follower parameters
# ---------------------------------------------------------------
dt = 0.05
T_max = 80.0           # maximum simulation time (s)

mass = 1.0             # kg
inertia = 0.5          # moment of inertia (kg.m^2)
lin_friction = 0.5     # linear friction (N per m/s)
ang_friction = 0.5     # angular friction (N.m per rad/s)
max_force = 3.0        # motor limit (N)
max_torque = 3.0       # turning limit (N.m)

v_max = 1.5            # cruise speed (m/s)
waypoint_spacing = 0.5 # distance between target points (m)
accept_radius = 0.3    # how close a point counts as "reached" (m)
goal_tol = 0.2


# ---------------------------------------------------------------
# 2. Utilities
# ---------------------------------------------------------------
def resample(path, spacing):
    """Evenly spaced points along the path (always includes the last one)."""
    path = np.asarray(path)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.append(np.arange(spacing, s[-1], spacing), s[-1])
    return np.column_stack([np.interp(targets, s, path[:, 0]),
                            np.interp(targets, s, path[:, 1])])


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi      # wraps the angle into [-pi, pi]


def tracking_error(trajectory, path):
    """Distance from each driven point to the nearest point on the planned path."""
    dense = resample(path, 0.02)
    d = np.linalg.norm(trajectory[:, None, :] - dense[None, :, :], axis=-1)
    return d.min(axis=1)


def collided(trajectory, obstacles):
    return any(np.any(np.linalg.norm(trajectory - c, axis=1) < r) for c, r in obstacles)


# ---------------------------------------------------------------
# 3. Robot with inertia following a list of points
#    State: position (x, y), angle th, speed v, angular speed w
#      m * dv/dt = F - lin_friction * v
#      J * dw/dt = tau - ang_friction * w
# ---------------------------------------------------------------
def follow(path):
    points = resample(path, waypoint_spacing)
    pid_v = PID(kp=3.0, ki=0.5, kd=0.0)      # speed
    pid_th = PID(kp=4.0, ki=0.0, kd=1.0)     # heading

    x, y = path[0]
    th = np.arctan2(points[0][1] - y, points[0][0] - x)   # starts already pointing at the 1st point
    v = w = 0.0
    k = 0                                                   # index of the current target point
    reached = False                                         # reached the end of the path?
    trajectory = [(x, y)]

    for _ in range(int(T_max / dt)):
        target = points[k]
        dist = np.hypot(target[0] - x, target[1] - y)

        # reached the target point? move to the next one (the last requires actually arriving)
        last = k == len(points) - 1
        if dist < (goal_tol if last else accept_radius):
            if last:
                reached = True
                break
            k += 1
            target = points[k]
            dist = np.hypot(target[0] - x, target[1] - y)

        # --- Heading PID: error = angle to the target - robot angle
        th_error = normalize_angle(np.arctan2(target[1] - y, target[0] - x) - th)
        tau = pid_th.compute(th_error, 0.0, dt)

        # --- Speed PID: go slowly if facing the wrong way
        #     or close to the last point
        v_des = v_max * max(0.0, np.cos(th_error)) ** 3
        if last:
            v_des = min(v_des, 1.5 * dist)
        F = pid_v.compute(v_des, v, dt)

        # --- physical actuator limits
        F = np.clip(F, -max_force, max_force)
        tau = np.clip(tau, -max_torque, max_torque)

        # --- physics (Euler)
        v += (F - lin_friction * v) / mass * dt
        w += (tau - ang_friction * w) / inertia * dt
        th += w * dt
        x += v * np.cos(th) * dt
        y += v * np.sin(th) * dt
        trajectory.append((x, y))

    trajectory = np.array(trajectory)
    return trajectory, reached


# ---------------------------------------------------------------
# 4. Main program
# ---------------------------------------------------------------
def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seed, obstacles = create_scenario(int(args[0]) if args else None)
    print("seed:", seed)
    for k, (c, r) in enumerate(obstacles):
        print(f"obstacle {k}: center=({c[0]:.2f}, {c[1]:.2f}) radius={r:.2f}")

    pf.verbose = "--quiet" not in sys.argv
    grid = create_grid(obstacles)
    plans = {
        "Potential Fields": run_pf(obstacles),
        "A*": run_astar(grid),
    }

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    print()
    print(f"{'':<18}{'path end':>9}{'L plan':>10}{'L real':>9}{'max err':>10}{'mean err':>12}{'collision':>11}")
    for ax, (name, (path, plan_ok)) in zip(axs, plans.items()):
        trajectory, reached = follow(path)
        error = tracking_error(trajectory, path)
        hit = collided(trajectory, obstacles)
        print(f"{name:<18}{str(reached):>9}{path_length(path):>10.2f}"
              f"{path_length(trajectory):>9.2f}{error.max():>10.2f}{error.mean():>12.3f}"
              f"{('YES' if hit else 'no'):>11}")

        for center, radius in obstacles:
            ax.add_patch(plt.Circle(center, radius, color="gray", alpha=0.6))
        ax.plot(path[:, 0], path[:, 1], "--", color="k", lw=1.5,
                label=f"without PID: planned (L={path_length(path):.2f})")
        ax.plot(trajectory[:, 0], trajectory[:, 1], "-", color="tab:red", lw=2,
                label=f"with PID: driven (L={path_length(trajectory):.2f})")
        ax.plot(*START, "go", markersize=10, label="start")
        ax.plot(*GOAL, "r*", markersize=15, label="goal")
        ax.set_aspect("equal")
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        title = name
        if not plan_ok:
            title += " — planner FAILED"
        elif hit:
            title += " — COLLIDED"
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"seed = {seed}")
    plt.tight_layout()
    plt.savefig("follow_path.png", dpi=120)
    if "--no-window" not in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
