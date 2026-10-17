"""
Potential Fields -- minimal, teaching-oriented version.

Idea (Khatib, 1986): the robot is a particle in a potential energy field U(q).
  - The goal creates an ATTRACTIVE potential (a "valley" at the goal).
  - Each obstacle creates a REPULSIVE potential (a "hill" around it).
The robot moves in the direction of the force F = -grad U, i.e. it "goes downhill".

Usage from other scripts:

    from potential_fields import generate_obstacles, plan
    obstacles = generate_obstacles(4, rng)               # [(center, radius), ...]
    trajectory, reached = plan(start, goal, obstacles)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------
k_att = 1.0    # attractive gain
k_rep = 5.0    # repulsive gain
rho_0 = 2.0    # obstacle influence distance
step = 0.05    # step size (gradient descent rate)
tolerance = 0.1
max_iter = 2000

# Random obstacles
min_radius, max_radius = 0.5, 1.2
clearance = 0.5    # minimum distance from start/goal/other obstacles

verbose = True  # prints the step-by-step forces


def _log(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)


# ---------------------------------------------------------------
# 2. Scenario
# ---------------------------------------------------------------
def generate_obstacles(n, rng, start=None, goal=None):
    """Draws up to n circular obstacles (center, radius) that do not cover
    start/goal and do not overlap each other."""
    start = np.array([0.0, 0.0]) if start is None else start
    goal = np.array([10.0, 10.0]) if goal is None else goal
    obs = []
    attempts = 0
    while len(obs) < n and attempts < 1000:
        attempts += 1
        center = rng.uniform(1.0, 9.0, size=2)
        radius = rng.uniform(min_radius, max_radius)
        # must not cover start or goal
        if np.linalg.norm(center - start) < radius + rho_0 / 2 + clearance:
            continue
        if np.linalg.norm(center - goal) < radius + rho_0 / 2 + clearance:
            continue
        # must not overlap other obstacles
        if any(np.linalg.norm(center - c) < radius + r + clearance for c, r in obs):
            continue
        obs.append((center, radius))
    return obs


# ---------------------------------------------------------------
# 3. Attractive force
#    U_att(q) = 1/2 * k_att * ||q - q_goal||^2
#    F_att(q) = -grad U_att = -k_att * (q - q_goal)
#    -> vector pointing to the goal, proportional to the distance.
# ---------------------------------------------------------------
def attractive_force(q, goal):
    _log("\n\n")
    _log("=========== attractive_force ==========")
    _log("k_att: ", k_att)
    _log("-k_att: ", (-k_att))
    _log("q: ", q)
    _log("goal: ", goal)
    _log("=========== =============== ==========")
    return -k_att * (q - goal)


# ---------------------------------------------------------------
# 4. Repulsive force (from ONE obstacle)
#    rho = distance from the obstacle's edge to the robot
#    U_rep(q) = 1/2 * k_rep * (1/rho - 1/rho_0)^2   if rho <= rho_0
#             = 0                                    if rho >  rho_0
#    F_rep(q) = k_rep * (1/rho - 1/rho_0) * (1/rho^2) * grad(rho)
#    grad(rho) = unit vector leaving the obstacle and pointing to the robot.
#    -> far away: no force. Close: a force that grows fast (tends to infinity).
# ---------------------------------------------------------------
def repulsive_force(q, center, radius):
    _log("\n\n")
    _log("=========== repulsive_force ==========")
    _log("center: ", center)
    _log("radius: ", radius)
    _log("q: ", q)

    difference = q - center
    _log("[difference = q - center]")
    _log("difference: ", difference)
    dist_center = np.linalg.norm(difference)
    _log("dist_center: ", dist_center)
    _log("rho = dist_center - radius")
    rho = dist_center - radius          # distance to the edge
    _log("rho: ", rho)
    _log("rho_0: ", rho_0)

    _log("rho > rho_0 : ", (rho > rho_0))
    if rho > rho_0:
        _log("OUTSIDE the influence zone")
        _log("Will return: ", np.zeros(2))
        return np.zeros(2)              # outside the influence zone

    _log("INSIDE the influence zone")
    _log("PREV rho: ", rho)
    _log("1e-3: ", 1e-3)
    rho = max(rho, 1e-3)                # avoids division by zero
    _log("NEW rho: ", rho)

    _log(f"{difference=}")
    _log(f"{dist_center=}")
    direction = difference / dist_center   # grad(rho), unit vector
    _log(f"{direction=}")

    _log(f"{k_rep * (1/rho - 1/rho_0) * (1/rho**2) * direction=}")

    _log("=========== =============== ===========")
    return k_rep * (1/rho - 1/rho_0) * (1/rho**2) * direction


# ---------------------------------------------------------------
# 5. Main loop: gradient descent
#    q_{k+1} = q_k + step * F_total(q_k)
# ---------------------------------------------------------------
def plan(start, goal, obstacles):
    """Returns (trajectory Nx2, reached: bool)."""
    q = np.array(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    trajectory = [q.copy()]
    reached = False

    for i in range(max_iter):
        F = attractive_force(q, goal)
        for center, radius in obstacles:
            F += repulsive_force(q, center, radius)

        q = q + step * F
        trajectory.append(q.copy())

        if np.linalg.norm(q - goal) < tolerance:
            _log(f"Goal reached in {i + 1} iterations.")
            reached = True
            break
    else:
        _log("Did not reach the goal (likely a local minimum).")

    return np.array(trajectory), reached


# ---------------------------------------------------------------
# 6. Direct run: random scenario + visualization
# ---------------------------------------------------------------
if __name__ == "__main__":
    start = np.array([0.0, 0.0])
    goal = np.array([10.0, 10.0])

    n_obstacles = 4
    seed = None     # set an integer to reproduce the same scenario
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2**32)
    rng = np.random.default_rng(seed)
    print("seed:", seed)

    obstacles = generate_obstacles(n_obstacles, rng, start, goal)
    print("=========== obstacles ===========")
    for k, (c, r) in enumerate(obstacles):
        print(f"obstacle {k}: center=({c[0]:.2f}, {c[1]:.2f}) radius={r:.2f}")
    print("=================================")

    trajectory, _ = plan(start, goal, obstacles)

    fig, ax = plt.subplots(figsize=(6, 6))
    for center, radius in obstacles:
        ax.add_patch(plt.Circle(center, radius, color="gray"))
        ax.add_patch(plt.Circle(center, radius + rho_0, fill=False, ls="--", color="gray"))
    ax.plot(trajectory[:, 0], trajectory[:, 1], "b.-", markersize=2, label="trajectory")
    ax.plot(*start, "go", markersize=10, label="start")
    ax.plot(*goal, "r*", markersize=15, label="goal")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Potential Fields")
    plt.savefig("potential_fields.png", dpi=120)
    plt.show()
