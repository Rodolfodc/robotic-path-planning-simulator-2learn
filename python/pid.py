"""
PID (Proportional-Integral-Derivative) -- minimal, teaching-oriented version.

Idea: the controller looks at the ERROR (e = target - measurement) and computes a command u:
  - P (proportional): reacts to the error RIGHT NOW.     u_P = kp * e
  - I (integral):     reacts to the ACCUMULATED error.   u_I = ki * sum(e * dt)
  - D (derivative):   reacts to the error's TREND.       u_D = kd * (de/dt)
  u = u_P + u_I + u_D

Example: speed control of a car (cruise control).
  - The engine applies a force u.
  - Air drag slows the car proportionally to its speed.
  - At 10 s the car hits a HILL (disturbance) that slows it down even more.
Compares P, PI and PID to see what each term does.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Scenario
# ---------------------------------------------------------------
target = 10.0        # desired speed (m/s)
v0 = 0.0             # initial speed (m/s)
dt = 0.05            # time step (s)
T = 20.0             # simulation duration (s)

mass = 1.0           # kg
drag = 0.5           # air drag coefficient (N per m/s)
t_hill = 10.0        # instant the hill starts (s)
hill_force = 2.0     # extra force against the car on the hill (N)


# ---------------------------------------------------------------
# 2. PID controller
#    Keeps the "memory" it needs: the integral and the previous error.
# ---------------------------------------------------------------
class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.prev_error = None

    def compute(self, target, measurement, dt):
        error = target - measurement

        # P: proportional to the current error
        p = self.kp * error

        # I: sum of the error over time (removes steady-state error)
        self.integral += error * dt
        i = self.ki * self.integral

        # D: rate of change of the error (damps, avoids overshooting the target)
        if self.prev_error is None:
            d = 0.0
        else:
            d = self.kd * (error - self.prev_error) / dt
        self.prev_error = error

        return p + i + d


# ---------------------------------------------------------------
# 3. Car simulation: m * dv/dt = u - drag * v - disturbance
#    (Euler integration: v_{k+1} = v_k + dv/dt * dt)
# ---------------------------------------------------------------
def simulate(pid):
    t = np.arange(0.0, T, dt)
    v = v0
    speeds, commands = [], []

    for tk in t:
        u = pid.compute(target, v, dt)                      # controller decides
        disturbance = hill_force if tk >= t_hill else 0.0
        acceleration = (u - drag * v - disturbance) / mass  # car physics
        v = v + acceleration * dt
        speeds.append(v)
        commands.append(u)

    return t, np.array(speeds), np.array(commands)


if __name__ == "__main__":
    # ---------------------------------------------------------------
    # 4. Comparison: P only, PI and PID
    #    Note: with this time step (dt) and this mass, kd >= 1 makes the
    #    simulation unstable (the D term amplifies Euler's numerical noise).
    # ---------------------------------------------------------------
    controllers = {
        "P   (kp=2)":              PID(kp=2.0, ki=0.0, kd=0.0),
        "PI  (kp=2, ki=1)":        PID(kp=2.0, ki=1.0, kd=0.0),
        "PID (kp=2, ki=1, kd=0.3)": PID(kp=2.0, ki=1.0, kd=0.3),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for name, pid in controllers.items():
        t, v, u = simulate(pid)
        final_error = target - v[-1]
        print(f"{name:<24} final speed = {v[-1]:.2f}  final error = {final_error:.2f}")
        ax1.plot(t, v, label=name)
        ax2.plot(t, u, label=name)

    ax1.axhline(target, color="k", ls="--", label="target")
    ax1.axvline(t_hill, color="gray", ls=":")
    ax1.text(t_hill + 0.2, 1, "hill", color="gray")
    ax1.set_ylabel("speed (m/s)")
    ax1.legend()
    ax1.set_title("PID: speed control")

    ax2.axvline(t_hill, color="gray", ls=":")
    ax2.set_ylabel("command u (N)")
    ax2.set_xlabel("time (s)")

    plt.tight_layout()
    plt.savefig("pid.png", dpi=120)
    plt.show()
