"""
Interactive interface: A*, Potential Fields and PID on the same scenario.

    python interface_planners.py

  - "Display" tab: toggles what appears on the plot (algorithms, planned
    path, path driven with PID, obstacles, grid, ...).
  - Other tabs: parameters for each part, all with a default value.
  - "random" box next to each parameter: on every run a new value is drawn
    within the range shown and stays visible in the field. Uncheck the box
    to LOCK that value (and repeat the experiment).
  - To run again after changing a parameter: "Run" button or Enter.

Reuses (without changing) the existing modules: potential_fields.py, astar.py,
pid.py, follow_path.py and compare_planners.py.
"""

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import potential_fields as pf
from astar import a_star, OBSTACLE, euclidean, manhattan, chebyshev
from pid import PID
from follow_path import resample, normalize_angle, collided
from compare_planners import path_length

# ---------------------------------------------------------------
# 1. Parameters: (key, label, default, type, random range)
#    range = (min, max) for numbers, or a list of options for choices.
# ---------------------------------------------------------------
SECTIONS = [
    ("Scenario", [
        ("seed",          "seed",                 42,    int,   (0, 99999)),
        ("n_obstacles",   "no. of obstacles",     4,     int,   (2, 8)),
        ("min_radius",    "minimum radius",       0.5,   float, (0.3, 0.8)),
        ("max_radius",    "maximum radius",       1.2,   float, (0.9, 1.5)),
        ("clearance",     "clearance between obstacles", 0.5, float, (0.2, 1.0)),
        ("start_x",       "start x",              0.0,   float, (0.0, 2.0)),
        ("start_y",       "start y",              0.0,   float, (0.0, 2.0)),
        ("goal_x",        "goal x",               10.0,  float, (8.0, 10.0)),
        ("goal_y",        "goal y",               10.0,  float, (8.0, 10.0)),
    ]),
    ("Potential Fields", [
        ("k_att",     "attractive gain (k_att)",   1.0,  float, (0.5, 3.0)),
        ("k_rep",     "repulsive gain (k_rep)",    5.0,  float, (1.0, 15.0)),
        ("rho_0",     "influence zone (rho_0)",    2.0,  float, (1.0, 3.0)),
        ("step",      "step",                      0.05, float, (0.02, 0.1)),
        ("tolerance", "goal tolerance",            0.1,  float, (0.05, 0.2)),
        ("max_iter",  "max. iterations",           2000, int,   (500, 4000)),
    ]),
    ("A*", [
        ("res",      "cell size",        0.2,         float, [0.2, 0.25, 0.5]),
        ("margin",   "safety margin",    0.1,         float, (0.0, 0.3)),
        ("distance", "distance metric",  "euclidean", str,
         ["euclidean", "manhattan", "chebyshev"]),
    ]),
    ("Robot and PID", [
        ("v_max",            "maximum speed",            1.5, float, (0.5, 2.5)),
        ("waypoint_spacing", "spacing between targets",  0.5, float, (0.25, 1.0)),
        ("accept_radius",    "acceptance radius",        0.3, float, (0.15, 0.5)),
        ("mass",             "mass",                     1.0, float, (0.5, 2.0)),
        ("inertia",          "rotational inertia",       0.5, float, (0.2, 1.0)),
        ("kp_v",             "speed: Kp",                3.0, float, (1.0, 6.0)),
        ("ki_v",             "speed: Ki",                0.5, float, (0.0, 1.0)),
        ("kd_v",             "speed: Kd",                0.0, float, (0.0, 0.3)),
        ("kp_th",            "heading: Kp",              4.0, float, (1.0, 8.0)),
        ("ki_th",            "heading: Ki",              0.0, float, (0.0, 0.3)),
        ("kd_th",            "heading: Kd",              1.0, float, (0.0, 2.0)),
    ]),
]
PARAMETERS = {c: (l, d, t, r) for _, ps in SECTIONS for c, l, d, t, r in ps}

# Items in the "Display" tab: (key, label, default)
DISPLAY = [
    ("pf",        "Potential Fields",                   True),
    ("astar",     "A*",                                 True),
    ("planned",   "Planned path (without PID)",         True),
    ("driven",    "Driven path (with PID)",             True),
    ("obstacles", "Obstacles",                          True),
    ("influence", "Influence zone (rho_0)",             False),
    ("grid",      "A* blocked grid",                    False),
    ("points",    "Follower target points",             False),
]

COLORS = {"pf": ("tab:blue", "navy"), "astar": ("tab:orange", "saddlebrown")}
NAMES = {"pf": "Potential Fields", "astar": "A*"}
METRICS = {"euclidean": euclidean, "manhattan": manhattan, "chebyshev": chebyshev}

# Robot physics constants (not exposed in the interface)
DT = 0.05
T_MAX = 80.0
LIN_FRICTION = 0.5
ANG_FRICTION = 0.5
MAX_FORCE = 3.0
MAX_TORQUE = 3.0
GOAL_TOL = 0.2


def draw_value(kind, value_range, rng):
    """Draws a value within the parameter's range (or among its options)."""
    if isinstance(value_range, list):
        return value_range[int(rng.integers(len(value_range)))]
    lo, hi = value_range
    if kind is int:
        return int(rng.integers(lo, hi + 1))
    return round(float(rng.uniform(lo, hi)), 2)


# ---------------------------------------------------------------
# 2. Computation (no graphical interface; runs in a thread)
# ---------------------------------------------------------------
def plan_pf(p, obstacles, start, goal):
    pf.k_att, pf.k_rep, pf.rho_0 = p["k_att"], p["k_rep"], p["rho_0"]
    pf.step, pf.tolerance, pf.max_iter = p["step"], p["tolerance"], int(p["max_iter"])
    pf.verbose = False
    path, ok = pf.plan(start, goal, obstacles)
    # with bad gains the method diverges (huge values/NaN): cut at the last sensible point
    valid = np.all(np.isfinite(path), axis=1) & np.all(np.abs(path) < 1e3, axis=1)
    if not valid.all():
        path, ok = path[:int(np.argmin(valid))], False
        if len(path) == 0:
            path = np.array([start])
    return path, ok


def create_grid(p, obstacles):
    """Rasterizes the circles. grid[y, x]: 1 = free, OBSTACLE = blocked."""
    res, margin = p["res"], p["margin"]
    n = int(round(10 / res)) + 1
    xs = np.arange(n) * res
    X, Y = np.meshgrid(xs, xs)
    grid = np.ones((n, n))
    for center, radius in obstacles:
        grid[np.hypot(X - center[0], Y - center[1]) <= radius + margin] = OBSTACLE
    return grid


def plan_astar(p, grid, start, goal):
    res = p["res"]
    start_cell = tuple(int(v) for v in np.round(start / res))
    goal_cell = tuple(int(v) for v in np.round(goal / res))
    grid = grid.copy()
    grid[start_cell[1], start_cell[0]] = 1    # start and goal are never blocked
    grid[goal_cell[1], goal_cell[0]] = 1
    path, _ = a_star(grid, start_cell, goal_cell, distance=METRICS[p["distance"]])
    if path is None:
        return np.array([start]), False
    path = np.array(path, dtype=float) * res
    path[0], path[-1] = start, goal           # exact endpoints (the cell is rounded)
    return path, True


def tracking_error(trajectory, path):
    """Distance from each driven point to the planned path.
    In chunks and with the path capped at ~2000 points, so memory always stays small."""
    dense = resample(path, max(0.02, path_length(path) / 2000))
    errors = np.empty(len(trajectory))
    for i in range(0, len(trajectory), 100):
        chunk = trajectory[i:i + 100]
        d = np.linalg.norm(chunk[:, None, :] - dense[None, :, :], axis=-1)
        errors[i:i + 100] = d.min(axis=1)
    return errors


def follow(path, p):
    """Robot with inertia following the path with two PIDs (speed and heading).
    State: (x, y), angle th, speed v, angular speed w."""
    if len(path) < 2:
        return path.copy(), False, path.copy()

    points = resample(path, p["waypoint_spacing"])
    pid_v = PID(p["kp_v"], p["ki_v"], p["kd_v"])
    pid_th = PID(p["kp_th"], p["ki_th"], p["kd_th"])

    x, y = path[0]
    th = np.arctan2(points[0][1] - y, points[0][0] - x)
    v = w = 0.0
    k = 0
    reached = False
    trajectory = [(x, y)]

    for _ in range(int(T_MAX / DT)):
        target = points[k]
        dist = np.hypot(target[0] - x, target[1] - y)

        last = k == len(points) - 1
        if dist < (GOAL_TOL if last else p["accept_radius"]):
            if last:
                reached = True
                break
            k += 1
            target = points[k]
            dist = np.hypot(target[0] - x, target[1] - y)

        th_error = normalize_angle(np.arctan2(target[1] - y, target[0] - x) - th)
        tau = pid_th.compute(th_error, 0.0, DT)

        v_des = p["v_max"] * max(0.0, np.cos(th_error)) ** 3
        if last:
            v_des = min(v_des, 1.5 * dist)
        F = pid_v.compute(v_des, v, DT)

        F = np.clip(F, -MAX_FORCE, MAX_FORCE)
        tau = np.clip(tau, -MAX_TORQUE, MAX_TORQUE)

        v += (F - LIN_FRICTION * v) / p["mass"] * DT
        w += (tau - ANG_FRICTION * w) / p["inertia"] * DT
        th += w * DT
        x += v * np.cos(th) * DT
        y += v * np.sin(th) * DT
        trajectory.append((x, y))

    return np.array(trajectory), reached, points


def compute(p, use_pf, use_astar, with_pid, with_grid):
    """Generates the scenario and runs what was requested. Returns a dictionary of results."""
    start = np.array([p["start_x"], p["start_y"]], dtype=float)
    goal = np.array([p["goal_x"], p["goal_y"]], dtype=float)

    # parameters the obstacle generator reads from the potential_fields module
    pf.min_radius, pf.max_radius, pf.clearance, pf.rho_0 = (
        p["min_radius"], p["max_radius"], p["clearance"], p["rho_0"])
    rng = np.random.default_rng(p["seed"])
    obstacles = pf.generate_obstacles(p["n_obstacles"], rng, start, goal)

    grid = create_grid(p, obstacles) if (use_astar or with_grid) else None

    plans = {}
    selected = [("pf", use_pf), ("astar", use_astar)]
    for key, active in selected:
        if not active:
            continue
        t0 = time.perf_counter()
        if key == "pf":
            path, ok = plan_pf(p, obstacles, start, goal)
        else:
            path, ok = plan_astar(p, grid, start, goal)
        info = {"path": path, "ok": ok, "t_plan": time.perf_counter() - t0,
                "L_plan": path_length(path) if len(path) > 1 else 0.0}

        if with_pid:
            traj, reached, points = follow(path, p)
            info.update(trajectory=traj, reached=reached, points=points,
                        L_real=path_length(traj) if len(traj) > 1 else 0.0,
                        error=tracking_error(traj, path),
                        hit=collided(traj, obstacles),
                        t_sim=(len(traj) - 1) * DT)
        plans[key] = info

    return {"params": p, "start": start, "goal": goal,
            "obstacles": obstacles, "grid": grid, "plans": plans}


# ---------------------------------------------------------------
# 3. Interface
# ---------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title("A*, Potential Fields and PID")
        root.protocol("WM_DELETE_WINDOW", self._close)

        self.vars = {}           # key -> StringVar with the displayed value
        self.random = {}         # key -> BooleanVar (draw on every run?)
        self.widgets = {}        # key -> input field
        self.display = {}        # key -> BooleanVar (show on the plot?)
        self.rng = np.random.default_rng()
        self.queue = queue.Queue()
        self.result = None
        self.busy = False

        self._build_panel()
        self._build_plot()
        self.run()

    # ---------- layout ----------
    def _build_panel(self):
        left = ttk.Frame(self.root, padding=8)
        left.pack(side="left", fill="y")

        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=(0, 8))
        self.buttons = [
            ttk.Button(buttons, text="Run", command=self.run),
            ttk.Button(buttons, text="Randomize all", command=self._randomize_all),
            ttk.Button(buttons, text="Lock values", command=self._lock_values),
            ttk.Button(buttons, text="Defaults", command=self._defaults),
        ]
        for b in self.buttons:
            b.pack(side="left", padx=2)

        self.tabs = ttk.Notebook(left)
        self.tabs.pack(fill="y", expand=True)

        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="Display")
        for i, (key, label, default) in enumerate(DISPLAY):
            self.display[key] = tk.BooleanVar(value=default)
            ttk.Checkbutton(tab, text=label, variable=self.display[key],
                            command=self._on_display_change).grid(row=i, column=0, sticky="w", pady=2)

        for name, parameters in SECTIONS:
            self._build_tab(name, parameters)

        self.status = tk.StringVar(value="")
        ttk.Label(left, textvariable=self.status, wraplength=320).pack(anchor="w", pady=(8, 0))

    def _build_tab(self, name, parameters):
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text=name)
        for col, title in enumerate(["parameter", "value", "random", "range"]):
            ttk.Label(tab, text=title, foreground="gray").grid(row=0, column=col, sticky="w", padx=4)

        for i, (key, label, default, kind, value_range) in enumerate(parameters, start=1):
            ttk.Label(tab, text=label).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            self.vars[key] = tk.StringVar(value=str(default))
            if isinstance(value_range, list):
                field = ttk.Combobox(tab, textvariable=self.vars[key], width=11, state="readonly",
                                     values=[str(o) for o in value_range])
                range_text = "options"
            else:
                field = ttk.Entry(tab, textvariable=self.vars[key], width=12)
                field.bind("<Return>", lambda e: self.run())
                range_text = f"{value_range[0]} – {value_range[1]}"
            field.grid(row=i, column=1, padx=4)
            self.widgets[key] = field

            self.random[key] = tk.BooleanVar(value=False)
            ttk.Checkbutton(tab, variable=self.random[key],
                            command=lambda c=key: self._update_state(c)).grid(row=i, column=2)
            ttk.Label(tab, text=range_text, foreground="gray").grid(row=i, column=3, sticky="w", padx=4)

    def _build_plot(self):
        right = ttk.Frame(self.root, padding=8)
        right.pack(side="left", fill="both", expand=True)

        self.fig = Figure(figsize=(7, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas, right, pack_toolbar=False)
        toolbar.pack(side="top", fill="x")

        self.text = tk.Text(right, height=7, font="TkFixedFont", state="disabled")
        self.text.pack(side="bottom", fill="x", pady=(8, 0))
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # ---------- button actions ----------
    def _update_state(self, key):
        self.widgets[key].state(["disabled"] if self.random[key].get() else ["!disabled"])

    def _randomize_all(self):
        for key in self.random:
            self.random[key].set(True)
            self._update_state(key)
        self.run()

    def _lock_values(self):
        for key in self.random:
            self.random[key].set(False)
            self._update_state(key)

    def _defaults(self):
        for key, (_, default, _, _) in PARAMETERS.items():
            self.vars[key].set(str(default))
            self.random[key].set(False)
            self._update_state(key)
        self.run()

    def _on_display_change(self):
        """Only recomputes if what was turned on has not been computed yet."""
        r = self.result
        missing = r is None
        if r is not None:
            for key in ("pf", "astar"):
                if self.display[key].get() and key not in r["plans"]:
                    missing = True
            if self.display["driven"].get():
                if any("trajectory" not in info for info in r["plans"].values()):
                    missing = True
            if self.display["grid"].get() and r["grid"] is None:
                missing = True
        if missing:
            self.run()
        else:
            self.redraw()

    # ---------- reading the parameters ----------
    def _read_parameters(self):
        p = {}
        for key, (label, _, kind, value_range) in PARAMETERS.items():
            if self.random[key].get():
                value = draw_value(kind, value_range, self.rng)
                self.vars[key].set(str(value))      # shows the drawn value
            else:
                text = self.vars[key].get().strip().replace(",", ".")
                try:
                    value = int(float(text)) if kind is int else kind(text)
                except ValueError:
                    raise ValueError(f"Invalid value in '{label}': {text!r}")
            p[key] = value

        if p["max_radius"] < p["min_radius"]:
            p["min_radius"], p["max_radius"] = p["max_radius"], p["min_radius"]
            self.vars["min_radius"].set(str(p["min_radius"]))
            self.vars["max_radius"].set(str(p["max_radius"]))
        for key in ("step", "res", "max_iter", "v_max", "mass", "inertia", "waypoint_spacing", "accept_radius"):
            if p[key] <= 0:
                raise ValueError(f"'{PARAMETERS[key][0]}' must be greater than zero.")
        for key in ("start_x", "start_y", "goal_x", "goal_y"):
            if not 0 <= p[key] <= 10:
                raise ValueError(f"'{PARAMETERS[key][0]}' must be between 0 and 10.")
        return p

    # ---------- execution (thread) ----------
    def run(self):
        if self.busy:
            return
        try:
            p = self._read_parameters()
        except ValueError as e:
            self.status.set(str(e))
            return

        request = dict(use_pf=self.display["pf"].get(), use_astar=self.display["astar"].get(),
                       with_pid=self.display["driven"].get(), with_grid=self.display["grid"].get())
        self.busy = True
        for b in self.buttons:
            b.state(["disabled"])
        self.status.set("Computing...")

        def job():
            try:
                self.queue.put(("ok", compute(p, **request)))
            except Exception as e:                     # shows the error in the interface
                self.queue.put(("error", f"{type(e).__name__}: {e}"))

        threading.Thread(target=job, daemon=True).start()
        self.root.after(100, self._poll)

    def _poll(self):
        try:
            kind, data = self.queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll)
            return
        self.busy = False
        for b in self.buttons:
            b.state(["!disabled"])
        if kind == "ok":
            self.result = data
            self.status.set(f"Done. seed = {data['params']['seed']}")
            self.redraw()
        else:
            self.status.set(f"Error: {data}")

    # ---------- drawing ----------
    def redraw(self):
        r = self.result
        if r is None:
            return
        p, ax = r["params"], self.ax
        ax.clear()

        if self.display["grid"].get() and r["grid"] is not None:
            m = p["res"] / 2
            ax.imshow(r["grid"] == OBSTACLE, origin="lower", extent=(-m, 10 + m, -m, 10 + m),
                      cmap="Greys", alpha=0.45)
        if self.display["obstacles"].get():
            for center, radius in r["obstacles"]:
                ax.add_patch(Circle(center, radius, fill=True, color="gray", alpha=0.7))
        if self.display["influence"].get():
            for center, radius in r["obstacles"]:
                ax.add_patch(Circle(center, radius + p["rho_0"], fill=False, ls="--",
                                    color="gray", alpha=0.5))

        for key, info in r["plans"].items():
            if not self.display[key].get():
                continue
            light, dark = COLORS[key]
            name = NAMES[key] + ("" if info["ok"] else " (FAILED)")
            if self.display["planned"].get():
                c = info["path"]
                ax.plot(c[:, 0], c[:, 1], "--", color=light, lw=1.6,
                        label=f"{name}: planned, without PID (L={info['L_plan']:.2f})")
            if self.display["driven"].get() and "trajectory" in info:
                t = info["trajectory"]
                ax.plot(t[:, 0], t[:, 1], "-", color=dark, lw=2.2,
                        label=f"{name}: driven, with PID (L={info['L_real']:.2f})")
                if self.display["points"].get():
                    ax.plot(info["points"][:, 0], info["points"][:, 1], ".", color=dark, markersize=5)

        ax.plot(*r["start"], "go", markersize=10, label="start")
        ax.plot(*r["goal"], "r*", markersize=15, label="goal")
        ax.set_aspect("equal")
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.grid(alpha=0.2)
        ax.set_title(f"seed = {p['seed']}")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper left", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._write_metrics()

    def _write_metrics(self):
        lines = [f"{'':<18}{'plan ok':>9}{'L plan':>9}{'L real':>8}{'max err':>10}"
                 f"{'reached':>9}{'collision':>11}{'plan (ms)':>11}"]
        for key, info in self.result["plans"].items():
            if not self.display[key].get():
                continue
            line = f"{NAMES[key]:<18}{str(info['ok']):>9}{info['L_plan']:>9.2f}"
            if "trajectory" in info:
                line += (f"{info['L_real']:>8.2f}{info['error'].max():>10.2f}{str(info['reached']):>9}"
                         f"{('YES' if info['hit'] else 'no'):>11}")
            else:
                line += f"{'-':>8}{'-':>10}{'-':>9}{'-':>11}"
            line += f"{info['t_plan'] * 1e3:>11.1f}"
            lines.append(line)
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n".join(lines))
        self.text.configure(state="disabled")

    def _close(self):
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1300x820")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
