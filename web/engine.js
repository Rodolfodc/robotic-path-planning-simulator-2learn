/* Planner engine: Potential Fields, A* and PID follower.
   JavaScript port of potential_fields.py, astar.py, pid.py and follow_path.py.
   Does not depend on the browser: also runs in Node (see the end of the file). */
const Engine = (() => {
  // ---------------------------------------------------------------
  // 1. Parameters: [key, label, default, type, random range, hint]
  //    range = [min, max] for numbers, or a list of options for choices.
  // ---------------------------------------------------------------
  const SECTIONS = [
    ["Scenario", [
      ["seed",        "Seed",                        2,     "int",   [0, 99999], "Number that drives the obstacle draw. The same seed always generates the same scenario."],
      ["n_obstacles", "No. of obstacles",            4,     "int",   [2, 8],     "How many circular obstacles to try to place. If they do not fit with the requested clearance, fewer appear."],
      ["min_radius",  "Minimum radius",              0.5,   "float", [0.3, 0.8], "Smallest radius an obstacle can have (the map goes from 0 to 10)."],
      ["max_radius",  "Maximum radius",              1.2,   "float", [0.9, 1.5], "Largest radius an obstacle can have. If it ends up smaller than the minimum, the two are swapped."],
      ["clearance",   "Clearance between obstacles", 0.5,   "float", [0.2, 1.0], "Minimum distance between obstacles, and from them to the start and the goal. Avoids scenarios with no way through."],
      ["start_x",     "Start x",                     0,     "float", [0, 2],     "Horizontal position the robot departs from (0 to 10)."],
      ["start_y",     "Start y",                     0,     "float", [0, 2],     "Vertical position the robot departs from (0 to 10)."],
      ["goal_x",      "Goal x",                      10,    "float", [8, 10],    "Horizontal position of the destination (0 to 10)."],
      ["goal_y",      "Goal y",                      10,    "float", [8, 10],    "Vertical position of the destination (0 to 10)."],
    ], { id: "scenario", chip: "Scenario", note: "Applies to both algorithms: both face the same obstacles." }],
    ["Potential Fields", [
      ["k_att",     "Attractive gain (k_att)",   1.0,  "float", [0.5, 3],    "How strongly the goal pulls the robot. Higher: heads more directly to the destination, but passes closer to the obstacles."],
      ["k_rep",     "Repulsive gain (k_rep)",    5.0,  "float", [1, 15],     "How strongly the obstacles push the robot. Higher: keeps more distance, but can get stuck in narrow passages."],
      ["rho_0",     "Influence zone (rho_0)",    2.0,  "float", [1, 3],      "Distance from the obstacle's edge beyond which it stops repelling. Outside this zone the force is zero. It also keeps obstacles away from the start and the goal when drawing the scenario."],
      ["step",      "Step",                      0.05, "float", [0.02, 0.1], "Size of each gradient descent step. Small: smooth and slow. Large: fast, but can oscillate or diverge."],
      ["tolerance", "Goal tolerance",            0.1,  "float", [0.05, 0.2], "How close to the goal the planner considers it has arrived."],
      ["max_iter",  "Max. iterations",           2000, "int",   [500, 4000], "Step limit. If it runs out before arriving, the planner fails (likely a local minimum)."],
    ], { id: "pf", chip: "PF", note: "Only affects Potential Fields. A* ignores these parameters." }],
    ["A*", [
      ["res",      "Cell size",       0.2,         "float", [0.1, 0.2, 0.25, 0.5], "Side of each grid cell. Smaller: more precise path, but more cells to search."],
      ["margin",   "Safety margin",   0.1,         "float", [0, 0.3],              "Extra clearance around obstacles when marking cells as blocked. Larger: path stays farther away."],
      ["distance", "Distance metric", "euclidean", "str",   ["euclidean", "manhattan", "chebyshev"], "How to measure the cost of a step and the distance to the goal. Euclidean: straight line. Manhattan: sum of the axes (a diagonal costs 2). Chebyshev: the largest axis (a diagonal costs 1)."],
    ], { id: "astar", chip: "A*", note: "Only affects A*. Potential Fields ignores these parameters." }],
    ["Robot and PID", [
      ["v_max",            "Maximum speed",            1.5, "float", [0.5, 2.5],  "Cruise speed the speed PID tries to reach on a straight line. Too high and the robot overshoots the points and cuts corners."],
      ["waypoint_spacing", "Spacing between targets",  0.5, "float", [0.25, 1],   "The planned path is split into points this far apart, and the robot goes from one to the next."],
      ["accept_radius",    "Acceptance radius",        0.3, "float", [0.15, 0.5], "How close to a target point the robot considers it reached and moves on to the next. Larger: cuts corners more."],
      ["mass",             "Mass",                     1.0, "float", [0.5, 2],    "Robot mass. Higher: more inertia, accelerates and brakes more slowly."],
      ["inertia",          "Rotational inertia",       0.5, "float", [0.2, 1],    "Resistance to turning. Higher: the robot takes longer to turn and to stop turning."],
      ["kp_v",             "Speed: Kp",                3.0, "float", [1, 6],      "Speed proportional term: reacts to the current error between the desired and actual speed."],
      ["ki_v",             "Speed: Ki",                0.5, "float", [0, 1],      "Speed integral term: accumulates the error over time and removes the remaining offset."],
      ["kd_v",             "Speed: Kd",                0.0, "float", [0, 0.3],    "Speed derivative term: damps fast changes. High values make the simulation unstable."],
      ["kp_th",            "Heading: Kp",              4.0, "float", [1, 8],      "Heading proportional term: how much the robot turns for each radian of error between its heading and the direction of the next point."],
      ["ki_th",            "Heading: Ki",              0.0, "float", [0, 0.3],    "Heading integral term: corrects a persistent angle error. Usually left at 0."],
      ["kd_th",            "Heading: Kd",              1.0, "float", [0, 2],      "Heading derivative term: damps the turning and avoids oscillating around the right heading."],
    ], { id: "pid", chip: "PID", note: "Controls the robot that follows the path, both A*'s and Potential Fields'. It does not change the planned path, only the driven one." }],
  ];

  const PARAMETERS = {};
  for (const [, ps, meta] of SECTIONS)
    for (const [k, l, d, t, r, hint] of ps) PARAMETERS[k] = { label: l, defaultValue: d, type: t, range: r, hint, section: meta.id };

  // Robot physics constants (not exposed)
  const DT = 0.05, T_MAX = 80, LIN_FRICTION = 0.5, ANG_FRICTION = 0.5;
  const MAX_FORCE = 3, MAX_TORQUE = 3, GOAL_TOL = 0.2;

  // ---------------------------------------------------------------
  // 2. Utilities
  // ---------------------------------------------------------------
  function createRng(seed) {                         // mulberry32: fast and reproducible
    let a = (seed >>> 0) + 0x6D2B79F5;
    return () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function draw(key, rnd = Math.random) {
    const { type, range } = PARAMETERS[key];
    if (range.length > 2 || type === "str") return range[Math.floor(rnd() * range.length)];
    const [lo, hi] = range;
    if (type === "int") return lo + Math.floor(rnd() * (hi - lo + 1));
    return Math.round((lo + rnd() * (hi - lo)) * 100) / 100;
  }

  function validate(p) {
    for (const k of ["step", "res", "max_iter", "v_max", "mass", "inertia", "waypoint_spacing", "accept_radius"])
      if (!(p[k] > 0)) return `"${PARAMETERS[k].label}" must be greater than zero.`;
    for (const k of ["start_x", "start_y", "goal_x", "goal_y"])
      if (!(p[k] >= 0 && p[k] <= 10)) return `"${PARAMETERS[k].label}" must be between 0 and 10.`;
    for (const [k, { type }] of Object.entries(PARAMETERS))
      if (type !== "str" && !Number.isFinite(p[k])) return `Invalid value in "${PARAMETERS[k].label}".`;
    return null;
  }

  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  function pathLength(c) { let s = 0; for (let i = 1; i < c.length; i++) s += dist(c[i - 1], c[i]); return s; }
  const wrapAngle = (a) => ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;

  function resample(c, spacing) {                    // evenly spaced points
    const s = [0];
    for (let i = 1; i < c.length; i++) s.push(s[i - 1] + dist(c[i - 1], c[i]));
    const total = s[s.length - 1];
    const targets = [];
    for (let a = spacing; a < total; a += spacing) targets.push(a);
    targets.push(total);
    const out = [];
    let j = 0;
    for (const a of targets) {
      while (j < s.length - 2 && s[j + 1] < a) j++;
      const seg = s[j + 1] - s[j];
      const f = seg > 0 ? (a - s[j]) / seg : 0;
      out.push([c[j][0] + f * (c[j + 1][0] - c[j][0]), c[j][1] + f * (c[j + 1][1] - c[j][1])]);
    }
    return out;
  }

  function distToSegment(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy;
    let t = l2 > 0 ? ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2 : 0;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
  }

  function trackingError(traj, path) {               // distance from each point to the planned path
    const errors = new Float64Array(traj.length);
    for (let i = 0; i < traj.length; i++) {
      let m = Infinity;
      for (let j = 1; j < path.length; j++) m = Math.min(m, distToSegment(traj[i], path[j - 1], path[j]));
      errors[i] = path.length > 1 ? m : dist(traj[i], path[0]);
    }
    return errors;
  }

  const collided = (traj, obs) => traj.some((q) => obs.some((o) => dist(q, o.c) < o.r));

  // ---------------------------------------------------------------
  // 3. Scenario
  // ---------------------------------------------------------------
  function generateObstacles(p, rnd, start, goal) {
    const obs = [];
    let attempts = 0;
    while (obs.length < p.n_obstacles && attempts < 1000) {
      attempts++;
      const c = [1 + rnd() * 8, 1 + rnd() * 8];
      const r = p.min_radius + rnd() * (p.max_radius - p.min_radius);
      if (dist(c, start) < r + p.rho_0 / 2 + p.clearance) continue;
      if (dist(c, goal) < r + p.rho_0 / 2 + p.clearance) continue;
      if (obs.some((o) => dist(c, o.c) < r + o.r + p.clearance)) continue;
      obs.push({ c, r });
    }
    return obs;
  }

  // ---------------------------------------------------------------
  // 4. Potential Fields:  q_{k+1} = q_k + step * F_total(q_k)
  //    F_att = -k_att (q - goal);  F_rep = k_rep (1/rho - 1/rho_0) / rho^2 * direction
  // ---------------------------------------------------------------
  function planPF(p, obs, start, goal) {
    let x = start[0], y = start[1];
    const traj = [[x, y]];
    let ok = false;
    for (let i = 0; i < p.max_iter; i++) {
      let fx = -p.k_att * (x - goal[0]), fy = -p.k_att * (y - goal[1]);
      for (const { c, r } of obs) {
        const dx = x - c[0], dy = y - c[1], d = Math.hypot(dx, dy);
        let rho = d - r;
        if (rho > p.rho_0 || d < 1e-9) continue;
        rho = Math.max(rho, 1e-3);
        const mag = p.k_rep * (1 / rho - 1 / p.rho_0) / (rho * rho);
        fx += mag * dx / d; fy += mag * dy / d;
      }
      x += p.step * fx; y += p.step * fy;
      if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e3 || Math.abs(y) > 1e3) break;  // diverged
      traj.push([x, y]);
      if (Math.hypot(x - goal[0], y - goal[1]) < p.tolerance) { ok = true; break; }
    }
    return { path: traj, ok };
  }

  // ---------------------------------------------------------------
  // 5. A* on an 8-connected grid (same logic as astar.py)
  // ---------------------------------------------------------------
  const METRICS = {
    euclidean: (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]),
    manhattan: (a, b) => Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]),
    chebyshev: (a, b) => Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])),
  };

  function createGrid(p, obs) {
    const n = Math.round(10 / p.res) + 1;
    const blocked = new Uint8Array(n * n);             // index = y * n + x
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
      const q = [x * p.res, y * p.res];
      if (obs.some((o) => dist(q, o.c) <= o.r + p.margin)) blocked[y * n + x] = 1;
    }
    return { n, blocked };
  }

  class Heap {                                         // min-priority queue by f
    constructor() { this.a = []; }
    get size() { return this.a.length; }
    push(item) {
      const a = this.a; a.push(item);
      let i = a.length - 1;
      while (i > 0) { const parent = (i - 1) >> 1; if (a[parent][0] <= a[i][0]) break; [a[parent], a[i]] = [a[i], a[parent]]; i = parent; }
    }
    pop() {
      const a = this.a, top = a[0], last = a.pop();
      if (a.length) {
        a[0] = last;
        let i = 0;
        for (;;) {
          let m = i; const l = 2 * i + 1, r = l + 1;
          if (l < a.length && a[l][0] < a[m][0]) m = l;
          if (r < a.length && a[r][0] < a[m][0]) m = r;
          if (m === i) break;
          [a[m], a[i]] = [a[i], a[m]]; i = m;
        }
      }
      return top;
    }
  }

  function planAStar(p, grid, start, goal) {
    const { n } = grid, blocked = grid.blocked.slice(), res = p.res;
    const d = METRICS[p.distance];
    const s = [Math.round(start[0] / res), Math.round(start[1] / res)];
    const g = [Math.round(goal[0] / res), Math.round(goal[1] / res)];
    blocked[s[1] * n + s[0]] = 0; blocked[g[1] * n + g[0]] = 0;   // start and goal are never blocked
    const gi = g[1] * n + g[0];
    const cost = new Float64Array(n * n).fill(Infinity);
    const parent = new Int32Array(n * n).fill(-1);
    const closed = new Uint8Array(n * n);
    const open = new Heap();
    cost[s[1] * n + s[0]] = 0;
    open.push([d(s, g), s[1] * n + s[0]]);
    while (open.size) {
      const [, idx] = open.pop();
      if (closed[idx]) continue;
      closed[idx] = 1;
      if (idx === gi) {
        const path = [];
        for (let k = idx; k !== -1; k = parent[k]) path.push([(k % n) * res, Math.floor(k / n) * res]);
        path.reverse();
        path[0] = [start[0], start[1]]; path[path.length - 1] = [goal[0], goal[1]];   // exact endpoints
        return { path, ok: true };
      }
      const x = idx % n, y = Math.floor(idx / n);
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        if (!dx && !dy) continue;
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= n || ny >= n) continue;
        const ni = ny * n + nx;
        if (blocked[ni] || closed[ni]) continue;
        const gn = cost[idx] + d([x, y], [nx, ny]);
        if (gn < cost[ni]) { cost[ni] = gn; parent[ni] = idx; open.push([gn + d([nx, ny], g), ni]); }
      }
    }
    return { path: [[start[0], start[1]]], ok: false };
  }

  // ---------------------------------------------------------------
  // 6. PID and robot with inertia
  //    m dv/dt = F - friction v ;  J dw/dt = tau - friction w
  // ---------------------------------------------------------------
  class PID {
    constructor(kp, ki, kd) { this.kp = kp; this.ki = ki; this.kd = kd; this.integral = 0; this.prev = null; }
    compute(target, measurement, dt) {
      const error = target - measurement;
      this.integral += error * dt;
      const d = this.prev === null ? 0 : this.kd * (error - this.prev) / dt;
      this.prev = error;
      return this.kp * error + this.ki * this.integral + d;
    }
  }

  function follow(path, p) {
    if (path.length < 2) return { traj: path.map((q) => [...q]), reached: false, points: path };
    const points = resample(path, p.waypoint_spacing);
    const pidV = new PID(p.kp_v, p.ki_v, p.kd_v), pidT = new PID(p.kp_th, p.ki_th, p.kd_th);
    let [x, y] = path[0];
    let th = Math.atan2(points[0][1] - y, points[0][0] - x), v = 0, w = 0, k = 0, reached = false;
    const traj = [[x, y]];
    for (let i = 0; i < Math.floor(T_MAX / DT); i++) {
      let target = points[k], d = Math.hypot(target[0] - x, target[1] - y);
      const last = k === points.length - 1;
      if (d < (last ? GOAL_TOL : p.accept_radius)) {
        if (last) { reached = true; break; }
        k++; target = points[k]; d = Math.hypot(target[0] - x, target[1] - y);
      }
      const errT = wrapAngle(Math.atan2(target[1] - y, target[0] - x) - th);
      let tau = pidT.compute(errT, 0, DT);
      let vDes = p.v_max * Math.max(0, Math.cos(errT)) ** 3;
      if (k === points.length - 1) vDes = Math.min(vDes, 1.5 * d);
      let F = pidV.compute(vDes, v, DT);
      F = Math.max(-MAX_FORCE, Math.min(MAX_FORCE, F));
      tau = Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, tau));
      v += (F - LIN_FRICTION * v) / p.mass * DT;
      w += (tau - ANG_FRICTION * w) / p.inertia * DT;
      th += w * DT;
      x += v * Math.cos(th) * DT; y += v * Math.sin(th) * DT;
      traj.push([x, y]);
    }
    return { traj, reached, points };
  }

  // ---------------------------------------------------------------
  // 7. Full run
  // ---------------------------------------------------------------
  function compute(p) {
    const start = [p.start_x, p.start_y], goal = [p.goal_x, p.goal_y];
    const obstacles = generateObstacles(p, createRng(p.seed), start, goal);
    const grid = createGrid(p, obstacles);
    const plans = {};
    for (const key of ["pf", "astar"]) {
      const t0 = performance.now();
      const { path, ok } = key === "pf" ? planPF(p, obstacles, start, goal) : planAStar(p, grid, start, goal);
      const tPlan = performance.now() - t0;
      const { traj, reached, points } = follow(path, p);
      const error = trackingError(traj, path);
      plans[key] = {
        path, ok, tPlan, lPlan: pathLength(path),
        traj, reached, points, lReal: pathLength(traj),
        maxError: error.reduce((m, e) => Math.max(m, e), 0),
        hit: collided(traj, obstacles), tSim: (traj.length - 1) * DT,
      };
    }
    return { params: p, start, goal, obstacles, grid, plans };
  }

  return { SECTIONS, PARAMETERS, draw, validate, compute, pathLength };
})();

if (typeof module !== "undefined") module.exports = Engine;
