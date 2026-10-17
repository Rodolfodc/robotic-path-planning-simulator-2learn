# Path planners

Compares two ways of getting from start to goal among random circular obstacles, and shows how a robot with inertia, driven by PID, follows each path.

- **A\*** searches the grid and finds the shortest path.
- **Potential fields** descend a gradient: the goal attracts, the obstacles repel. It is fast, but it can get stuck in a local minimum.
- **PID** makes the robot track the planned path, with mass, inertia and motor limits.

## View online

**https://rodolfodc.github.io/robotic-path-planning-simulator-2learn/**

Nothing to install. Change any parameter and the result is recomputed instantly. Each parameter shows which algorithm it belongs to and an explanation on hover, and can be marked as **random** to be drawn on every run.

![Planned path (dashed) and path driven with PID (solid)](docs/follow_path.png)

## Run the Python code

```bash
pip install -r python/requirements.txt
cd python
python compare_planners.py 2      # compares the two planners (seed 2)
python follow_path.py 2           # adds the PID-driven robot following each path
python potential_fields.py        # potential fields only, with the step-by-step forces
python pid.py                     # PID alone: P, PI and PID on a speed controller
python interface_planners.py      # graphical interface (Tkinter)
```

`compare_planners.py` and `follow_path.py` accept `--quiet` (silences the force log), and `follow_path.py` also accepts `--no-window` (only saves the PNG, without opening a window). The seed defines the scenario: the same seed always generates the same obstacles.

| file | what it does |
|---|---|
| `astar.py` | A\* on a 2D grid, no interface |
| `potential_fields.py` | potential fields: forces, obstacle generation, planning |
| `pid.py` | `PID` class and a speed-control example |
| `follow_path.py` | robot with inertia following a path with two PIDs |
| `compare_planners.py` | runs both planners on the same scenario |
| `interface_planners.py` | graphical interface with parameters and random draws |
| `interface_astar.py` | interactive step-by-step A\* interface (opened by `run.sh`) |

## The web page

The page is a **JavaScript rewrite** of the engine (`web/engine.js`), so it runs in the browser without a server. Two consequences:

- If you change the `.py` files, the page does not follow. Results may diverge until `engine.js` is updated.
- The random number generator is different, so the same seed does not produce the same scenario in Python and on the page.

To regenerate `index.html` after editing `web/engine.js` or `web/template.html`:

```bash
python web/build.py
```

## Publish on GitHub Pages

In **Settings → Pages**, choose *Deploy from a branch*, branch `main`, folder `/ (root)`. The repository must be public.

## License

[Apache 2.0](LICENSE). You may use, modify and distribute it freely, including in commercial projects, as long as you keep the copyright notice and the [NOTICE](NOTICE) file, giving credit to Rodolfo Dalla Costa.
