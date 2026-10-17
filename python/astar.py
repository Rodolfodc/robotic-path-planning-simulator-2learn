"""A* algorithm on a 2D grid, with no dependency on a graphical interface.

Usage from other scripts:

    import numpy as np
    from astar import a_star, OBSTACLE, manhattan

    grid = np.ones((8, 8))            # traversal weight of each cell
    grid[3, 2:6] = OBSTACLE           # grid[y, x] = grid[row, column]
    grid[5, :] = 2.0                  # strip of more expensive terrain
    path, cost = a_star(grid, start=(0, 0), goal=(7, 7))
    path, cost = a_star(grid, (0, 0), (7, 7), distance=manhattan)
    # path = [(x, y), ...] from start to goal; (None, None) if there is none

Conventions:
- Coordinates (x, y) = (column, row), 0-based, and the grid is indexed grid[y, x].
- Each cell stores its traversal weight (>= 0). Cells equal to
  OBSTACLE (-1) or infinite (np.inf) are blocked.
- 8-connectivity.
- Step cost = distance(current, neighbor) * neighbor weight.
- Heuristic = distance(neighbor, goal) * smallest free weight in the grid.
  With a metric (triangle inequality holds), this heuristic is
  consistent and the returned path is optimal.
"""

import numpy as np

OBSTACLE = -1

# Columns of the Open table
IS_OPEN, X_PARENT, Y_PARENT, X, Y, G, F = range(7)


# ------------------------------------------------------------------------
# Distance metrics (strategies). Any function d(a, b) that is a metric
# works. Do not use distances that violate the triangle inequality
# (e.g. squared Euclidean): the heuristic overestimates and loses optimality.
# ------------------------------------------------------------------------
def euclidean(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def chebyshev(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def a_star(grid, start, goal, distance=euclidean, on_event=None):
    """Finds the lowest-cost path from `start` to `goal` with A*.

    Parameters
    ----------
    grid : array NR x NC
        Traversal weight of each cell; OBSTACLE or np.inf = blocked.
    start, goal : tuple (x, y)
        0-based coordinates.
    distance : function d(a, b) -> float
        Metric used for the step cost and the heuristic. Default: euclidean.
    on_event : optional function
        Called at every step, for anyone who wants to visualize the search:
            on_event('opened', x, y, F)              new cell in Open
            on_event('updated', x, y, F, old_F)      better path found
            on_event('expanded', x, y, open_table)   new current cell
        `open_table` is a copy of the Open table (columns IS_OPEN..F).

    Returns
    -------
    (path, cost) : (list of (x, y), float) or (None, None) if there is no path.
    """
    grid = np.asarray(grid, dtype=float)
    NR, NC = grid.shape
    xStart, yStart = start
    xGoal, yGoal = goal
    notify = on_event or (lambda *args: None)

    blocked = (grid == OBSTACLE) | np.isinf(grid)
    if np.any(grid[~blocked] < 0):
        raise ValueError('Negative weights invalidate A*. Use OBSTACLE or np.inf for blocking.')

    # Lower bound of the weight: the heuristic assumes the cheapest terrain until the end
    free = grid[~blocked]
    min_weight = float(free.min()) if free.size else 0.0

    def h(x, y):
        return distance((x, y), (xGoal, yGoal)) * min_weight

    # Initial cell: g = 0 and F = 0
    xCur, yCur = xStart, yStart
    CostStart2Cur = 0.0

    # Open: [ IsOpen xParent yParent x y g F ]
    # IsOpen = 1: still to explore; 0: already expanded
    Open = np.array([[0, xCur, yCur, xCur, yCur, 0.0, 0.0]])

    # Closed: blocked cells + already expanded cells, as [x, y]
    yObst, xObst = np.nonzero(blocked)
    Closed = np.vstack([np.column_stack([xObst, yObst]), [xCur, yCur]])

    # ------------------------------------------------------------
    # Forward phase: expand cells until the goal is reached
    # ------------------------------------------------------------
    while xCur != xGoal or yCur != yGoal:

        for dx in (-1, 0, 1):      # X axis (columns)
            for dy in (-1, 0, 1):  # Y axis (rows)

                xNb = xCur + dx
                yNb = yCur + dy

                # Inside the grid and not in the closed list
                if not (0 <= xNb < NC and 0 <= yNb < NR):
                    continue
                if np.any((Closed[:, 0] == xNb) & (Closed[:, 1] == yNb)):
                    continue

                step = distance((xCur, yCur), (xNb, yNb)) * grid[yNb, xNb]
                g = CostStart2Cur + step
                f = g + h(xNb, yNb)

                inList = (Open[:, X] == xNb) & (Open[:, Y] == yNb)
                if np.sum(inList) == 1:
                    # Already in Open: update if the new F is smaller
                    idxOpen = np.flatnonzero(inList)[0]
                    if Open[idxOpen, F] > f:
                        old_F = Open[idxOpen, F]
                        Open[idxOpen, [X_PARENT, Y_PARENT, G, F]] = [xCur, yCur, g, f]
                        notify('updated', xNb, yNb, f, old_F)
                else:
                    # New possibility to explore
                    Open = np.vstack([Open, [1, xCur, yCur, xNb, yNb, g, f]])
                    notify('opened', xNb, yNb, f)

        # Cells in Open not yet explored
        Unexplored = np.flatnonzero(Open[:, IS_OPEN] == 1)
        if Unexplored.size == 0:
            return None, None  # All cells analyzed: there is no path

        # Smallest F; on a tie, the first in the list
        IdxMinF = Unexplored[np.argmin(Open[Unexplored, F])]
        xCur = int(Open[IdxMinF, X])
        yCur = int(Open[IdxMinF, Y])
        CostStart2Cur = Open[IdxMinF, G]
        Closed = np.vstack([Closed, [xCur, yCur]])
        Open[IdxMinF, IS_OPEN] = 0

        notify('expanded', xCur, yCur, Open.copy())

    # ------------------------------------------------------------
    # Backward phase: from the goal back to the start following the parents
    # ------------------------------------------------------------
    def last_expanded(x, y):
        idx = np.flatnonzero((Open[:, X] == x) & (Open[:, Y] == y)
                             & (Open[:, IS_OPEN] == 0))
        return idx[-1]

    cost = float(Open[last_expanded(xGoal, yGoal), G])
    path = [(xGoal, yGoal)]
    x, y = xGoal, yGoal
    while x != xStart or y != yStart:
        ind = last_expanded(x, y)
        x, y = int(Open[ind, X_PARENT]), int(Open[ind, Y_PARENT])
        path.append((x, y))
    path.reverse()

    return path, cost
