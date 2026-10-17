# ------------------------------------------------------------------------
# Interactive interface for A* (choose obstacles, start and goal)
# ------------------------------------------------------------------------
# Run in the terminal (with astar.py in the same folder):
#     pip install numpy matplotlib
#     python interface_astar.py

import sys

import numpy as np
import matplotlib.pyplot as plt

from astar import a_star, OBSTACLE

# Number of rows and columns in the grid
NR = 8
NC = 8

# Show the F values and advance step by step only on small grids
SHOW_NUMBERS = NR < 11 and NC < 11


# ------------------------------------------------------------------------
# Window utilities
# ------------------------------------------------------------------------
def round_half_up(v):
    # MATLAB's round (half away from zero); Python's round is banker's rounding
    return int(np.floor(v + 0.5))


def exit_if_closed(fig):
    # If the user closes the window, exit the script instead of hanging the terminal
    if not plt.fignum_exists(fig.number):
        print('Window closed. Exiting.')
        sys.exit(0)


def click(fig):
    # Equivalent to MATLAB's [x, y, button] = ginput(1), returning the cell.
    # Clicks outside the grid are ignored.
    while True:
        event = {}

        def on_click(e):
            if e.inaxes is not None:
                event['x'], event['y'], event['b'] = e.xdata, e.ydata, int(e.button)

        cid = fig.canvas.mpl_connect('button_press_event', on_click)
        while not event:
            exit_if_closed(fig)
            plt.pause(0.05)
        fig.canvas.mpl_disconnect(cid)

        x, y = round_half_up(event['x']), round_half_up(event['y'])
        if 0 <= x < NC and 0 <= y < NR:
            return x, y, event['b']


def wait_for_key(fig):
    # Equivalent to MATLAB's "pause" with no argument: waits for a key press
    while True:
        exit_if_closed(fig)
        if plt.waitforbuttonpress():  # True = key, False = mouse
            return


def create_interface(NR, NC):
    # Creates the window with the NR x NC grid (cells centered at 0..NC-1, 0..NR-1)
    plt.ion()
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title('A* Algorithm')
    for i in np.arange(-0.5, NC):
        ax.plot([i, i], [-0.5, NR - 0.5], 'k')
    for i in np.arange(-0.5, NR):
        ax.plot([-0.5, NC - 0.5], [i, i], 'k')
    ax.set_aspect('equal')
    plt.show(block=False)
    plt.pause(0.1)
    return fig, ax


# ------------------------------------------------------------------------
# Interactive choices
# ------------------------------------------------------------------------
def choose_obstacles(fig, ax, grid):
    msg = 'Place the obstacles with the left button and the last one with the right button.'
    ax.set_xlabel(msg)
    print(msg)
    button = 1
    while button == 1:  # left button = 1, right = 3
        x, y, button = click(fig)
        grid[y, x] = OBSTACLE
        ax.plot(x, y, 'sk', markersize=20, linewidth=4)
        ax.plot(x, y, '*k', markersize=20, linewidth=4)
        fig.canvas.draw_idle()


def choose_point(fig, ax, msg, color, label, label_dx):
    ax.set_xlabel(msg)
    print(msg)
    x, y, _ = click(fig)
    ax.plot(x, y, 'o' + color)
    ax.plot(x, y, '*' + color)
    ax.text(x + label_dx, y + 0.3, label, color={'b': 'blue', 'r': 'red'}[color])
    return x, y


# ------------------------------------------------------------------------
# Visualization of the A* steps
# ------------------------------------------------------------------------
def create_visualizer(fig, ax):
    green = np.array([60, 153, 95]) / 256

    def on_event(kind, x, y, *data):
        if kind == 'opened':
            (f,) = data
            ax.plot(x, y, 'o', color=green, linewidth=2)
            if SHOW_NUMBERS:
                ax.text(x - 0.25, y - 0.25, f'{f:.2f}')
            plt.pause(0.001)

        elif kind == 'updated':
            f, old_f = data
            if SHOW_NUMBERS:
                ax.text(x - 0.25, y - 0.25, f'{old_f:.2f}', color='white')  # "erases"
                ax.plot(x, y, 'om', linewidth=2)
                ax.text(x - 0.25, y - 0.25, f'{f:.2f}')
            plt.pause(0.001)

        elif kind == 'expanded':
            (open_table,) = data
            if SHOW_NUMBERS:
                wait_for_key(fig)
            else:
                plt.pause(0.000001)
            ax.plot(x, y, '.m', linewidth=2, markersize=15)
            plt.pause(0.001)
            print('-------------------------------------')
            print(np.round(open_table).astype(int))

    return on_event


def draw_path(ax, path, cost):
    ax.set_title(f'Total path cost: {cost:g}')
    ax.set_xlabel('Best path found by A*.')
    for x, y in reversed(path):  # from goal to start, as in the original
        ax.plot(x, y, 'ob', linewidth=2, markersize=10)
        plt.pause(0.05)
    xs, ys = zip(*path)
    ax.plot(xs, ys, 'b', linewidth=2)


# ------------------------------------------------------------------------
# Main program
# ------------------------------------------------------------------------
def main():
    grid = np.ones((NR, NC))
    fig, ax = create_interface(NR, NC)

    choose_obstacles(fig, ax, grid)
    start = choose_point(fig, ax, 'Place the robot at the Start Point with the left button.',
                         'b', 'Start', -0.25)
    goal = choose_point(fig, ax, 'Choose the location of the Goal Point with the left mouse button.',
                        'r', 'Goal', -0.25)
    grid[goal[1], goal[0]] = 1  # the goal is never an obstacle (free weight)

    if SHOW_NUMBERS:
        ax.text(start[0] - 0.25, start[1] - 0.25, f'{0:.2f}')
        ax.set_xlabel('Press a key to advance the algorithm.')
    else:
        ax.set_xlabel('Analyzing the path.')

    path, cost = a_star(grid, start, goal, on_event=create_visualizer(fig, ax))

    if path is None:
        print('All cells were analyzed. There is no path.')
    else:
        draw_path(ax, path, cost)

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
