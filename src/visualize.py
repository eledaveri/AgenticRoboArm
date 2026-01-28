import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from scipy.ndimage import label
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

def plot_cspace(cspace, filename="cspace.png"):
    """Plot the configuration space grid"""
    cmap = mcolors.ListedColormap(["white", "Blue"])  # 0=free, 1=obstacle
    bounds = [0, 1, 2]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6,5))
    plt.imshow(
        cspace.grid.T,
        origin="lower",
        extent=[cspace.theta1_vals[0], cspace.theta1_vals[-1],
                cspace.theta2_vals[0], cspace.theta2_vals[-1]],
        cmap=cmap,
        norm=norm,
        aspect="auto"
    )
    plt.xlabel(r"$\theta_1$ (rad)")
    plt.ylabel(r"$\theta_2$ (rad)")
    plt.title("Configuration Space")

    cbar = plt.colorbar(ticks=[0.5, 1.5])
    cbar.ax.set_yticklabels(["Free", "Obstacle"])

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_cspace_components(cspace, start=None, goal=None, filename="cspace_components.png"):
    """
    Plot C-space with connected components of free space.

    Args:
        cspace: ConfigurationSpace object
        start: (i,j) tuple for start state (optional)
        goal: (i,j) tuple for goal state (optional)
        filename: output image file name
    """

    # Copy the grid
    grid = cspace.grid.copy()

    # Find connected components in free space
    free_space = (grid == 0)
    labeled, num_components = label(free_space)

    print(f"Found {num_components} connected components in free space")

    # Component map, -1 for obstacles
    comp_map = labeled.copy()
    comp_map[grid == 1] = -1

    # Number of bins: obstacles (-1) + components (0..num_components-1)
    num_bins = 1 + num_components

    # Colormap: red for obstacles + colors for components
    cmap = plt.cm.get_cmap("tab20", num_components)
    colors = cmap(np.arange(num_components))
    comp_cmap = mcolors.ListedColormap(["red"] + list(colors))

    # Boundaries for BoundaryNorm: center each color on an integer
    bounds = np.arange(-0.5, num_bins + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, comp_cmap.N)

    # Plot
    plt.figure(figsize=(6, 5))
    plt.imshow(
        comp_map.T,  # obstacles = -1, components = 0..num_components-1
        origin="lower",
        extent=[cspace.theta1_vals[0], cspace.theta1_vals[-1],
                cspace.theta2_vals[0], cspace.theta2_vals[-1]],
        cmap=comp_cmap,
        norm=norm,
        aspect="auto"
    )
    
    # Plot start and goal if provided
    if start is not None:
        theta1_start = cspace.theta1_vals[start[0]]
        theta2_start = cspace.theta2_vals[start[1]]
        plt.scatter(theta1_start, theta2_start, color='lime', s=150, 
                   marker='*', edgecolors='black', linewidths=1.5, 
                   label='Start', zorder=5)
    
    if goal is not None:
        theta1_goal = cspace.theta1_vals[goal[0]]
        theta2_goal = cspace.theta2_vals[goal[1]]
        plt.scatter(theta1_goal, theta2_goal, color='yellow', s=150, 
                   marker='*', edgecolors='black', linewidths=1.5, 
                   label='Goal', zorder=5)
    
    if start is not None or goal is not None:
        plt.legend(loc='upper right')
    
    plt.xlabel(r"$\theta_1$ (rad)")
    plt.ylabel(r"$\theta_2$ (rad)")
    plt.title("Configuration Space - Connected Components")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def plot_workspace(arm, theta1, theta2, obstacles, start=None, goal=None, cspace=None, filename="workspace.png"):
    """
    Plot the workspace with the arm in a given configuration and obstacles.
    Optionally, show start and goal positions from the C-space.
    
    Args:
        arm: PlanarArm2DOF object
        theta1, theta2: current joint angles
        obstacles: list of obstacle objects
        start: (i,j) tuple in C-space for start (optional)
        goal: (i,j) tuple in C-space for goal (optional)
        cspace: ConfigurationSpace object (required if start/goal are provided)
        filename: output image file name
    """
    fig, ax = plt.subplots()

    # Arm segments
    segments = arm.get_segments(theta1, theta2)
    for (p0, p1) in segments:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "bo-", linewidth=3)

    # Obstacles
    for obs in obstacles:
        if hasattr(obs, "bounds"):  # shapely polygon
            patch = Polygon(list(obs.exterior.coords), facecolor="red", alpha=0.4)
            ax.add_patch(patch)

    # Plot start and goal positions if provided
    if start is not None and cspace is not None:
        start_pos = arm.forward_kinematics(cspace.theta1_vals[start[0]], cspace.theta2_vals[start[1]])
        ax.scatter(*start_pos, color='green', s=100, label='Start')

    if goal is not None and cspace is not None:
        goal_pos = arm.forward_kinematics(cspace.theta1_vals[goal[0]], cspace.theta2_vals[goal[1]])
        ax.scatter(*goal_pos, color='blue', s=100, label='Goal')

    if (start is not None and cspace is not None) or (goal is not None and cspace is not None):
        ax.legend()

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Workspace: robot + obstacles")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_cspace_path(cspace, path, filename="cspace_path.png"):
    """
    Plot the discretized C-space with a path overlaid.

    Args:
        cspace: ConfigurationSpace object
        path: list of (i,j) states representing the path
        filename: output image file name
    """
    # Base C-space
    cmap = mcolors.ListedColormap(["white", "blue"])  # 0=free, 1=obstacle
    bounds = [0, 1, 2]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6,5))
    plt.imshow(
        cspace.grid.T,
        origin="lower",
        extent=[cspace.theta1_vals[0], cspace.theta1_vals[-1],
                cspace.theta2_vals[0], cspace.theta2_vals[-1]],
        cmap=cmap,
        norm=norm,
        aspect="auto"
    )

    # Convert path indices (i,j) to theta1/theta2 values
    theta1_path = [cspace.theta1_vals[i] for i, j in path]
    theta2_path = [cspace.theta2_vals[j] for i, j in path]

    # Overlay path
    plt.plot(theta1_path, theta2_path, color="red", marker="o", markersize=3, linewidth=2, label="Path")

    plt.xlabel(r"$\theta_1$ (rad)")
    plt.ylabel(r"$\theta_2$ (rad)")
    plt.title("Configuration Space with Q-learning Path")
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_workspace_path(arm, path, cspace, start=None, goal=None, filename="workspace_path.png"):
    """
    Plot the robot trajectory in the workspace (x,y) with highlighted start and goal.
    
    Args:
        arm: PlanarArm2DOF object,
        path: list of (i,j) states representing the path,
        cspace: ConfigurationSpace object,
        start: (i,j) tuple for start state (optional),
        goal: (i,j) tuple for goal state (optional),
        filename: output image file name
    """
    start = start if start else (0, 0)
    goal = goal if goal else (cspace.N1-1, cspace.N2-1)

    plt.figure(figsize=(6,6))
    ax = plt.gca()

    # Draw the obstacles (Shapely Polygons)
    for obs in cspace.obstacles:
        if isinstance(obs, Polygon):
            x, y = obs.exterior.xy
            ax.fill(x, y, color='k', alpha=0.3)

    # Converts the path from (i,j) to θ1, θ2
    theta1_path = [cspace.theta1_vals[i] for i,j in path]
    theta2_path = [cspace.theta2_vals[j] for i,j in path]

    # Compute the coordinates of the end-effector
    x_path = []
    y_path = []
    for th1, th2 in zip(theta1_path, theta2_path):
        pos = arm.forward_kinematics(th1, th2)
        x_path.append(pos[0])
        y_path.append(pos[1])

    # Trajectory
    plt.plot(x_path, y_path, 'r-o', markersize=3, linewidth=2, label="Path")

    # Highlight start and goal
    start_pos = arm.forward_kinematics(cspace.theta1_vals[start[0]], cspace.theta2_vals[start[1]])
    goal_pos = arm.forward_kinematics(cspace.theta1_vals[goal[0]], cspace.theta2_vals[goal[1]])
    plt.scatter(*start_pos, color='green', s=100, label='Start')
    plt.scatter(*goal_pos, color='blue', s=100, label='Goal')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Workspace Path")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

def animate_training_path(arm, path, cspace, obstacles, start, goal, filename="training_animation.gif"):
    """Create an animation of the robot arm moving along the learned path.
    
    Args:
        arm: PlanarArm2DOF object
        path: list of (i,j) states representing the path
        cspace: ConfigurationSpace object
        obstacles: list of obstacle objects
        start: (i,j) tuple for start state
        goal: (i,j) tuple for goal state
        filename: output filename for the animation
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Obstacles
    for obs in obstacles:
        if hasattr(obs, 'exterior'):
            patch = Polygon(list(obs.exterior.coords), facecolor='red', alpha=0.3)
            ax.add_patch(patch)
    
    # Start and goal positions using forward kinematics
    i_s, j_s = start
    i_g, j_g = goal
    start_pos = arm.forward_kinematics(cspace.theta1_vals[i_s], cspace.theta2_vals[j_s])
    goal_pos = arm.forward_kinematics(cspace.theta1_vals[i_g], cspace.theta2_vals[j_g])
    ax.scatter(*start_pos, color='green', s=200, marker='*', label='Start')
    ax.scatter(*goal_pos, color='gold', s=200, marker='*', label='Goal')
    
    # Setup plot
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Robot Arm Motion')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Animated elements
    line1, = ax.plot([], [], 'bo-', linewidth=3, markersize=8)
    line2, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1)
    trail_x, trail_y = [], []
    
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1, line2
    
    def update(frame):
        i, j = path[frame]
        th1 = cspace.theta1_vals[i]
        th2 = cspace.theta2_vals[j]
        
        # Arm
        segments = arm.get_segments(th1, th2)
        x_arm = [0] + [seg[1][0] for seg in segments]
        y_arm = [0] + [seg[1][1] for seg in segments]
        line1.set_data(x_arm, y_arm)
        
        # Trace
        pos = arm.forward_kinematics(th1, th2)
        trail_x.append(pos[0])
        trail_y.append(pos[1])
        line2.set_data(trail_x, trail_y)
        
        ax.set_title(f'Robot Arm Motion - Step {frame}/{len(path)-1}')
        return line1, line2
    
    anim = FuncAnimation(fig, update, frames=len(path), init_func=init,
                        blit=True, interval=100, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=20)
    anim.save(filename, writer=writer)
    print(f"Animation saved in {filename}")
    plt.close()