import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon
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
    plt.close()

def plot_cspace_components(cspace, start=None, goal=None, filename="cspace_components.png"):
    """Plot C-space with connected components of free space highlighted."""
    grid = cspace.grid.copy()
    free_space = (grid == 0)
    labeled, num_components = label(free_space)
    
    # Check connectivity
    if start and goal:
        start_comp = labeled[start[0], start[1]]
        goal_comp = labeled[goal[0], goal[1]]
        print(f"\n[C-Space] Start Comp: {start_comp}, Goal Comp: {goal_comp}")
        if start_comp != goal_comp or start_comp == 0:
            print(" -> WARNING: Start and Goal are NOT connected!")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Background (Obstacles)
    ax.imshow(grid.T, origin="lower", cmap=mcolors.ListedColormap(['white', 'black']), 
              extent=[cspace.theta1_vals[0], cspace.theta1_vals[-1],
                      cspace.theta2_vals[0], cspace.theta2_vals[-1]], aspect='auto', alpha=0.3)

    # Components
    masked_data = np.ma.masked_where(grid.T == 1, labeled.T)
    im = ax.imshow(
        masked_data,
        origin="lower",
        extent=[cspace.theta1_vals[0], cspace.theta1_vals[-1],
                cspace.theta2_vals[0], cspace.theta2_vals[-1]],
        cmap="tab20",
        aspect="auto",
        alpha=0.8
    )
    
    if start:
        theta1_s = cspace.theta1_vals[start[0]]
        theta2_s = cspace.theta2_vals[start[1]]
        ax.scatter(theta1_s, theta2_s, c='lime', s=200, marker='*', edgecolors='k', label='Start', zorder=10)
    
    if goal:
        theta1_g = cspace.theta1_vals[goal[0]]
        theta2_g = cspace.theta2_vals[goal[1]]
        ax.scatter(theta1_g, theta2_g, c='red', s=200, marker='*', edgecolors='k', label='Goal', zorder=10)

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(f"C-Space Connectivity ({num_components} components)")
    plt.colorbar(im, ax=ax, label="Component ID")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_workspace(arm, theta1, theta2, obstacles, start=None, goal=None, cspace=None, filename="workspace.png"):
    """Plot workspace with arm configuration and obstacles."""
    fig, ax = plt.subplots(figsize=(6,6))

    # Arm segments
    segments = arm.get_segments(theta1, theta2)
    for (p0, p1) in segments:
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], "bo-", linewidth=3)

    # Obstacles (Fixed for Shapely)
    for obs in obstacles:
        if isinstance(obs, ShapelyPolygon):
            x, y = obs.exterior.xy
            ax.fill(x, y, color="red", alpha=0.4)

    # Start/Goal
    if start is not None and cspace is not None:
        start_pos = arm.forward_kinematics(cspace.theta1_vals[start[0]], cspace.theta2_vals[start[1]])
        ax.scatter(*start_pos, color='green', s=100, label='Start')

    if goal is not None and cspace is not None:
        goal_pos = arm.forward_kinematics(cspace.theta1_vals[goal[0]], cspace.theta2_vals[goal[1]])
        ax.scatter(*goal_pos, color='blue', s=100, label='Goal')

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Workspace")
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_cspace_path(cspace, path, filename="cspace_path.png"):
    """Plot the discretized C-space with a path overlaid."""
    cmap = mcolors.ListedColormap(["white", "blue"])
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

    if len(path) > 0:
        theta1_path = [cspace.theta1_vals[i] for i, j in path]
        theta2_path = [cspace.theta2_vals[j] for i, j in path]
        plt.plot(theta1_path, theta2_path, color="red", marker="o", markersize=2, linewidth=1, label="Path")

    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title("C-Space Path")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_workspace_path(arm, path, env_or_cspace, start=None, goal=None, filename="workspace_path.png"):
    """Plot path in workspace with obstacles correctly drawn."""
    theta1_vals = env_or_cspace.theta1_vals
    theta2_vals = env_or_cspace.theta2_vals
    obstacles = env_or_cspace.obstacles

    plt.figure(figsize=(6,6))
    ax = plt.gca()

    # Draw Obstacles (CORRECTED)
    for obs in obstacles:
        if isinstance(obs, ShapelyPolygon):
            x, y = obs.exterior.xy
            ax.fill(x, y, color='red', alpha=0.5)

    # Draw Path
    x_path, y_path = [], []
    for i, j in path:
        th1 = theta1_vals[i]
        th2 = theta2_vals[j]
        pos = arm.forward_kinematics(th1, th2)
        x_path.append(pos[0])
        y_path.append(pos[1])

    plt.plot(x_path, y_path, 'b-o', markersize=2, linewidth=1, label="Path", alpha=0.7)

    if start:
        s_pos = arm.forward_kinematics(theta1_vals[start[0]], theta2_vals[start[1]])
        plt.scatter(*s_pos, color='lime', s=150, marker='*', edgecolors='k', label='Start', zorder=5)
    
    if goal:
        g_pos = arm.forward_kinematics(theta1_vals[goal[0]], theta2_vals[goal[1]])
        plt.scatter(*g_pos, color='red', s=150, marker='*', edgecolors='k', label='Goal', zorder=5)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Workspace Path")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def animate_training_path(arm, path, env_or_cspace, obstacles, start, goal, filename="training_animation.gif"):
    """Create GIF animation of the arm moving."""
    theta1_vals = env_or_cspace.theta1_vals
    theta2_vals = env_or_cspace.theta2_vals
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    for obs in obstacles:
        if isinstance(obs, ShapelyPolygon):
            x, y = obs.exterior.xy
            ax.fill(x, y, color='red', alpha=0.5)

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title("Robot Arm Animation")

    line_arm, = ax.plot([], [], 'ko-', linewidth=3, markersize=5)
    line_trail, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5)
    
    trail_x, trail_y = [], []

    def init():
        line_arm.set_data([], [])
        line_trail.set_data([], [])
        return line_arm, line_trail

    def update(frame):
        idx = frame
        if idx >= len(path): idx = len(path) - 1
        
        i, j = path[idx]
        th1 = theta1_vals[i]
        th2 = theta2_vals[j]

        segments = arm.get_segments(th1, th2)
        xs = [segments[0][0][0], segments[0][1][0], segments[1][1][0]]
        ys = [segments[0][0][1], segments[0][1][1], segments[1][1][1]]
        
        line_arm.set_data(xs, ys)

        trail_x.append(xs[-1])
        trail_y.append(ys[-1])
        line_trail.set_data(trail_x, trail_y)

        return line_arm, line_trail

    frames = len(path)
    step = 1
    if frames > 200:
        step = frames // 200
        frames = 200
        
    anim = FuncAnimation(fig, update, frames=range(0, len(path), step), 
                        init_func=init, blit=True, interval=50)
    
    anim.save(filename, writer=PillowWriter(fps=20))
    print(f"Animation saved to {filename}")
    plt.close()