"""
Adattamento del main.py originale per il nuovo sistema Gymnasium.
Mantiene la stessa logica ma usa il nuovo ambiente e agenti.
"""

from src.arm import PlanarArm2DOF
from src.obstacle import make_rect, make_circle, make_polygon
from src.arm_env import ArmNavigationEnv
from src.agents.qlearning import QLearning2DOF
from src.visualize import plot_cspace, plot_cspace_components, plot_workspace, \
    plot_workspace_path, animate_training_path
import numpy as np


def main():
    print("Initializing 2-DOF Arm Navigation with Gymnasium...")
    
    # Create arm
    arm = PlanarArm2DOF([1.0, 1.0])
    
    # Define obstacles
    obstacles = [
        make_rect(0.6, 0.7, 0.2, 0.3),
        make_rect(0.8, 0.9, 0.8, 0.9),
        make_circle(-0.2, 0.5, 0.1),
        make_polygon([(-0.6, -0.6), (-0.1, -0.9), (-0.4, -0.2)])
    ]
    
    # Start and goal positions in discretized C-space
    start = (5, 5)
    goal = (130, 130)
    
    # Create Gymnasium environment (replaces old ConfigurationSpace)
    print("\nCreating Gymnasium environment...")
    env = ArmNavigationEnv(
        arm=arm,
        theta1_range=(0, 2*np.pi),
        theta2_range=(0, 2*np.pi),
        n_discretization=150,
        obstacles=obstacles,
        start=start,
        goal=goal,
        continuous=False  # Use discrete actions like original
    )
    
    # Get theta values for visualization
    theta1_start = env.theta1_vals[start[0]]
    theta2_start = env.theta2_vals[start[1]]
    
    # Plot C-space (original visualization)
    print("Plotting C-space...")
    from cspace import ConfigurationSpace
    old_cspace = ConfigurationSpace(
        arm=arm,
        theta1_range=(0, 2*np.pi),
        theta2_range=(0, 2*np.pi),
        N1=150, N2=150,
        obstacles=obstacles
    )
    old_cspace.build()
    plot_cspace(old_cspace)
    plot_cspace_components(old_cspace, start=start, goal=goal, filename="cspace_components.png")
    plot_workspace(arm, theta1_start, theta2_start, obstacles, 
                  start=start, goal=goal, cspace=old_cspace, 
                  filename="workspace_periodical_theta.png")
    
    # Train Q-learning agent using Gymnasium
    print("\nTraining Q-Learning agent...")
    ql = QLearning2DOF(
        env,
        alpha=0.1,        # Learning rate
        gamma=0.95,       # Discount factor
        epsilon=0.9       # Initial exploration rate
    )
    
    ql.train(
        num_episodes=7500,
        max_steps=500,
        verbose=True
    )
    
    # Get learned path
    path = ql.get_path()
    print(f"Learned path: {len(path)} steps")
    
    # Animate training
    print("Creating animation...")
    animate_training_path(arm, path, old_cspace, obstacles, 
                         start=start, goal=goal, filename="training_periodical_theta.gif")
    
    # Plot workspace with path
    print("Plotting workspace with path...")
    plot_workspace_path(arm, path, old_cspace, start=start, goal=goal, 
                       filename="workspace_periodical_theta_path.png")
    
    print("\nTraining completed!")
    print(f"Final success rate: {100.0 * np.sum(ql.episode_success) / len(ql.episode_success):.1f}%")


if __name__ == "__main__":
    main()