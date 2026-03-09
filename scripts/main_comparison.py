import sys
import os

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from scipy.ndimage import label

# Import environment and agents
from src import PlanarArm2DOF
from src.obstacle import make_rect, make_circle, make_polygon
from src.arm_env  import ArmNavigationEnv
from src.cspace import ConfigurationSpace
from src.agents import QLearning, DQL, PPO, SAC

# Import viz functions
from src.visualize import plot_workspace_path, animate_training_path, plot_cspace_components

def create_environment():
    """Create arm and obstacles"""
    arm = PlanarArm2DOF([1.0, 1.0])
    obstacles = [
        make_rect(0.6, 0.7, 0.2, 0.3),
        make_rect(0.8, 0.9, 0.8, 0.9),
        make_circle(-0.2, 0.5, 0.1),
        make_polygon([(-0.6, -0.6), (-0.1, -0.9), (-0.4, -0.2)])
    ]
    return arm, obstacles

def find_valid_points(cspace):
    """
    Auto-configure Start and Goal points in the largest free component of the C-Space.
    This ensures that both points are in a valid, connected region, increasing the chances of successful training for all agents.
    """
    print("\n[Auto-Config] Cercando punti validi nel C-Space...")
    
    # 1. Label the connected components (0 = free, 1 = obstacle)
    free_space_mask = (cspace.grid == 0)
    labeled_grid, num_features = label(free_space_mask)
    
    if num_features == 0:
        raise ValueError("ERROR: There is no free space! Remove some obstacles.")

    # 2. Find the largest component (the one with the most free cells)
    component_sizes = [np.sum(labeled_grid == i) for i in range(1, num_features + 1)]
    largest_component_label = np.argmax(component_sizes) + 1
    
    print(f"  - Found {num_features} separate components.")
    print(f"  - Selected the largest component (Label #{largest_component_label}) with {max(component_sizes)} free cells.")

    # 3. Get all coordinates (i, j) that belong to this component
    valid_indices = np.argwhere(labeled_grid == largest_component_label)
    
    # 4. Choose Start and Goal points
    np.random.shuffle(valid_indices)
    
    start_idx = tuple(valid_indices[0]) 
    goal_idx = tuple(valid_indices[-1]) 
    
    # Refine goal_idx to ensure it's sufficiently far from start_idx
    for candidate in valid_indices:
        dist = np.abs(candidate[0] - start_idx[0]) + np.abs(candidate[1] - start_idx[1])
        if dist > cspace.N1 // 2: 
            goal_idx = tuple(candidate)
            break
            
    print(f"  - Start Point: {start_idx} (theta: {cspace.theta1_vals[start_idx[0]]:.2f}, {cspace.theta2_vals[start_idx[1]]:.2f})")
    print(f"  - Goal Point : {goal_idx} (theta: {cspace.theta1_vals[goal_idx[0]]:.2f}, {cspace.theta2_vals[goal_idx[1]]:.2f})")
    
    return start_idx, goal_idx

def train_agent(agent_class, env, agent_name, num_episodes=2000, **kwargs):
    print(f"\n{'='*60}\nTraining {agent_name} for {num_episodes} episodes\n{'='*60}")
    agent = agent_class(env, **kwargs)
    
    start_time = time.time()
    agent.train(num_episodes=num_episodes, verbose=True)
    training_time = time.time() - start_time
    
    path = agent.get_path()
    
    # Compute final success rate and average reward
    final_succ = 0.0
    if len(agent.episode_success) > 0:
        final_succ = 100.0 * np.sum(agent.episode_success) / num_episodes
        
    avg_rew = 0.0
    if len(agent.episode_rewards) > 0:
        avg_rew = np.mean(agent.episode_rewards[-100:])

    results = {
        'final_success_rate': final_succ,
        'average_reward': avg_rew,
        'path_length': len(path),
        'training_time': training_time,
        'path': path,
        'episode_rewards': agent.episode_rewards,
        'episode_success': agent.episode_success,
        'trained_episodes': num_episodes #
    }
    return results, agent

def plot_comparison(results_dict, output_dir='./results'):
    """Comparison plot for rewards"""
    plt.figure(figsize=(10, 5))
    for name, res in results_dict.items():
        rewards = res['episode_rewards']
        if len(rewards) > 50:
            smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
            plt.plot(smoothed, label=name)
        else:
            plt.plot(rewards, label=name)
            
    plt.title("Reward Comparison (Smoothed)")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/comparison_reward.png')
    plt.close()

def main():
    print("="*60)
    print("Multi-Agent RL Comparison (Smart Start/Goal)")
    print("="*60)
    
    # Configuration
    N_DISCRETIZATION = 100 
    
    # Dictionary to specify different episode counts for each agent based on their learning characteristics:
    EPISODES = {
        'Q-Learning': 30000,  # 5000
        'SAC': 30000,          # 15000
        'DQL': 30000,          # 15000
        'PPO': 30000           # 15000
    }

    import os
    os.makedirs("./results", exist_ok=True)

    # 1. Create Arm and Obstacles
    arm, obstacles = create_environment()
    
    # 2. Build C-Space 
    print("Building C-Space...")
    cspace = ConfigurationSpace(
        arm, (0, 2*np.pi), (0, 2*np.pi), 
        N_DISCRETIZATION, N_DISCRETIZATION, obstacles
    )
    cspace.build()
    
    # 3. Find valid Start and Goal points
    start_state, goal_state = find_valid_points(cspace)
    plot_cspace_components(
        cspace, 
        start=start_state, 
        goal=goal_state, 
        filename="./results/cspace_connectivity.png"
    )
    
    # 4. Initialize Environment
    env = ArmNavigationEnv(
        arm=arm,
        theta1_range=(0, 2*np.pi),
        theta2_range=(0, 2*np.pi),
        n_discretization=N_DISCRETIZATION,
        obstacles=obstacles,
        start=start_state,
        goal=goal_state,
        continuous=False
    )

    results = {}
    
    global_start_time = time.time()

    # 5. Training Loop
    agent_list = [
        (SAC, 'SAC', {'actor_lr': 0.001}),
        (QLearning, 'Q-Learning', {'alpha': 0.1, 'gamma': 0.99}),
        (DQL, 'DQL', {'learning_rate': 0.001}),
        (PPO, 'PPO', {'learning_rate': 0.001})
    ]

    for agent_class, name, params in agent_list:
        try:
            n_episodes = EPISODES.get(name, 5000) # If not specified default to 5000
            res, _ = train_agent(agent_class, env, name, num_episodes=n_episodes, **params)
            results[name] = res
        except Exception as e:
            print(f"{name} Error: {e}")

    end_time = time.time()
    training_duration = end_time - global_start_time

    # Create summary of results and save to JSON
    for name, res in results.items():
        visited = set()
        has_loop = False
        for pos in res['path']:
            p_tuple = tuple(np.round(pos, 2))
            if p_tuple in visited:
                has_loop = True
                break
            visited.add(p_tuple)

        last_state = tuple(np.round(res['path'][-1]).astype(int))
        
        summary = {
            "agent_name": name,
            "episodes": res.get('trained_episodes', EPISODES.get(name, 0)), 
            "final_success_rate": float(res['final_success_rate']),
            "average_reward": float(res['average_reward']),
            "path_length": res['path_length'],
            "training_time_seconds": round(training_duration, 2), 
            "goal_reached": bool(last_state == goal_state), 
            "has_loops_in_final_path": bool(has_loop)
        }

        print(f"\n--- SUMMARY: {name} ---")
        print(json.dumps(summary, indent=4))

        filename = f"./results/summary_{name}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=4)
            
        print(f"Summary saved correctly in {filename}")

    print("\n" + "="*60)
    print("Generating Visualizations...")
    print("="*60)
    
    # Final comparison plot for rewards
    plot_comparison(results)
    
    for name, res in results.items():
        print(f"Generating plots for {name}...")
        path = res['path']
        
        # Static workspace plot
        plot_workspace_path(
            arm, path, env, 
            start=start_state, goal=goal_state, 
            filename=f"./results/path_{name}.png"
        )
        
        # Animation of the training path
        if len(path) > 1:
            animate_training_path(
                arm, path, env, obstacles, 
                start=start_state, goal=goal_state, 
                filename=f"./results/anim_{name}_{EPISODES.get(name)}.gif"
            )

    print("\n" + "="*60)
    print("Finished! All results and visualizations saved in the './results' directory.")
    print("="*60)

if __name__ == "__main__":
    main()