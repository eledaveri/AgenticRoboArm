import sys
import os

# Setup del path per trovare la cartella 'src'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

# Import environment and agents
from src import PlanarArm2DOF
from src.obstacle import make_rect, make_circle, make_polygon
from src.arm_env  import ArmNavigationEnv

from src.agents import QLearning2DOF, DQL, PPO, SAC

from src import plot_workspace_path, animate_training_path


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


def train_agent(agent_class, env, agent_name, num_episodes=5000, **kwargs):
    """Train an agent and measure performance"""
    print(f"\n{'='*60}")
    print(f"Training {agent_name}")
    print(f"{'='*60}")
    
    agent = agent_class(env, **kwargs)
    
    start_time = time.time()
    agent.train(num_episodes=num_episodes, verbose=True)
    training_time = time.time() - start_time
    
    path = agent.get_path()
    
    results = {
        'agent': agent_name,
        'num_episodes': num_episodes,
        'training_time': training_time,
        'episode_rewards': agent.episode_rewards,
        'episode_success': agent.episode_success,
        'path_length': len(path),
        'final_success_rate': 100.0 * np.sum(agent.episode_success) / num_episodes,
        'average_reward': np.mean(agent.episode_rewards[-500:]),
        'path': path
    }
    
    return results, agent


def plot_comparison(results_dict, output_dir='./results'):
    """Plot comparison of all agents"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    for name, results in results_dict.items():
        # Smooth rewards
        rewards = results['episode_rewards']
        smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
        ax.plot(smoothed, label=name, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (smoothed)')
    ax.set_title('Training Rewards Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Success Rate Over Time
    ax = axes[0, 1]
    for name, results in results_dict.items():
        success = results['episode_success']
        window_size = 100
        success_rate = np.convolve(success, np.ones(window_size)/window_size, mode='valid')
        ax.plot(success_rate * 100, label=name, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # 3. Final Performance Comparison
    ax = axes[1, 0]
    agents = list(results_dict.keys())
    final_success = [results_dict[a]['final_success_rate'] for a in agents]
    avg_reward = [results_dict[a]['average_reward'] for a in agents]
    
    x = np.arange(len(agents))
    width = 0.35
    
    ax.bar(x - width/2, final_success, width, label='Success Rate (%)', color='skyblue')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, avg_reward, width, label='Avg Reward (last 500)', color='orange')
    
    ax.set_xlabel('Agent')
    ax.set_ylabel('Success Rate (%)', color='skyblue')
    ax2.set_ylabel('Average Reward', color='orange')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # 4. Training Time vs Path Length
    ax = axes[1, 1]
    training_times = [results_dict[a]['training_time'] for a in agents]
    path_lengths = [results_dict[a]['path_length'] for a in agents]
    
    ax.scatter(training_times, path_lengths, s=200, alpha=0.6)
    for i, agent in enumerate(agents):
        ax.annotate(agent, (training_times[i], path_lengths[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Training Time (s)')
    ax.set_ylabel('Final Path Length (steps)')
    ax.set_title('Training Efficiency')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/agents_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_dir}/agents_comparison.png")
    plt.show()


def save_results(results_dict, output_dir='./results'):
    """Save detailed results to JSON"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    summary = {}
    for agent_name, results in results_dict.items():
        summary[agent_name] = {
            'final_success_rate': results['final_success_rate'],
            'average_reward': results['average_reward'],
            'path_length': results['path_length'],
            'training_time': results['training_time'],
            'episodes': results['num_episodes']
        }
    
    with open(f'{output_dir}/results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_dir}/results_summary.json")


def main():
    print("="*60)
    print("Multi-Agent RL Comparison for 2-DOF Arm Navigation")
    print("="*60)
    
    # Configuration
    N_DISCRETIZATION = 150
    NUM_EPISODES = 7500
    
    # Create environment components
    arm, obstacles = create_environment()
    
    # Create Gymnasium environment
    env = ArmNavigationEnv(
        arm=arm,
        theta1_range=(0, 2*np.pi),
        theta2_range=(0, 2*np.pi),
        n_discretization=N_DISCRETIZATION,
        obstacles=obstacles,
        start=(5, 5),
        goal=(130, 130),
        continuous=False
    )
    
    # Train all agents
    results = {}
    agents_trained = {}
    
    # 1. Q-Learning
    try:
        print("\n[1/4] Q-Learning")
        results['Q-Learning'], agents_trained['Q-Learning'] = train_agent(
            QLearning, env, 'Q-Learning',
            num_episodes=NUM_EPISODES,
            alpha=0.1, gamma=0.95, epsilon=0.9
        )
    except Exception as e:
        print(f"Error training Q-Learning: {e}")
    
    # 2. DQL (Deep Q-Learning)
    try:
        print("\n[2/4] Deep Q-Learning (DQL)")
        results['DQL'], agents_trained['DQL'] = train_agent(
            DQL, env, 'Deep Q-Learning',
            num_episodes=NUM_EPISODES,
            learning_rate=0.001, gamma=0.95, epsilon=0.9,
            memory_size=10000, batch_size=32
        )
    except Exception as e:
        print(f"Error training DQL: {e}")
    
    # 3. PPO (Proximal Policy Optimization)
    try:
        print("\n[3/4] Proximal Policy Optimization (PPO)")
        results['PPO'], agents_trained['PPO'] = train_agent(
            PPO, env, 'PPO',
            num_episodes=NUM_EPISODES,
            learning_rate=0.001, gamma=0.95, gae_lambda=0.95,
            clip_ratio=0.2, epochs=5
        )
    except Exception as e:
        print(f"Error training PPO: {e}")
    
    # 4. SAC (Soft Actor-Critic)
    try:
        print("\n[4/4] Soft Actor-Critic (SAC)")
        results['SAC'], agents_trained['SAC'] = train_agent(
            SAC, env, 'SAC',
            num_episodes=NUM_EPISODES,
            actor_lr=0.001, q_lr=0.001, alpha_lr=0.001,
            gamma=0.95, tau=0.005, alpha=0.2
        )
    except Exception as e:
        print(f"Error training SAC: {e}")
    
    # Analysis and visualization
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for agent_name in sorted(results.keys()):
        res = results[agent_name]
        print(f"\n{agent_name}:")
        print(f"  - Final Success Rate: {res['final_success_rate']:.1f}%")
        print(f"  - Average Reward (last 500 ep): {res['average_reward']:.2f}")
        print(f"  - Final Path Length: {res['path_length']} steps")
        print(f"  - Training Time: {res['training_time']:.2f} seconds")
    
    # Create comparison plots
    plot_comparison(results)
    
    # Save results
    save_results(results)
    
    # Generate paths visualization for best agents
    print("\nGenerating path visualizations...")
    for agent_name in ['Q-Learning', 'DQL', 'PPO', 'SAC']:
        if agent_name in results:
            agent = agents_trained[agent_name]
            path = results[agent_name]['path']
            
            filename = f"./results/workspace_path_{agent_name.lower().replace('-', '')}.png"
            plot_workspace_path(arm, path, env, start=(5, 5), goal=(130, 130), filename=filename)
            
            print(f"  - Saved path visualization for {agent_name}")
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == "__main__":
    main()