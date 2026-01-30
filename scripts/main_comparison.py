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
    Trova automaticamente due punti (Start e Goal) che appartengono
    alla stessa componente connessa (sono raggiungibili).
    """
    print("\n[Auto-Config] Cercando punti validi nel C-Space...")
    
    # 1. Etichetta le componenti connesse (0 = libero, 1 = ostacolo)
    # label() vuole 0 come sfondo e numeri interi come oggetti.
    # Invertiamo: vogliamo etichettare lo SPAZIO LIBERO (che è 0 nella grid).
    free_space_mask = (cspace.grid == 0)
    labeled_grid, num_features = label(free_space_mask)
    
    if num_features == 0:
        raise ValueError("ERRORE: Non c'è spazio libero! Rimuovi qualche ostacolo.")

    # 2. Trova la componente più grande (quella con più celle libere)
    component_sizes = [np.sum(labeled_grid == i) for i in range(1, num_features + 1)]
    largest_component_label = np.argmax(component_sizes) + 1
    
    print(f"  - Trovate {num_features} isole separate.")
    print(f"  - Selezionata l'isola più grande (Label #{largest_component_label}) con {max(component_sizes)} celle.")

    # 3. Prendi tutte le coordinate (i, j) che appartengono a questa isola
    valid_indices = np.argwhere(labeled_grid == largest_component_label)
    
    # 4. Scegli Start e Goal
    # Prendiamo due punti a caso, ma cerchiamo di averli distanti
    np.random.shuffle(valid_indices)
    
    start_idx = tuple(valid_indices[0]) # Primo punto a caso
    
    # Cerchiamo un goal che sia distante almeno un po' (es. 50 step di grid)
    goal_idx = tuple(valid_indices[-1]) # Ultimo punto (spesso lontano dopo lo shuffle, ma non garantito)
    
    # Raffinamento: scorri finché non trovi un punto lontano
    for candidate in valid_indices:
        dist = np.abs(candidate[0] - start_idx[0]) + np.abs(candidate[1] - start_idx[1])
        if dist > cspace.N1 // 2: # Almeno metà griglia di distanza
            goal_idx = tuple(candidate)
            break
            
    print(f"  - Start Point: {start_idx} (theta: {cspace.theta1_vals[start_idx[0]]:.2f}, {cspace.theta2_vals[start_idx[1]]:.2f})")
    print(f"  - Goal Point : {goal_idx} (theta: {cspace.theta1_vals[goal_idx[0]]:.2f}, {cspace.theta2_vals[goal_idx[1]]:.2f})")
    
    return start_idx, goal_idx

def train_agent(agent_class, env, agent_name, num_episodes=2000, **kwargs):
    print(f"\n{'='*60}\nTraining {agent_name}\n{'='*60}")
    agent = agent_class(env, **kwargs)
    
    start_time = time.time()
    agent.train(num_episodes=num_episodes, verbose=True)
    training_time = time.time() - start_time
    
    path = agent.get_path()
    
    # Calcolo statistiche sicure (evita errori su liste vuote)
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
        'episode_success': agent.episode_success
    }
    return results, agent

def plot_comparison(results_dict, output_dir='./results'):
    """Plot comparativo semplice"""
    plt.figure(figsize=(10, 5))
    for name, res in results_dict.items():
        # Smoothing reward
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
    
    # Configurazione
    N_DISCRETIZATION = 100 
    NUM_EPISODES = 20000    
    
    # Setup Cartelle
    import os
    os.makedirs("./results", exist_ok=True)

    # 1. Crea Mondo
    arm, obstacles = create_environment()
    
    # 2. Costruisci C-Space per analizzare la connettività
    print("Building C-Space...")
    cspace = ConfigurationSpace(
        arm, (0, 2*np.pi), (0, 2*np.pi), 
        N_DISCRETIZATION, N_DISCRETIZATION, obstacles
    )
    cspace.build()
    
    # 3. TROVA START/GOAL VALIDI AUTOMATICAMENTE
    start_state, goal_state = find_valid_points(cspace)
    
    # Salva l'immagine del C-Space con Start e Goal evidenziati (e Legenda)
    plot_cspace_components(
        cspace, 
        start=start_state, 
        goal=goal_state, 
        filename="./results/cspace_connectivity.png"
    )

    # 4. Inizializza l'ambiente con le coordinate sicure
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
    
    # 5. Training Loop
    # Q-Learning
    try:
        results['Q-Learning'], _ = train_agent(
            QLearning, env, 'Q-Learning', num_episodes=NUM_EPISODES, 
            alpha=0.1, gamma=0.99, epsilon=0.9
        )
    except Exception as e: print(f"Q-Learning Error: {e}")

    # SAC
    try:
        results['SAC'], _ = train_agent(
            SAC, env, 'SAC', num_episodes=NUM_EPISODES,
            actor_lr=0.001, q_lr=0.001, alpha_lr=0.001
        )
    except Exception as e: print(f"SAC Error: {e}")
    
    # DQL
    try:
        results['DQL'], _ = train_agent(
            DQL, env, 'DQL', num_episodes=NUM_EPISODES,
            learning_rate=0.001, gamma=0.95
        )
    except Exception as e: print(f"DQL Error: {e}")

    # PPO
    try:
        results['PPO'], _ = train_agent(
            PPO, env, 'PPO', num_episodes=NUM_EPISODES,
            learning_rate=0.001
        )
    except Exception as e: print(f"PPO Error: {e}")

    # 6. Salvataggio Risultati e Grafici
    plot_comparison(results)
    
    for name, res in results.items():
        print(f"\nGenerating plots for {name}...")
        path = res['path']
        
        # Plot Workspace Statico
        plot_workspace_path(
            arm, path, env, 
            start=start_state, goal=goal_state, 
            filename=f"./results/path_{name}.png"
        )
        
        # Animazione GIF
        if len(path) > 1:
            animate_training_path(
                arm, path, env, obstacles, 
                start=start_state, goal=goal_state, 
                filename=f"./results/anim_{name}.gif"
            )

    print("\n" + "="*60)
    print("Finito! Controlla la cartella /results")
    print("="*60)

if __name__ == "__main__":
    main()