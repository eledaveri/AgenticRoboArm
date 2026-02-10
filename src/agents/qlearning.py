import numpy as np
import random
from src.arm import PlanarArm2DOF

class QLearning:
    """Q-Learning agent for Gymnasium environment"""
    
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.1, num_episodes=1000, **kwargs):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        
        # Q-table: dictionary mapping (state_tuple, action) -> value
        self.q_table = {} 
        
        self.episode_rewards = []
        self.episode_success = []

    def get_q(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def choose_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        # Greedy action
        q_values = [self.get_q(state, a) for a in range(self.env.action_space.n)]
        max_q = max(q_values)
        # Random choice among ties
        actions_with_max_q = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(actions_with_max_q)

    def train(self, num_episodes=None, verbose=False):
        if num_episodes is None:
            num_episodes = self.num_episodes
            
        min_epsilon = 0.01
        decay_rate = 0.9995

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0
            terminated = False
            truncated = False
            
            if self.epsilon > min_epsilon:
                self.epsilon *= decay_rate

            while not (terminated or truncated):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Q-Learning update
                old_q = self.get_q(state, action)
                next_max = max([self.get_q(next_state, a) for a in range(self.env.action_space.n)])
                
                new_q = old_q + self.alpha * (reward + self.gamma * next_max - old_q)
                self.q_table[(tuple(state), action)] = new_q
                
                state = next_state
                total_reward += reward

            self.episode_rewards.append(total_reward)
            self.episode_success.append(terminated and total_reward > 0)

            if verbose and (episode + 1) % 500 == 0:
                avg_rew = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode+1}: Avg Reward {avg_rew:.2f}")

    # Sostituisci il metodo get_path in src/agents/qlearning.py con questo:

    def get_path(self, max_steps=500):
        path = []
        state, _ = self.env.reset()
        state = tuple(state)
        path.append(state)
        
        for _ in range(max_steps):
            action = self.choose_action(state, training=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state_t = tuple(next_state)
            
            path.append(next_state_t)
            state = next_state_t
            
            if terminated:
                if reward > 0:
                    print(f"Goal reached in {len(path)} steps!")
                else:
                    print(f"Failed (Collision) in {len(path)} steps.")
                break
            
            if truncated:
                print(f"Failed (Truncated) after {len(path)} steps.")
                break
            
        return path