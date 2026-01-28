import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

class DQNNetwork(nn.Module):
    """Deep Q-Network with 2 hidden layers"""
    
    def __init__(self, state_size=2, action_size=4, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQL:
    """Deep Q-Learning agent"""
    
    def __init__(self, env, learning_rate=0.001, gamma=0.95, epsilon=0.9, 
                 memory_size=10000, batch_size=32, device=None):
        """
        Args:
            env: Gymnasium environment
            learning_rate: optimizer learning rate
            gamma: discount factor
            epsilon: initial exploration rate
            memory_size: replay buffer size
            batch_size: batch size for training
            device: 'cuda' or 'cpu'
        """
        self.env = env
        self.gamma = gamma
        self.epsilon_initial = epsilon
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.batch_size = batch_size
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.q_network = DQNNetwork(state_size=2, action_size=env.action_space.n).to(self.device)
        self.target_network = DQNNetwork(state_size=2, action_size=env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Statistics
        self.episode_rewards = []
        self.episode_success = []
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax(1).item()
    
    def replay(self):
        """Train on batch from replay buffer"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Loss and optimization
        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(self, num_episodes=5000, max_steps=500, update_target_freq=500, verbose=True):
        """Training loop"""
        epsilon_decay = (self.epsilon_min / self.epsilon_initial) ** (1.0 / num_episodes)
        steps_total = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            success = False
            
            for step in range(max_steps):
                action = self.choose_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                self.remember(state, action, reward, next_state, terminated)
                self.replay()
                
                state = next_state
                steps_total += 1
                
                if steps_total % update_target_freq == 0:
                    self.update_target_network()
                
                if terminated:
                    success = (reward > 0)
                    break
            
            self.episode_rewards.append(episode_reward)
            self.episode_success.append(success)
            
            if verbose and (episode < 10 or episode % 100 == 0):
                success_rate = 100.0 * np.sum(self.episode_success) / (episode + 1)
                print(f"Episode {episode}/{num_episodes}: reward={episode_reward:.2f}, "
                      f"epsilon={self.epsilon:.3f}, success_rate={success_rate:.1f}%")
            
            self.epsilon = max(self.epsilon_min, self.epsilon * epsilon_decay)
        
        success_rate = 100.0 * np.sum(self.episode_success) / num_episodes
        print(f"\nTraining completed. Final success rate: {success_rate:.1f}%")
    
    def get_path(self, max_steps=1000):
        """Extract path using trained network"""
        state, _ = self.env.reset()
        path = [state]
        visited = {tuple(state)}
        
        for step in range(max_steps):
            action = self.choose_action(state, training=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            state_tuple = tuple(next_state)
            if state_tuple in visited:
                print(f"WARNING: Loop detected at step {step}")
                break
            
            path.append(next_state)
            visited.add(state_tuple)
            state = next_state
            
            if terminated:
                print(f"Goal reached in {len(path)} steps!")
                break
        
        return path