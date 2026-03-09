import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_size=2, action_size=4, hidden_size=64):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_size, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value


class PPO:
    """Proximal Policy Optimization agent"""
    
    def __init__(self, env, learning_rate=0.0003, gamma=0.95, gae_lambda=0.95,
                 clip_ratio=0.2, epochs=5, device=None):
        """
        Args:
            env: Gymnasium environment
            learning_rate: optimizer learning rate
            gamma: discount factor
            gae_lambda: GAE lambda parameter
            clip_ratio: PPO clipping ratio
            epochs: update epochs per batch
            device: 'cuda' or 'cpu'
        """
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.epochs = epochs
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.policy = ActorCritic(state_size=2, action_size=env.action_space.n).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.episode_rewards = []
        self.episode_success = []
    
    def choose_action(self, state, training=True):
        """Sample action from policy
        Args:
            state: current state
            training: if True, sample from distribution; if False, take argmax
        Returns:
            action: chosen action
            log_prob: log probability of the chosen action
        """
        # Input normalization
        state_norm = np.array(state, dtype=np.float32) / self.env.n_discretization
        state_tensor = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, _ = self.policy(state_tensor)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()

    def compute_gae(self, trajectory):
        """Compute Generalized Advantage Estimation
        Args:
            trajectory: list of (state, action, reward, next_state, done, log_prob) tuples for one episode
            Returns: advantages, returns, old_log_probs tensors"""
        states, actions, rewards, next_states, dones, log_probs = trajectory
        
        # If normalized input:
        states_t = torch.FloatTensor(np.array(states, dtype=np.float32) / self.env.n_discretization).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states, dtype=np.float32) / self.env.n_discretization).to(self.device)
        
        # If not normalized:
        # states_t = torch.FloatTensor(np.array(states)).to(self.device)
        # next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)

        rewards_t = torch.FloatTensor(rewards).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)
        log_probs_t = torch.FloatTensor(log_probs).to(self.device)
        
        with torch.no_grad():
            _, values = self.policy(states_t)
            _, next_values = self.policy(next_states_t)
        
        values = values.view(-1)
        next_values = next_values.view(-1)
        
        advantages = torch.zeros_like(rewards_t)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_values[t]
            else:
                next_val = values[t + 1]
            
            delta = rewards_t[t] + self.gamma * next_val * (1 - dones_t[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones_t[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        
        return advantages.detach(), returns.detach(), log_probs_t
    
    def update(self, trajectory):
        """Update policy using PPO
        Args:
            trajectory: list of (state, action, reward, next_state, done, log_prob) tuples for one episode"""
        advantages, returns, old_log_probs = self.compute_gae(trajectory)
        states = trajectory[0]
        actions = trajectory[1]
        
        # Batch normalization
        states_t = torch.FloatTensor(np.array(states, dtype=np.float32) / self.env.n_discretization).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs = old_log_probs.detach()
        
        # Advantages normalization
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean() # If there's only one advantage no std to divide by
        
        for _ in range(self.epochs):
            action_probs, values = self.policy(states_t)
            values = values.squeeze()
            
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions_t)
            
            # PPO clip ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = 0.5 * ((values - returns) ** 2).mean()
            loss = actor_loss + critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
    
    def train(self, num_episodes=5000, max_steps=500, update_freq=10, verbose=True):
        """Training loop
        Args:
            num_episodes: total number of episodes to train
            max_steps: max steps per episode
            update_freq: how often to update the policy (in episodes)
            verbose: if True, print training progress
        Returns: training results dictionary and the last trajectory"""
        trajectories = []
        trajectory_count = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            success = False
            
            trajectory = [[], [], [], [], [], []]
            
            for step in range(max_steps):
                action, log_prob = self.choose_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                trajectory[0].append(state)
                trajectory[1].append(action)
                trajectory[2].append(reward)
                trajectory[3].append(next_state)
                trajectory[4].append(terminated)
                trajectory[5].append(log_prob)
                
                state = next_state
                
                if terminated:
                    success = (reward > 0)
                    break
            
            trajectories.append(trajectory)
            trajectory_count += 1
            
            if trajectory_count % update_freq == 0:
                for traj in trajectories:
                    self.update(traj)
                trajectories = []
            
            self.episode_rewards.append(episode_reward)
            self.episode_success.append(success)
            
            if verbose and (episode < 10 or episode % 100 == 0):
                success_rate = 100.0 * np.sum(self.episode_success) / (episode + 1)
                print(f"Episode {episode}/{num_episodes}: reward={episode_reward:.2f}, "
                      f"success_rate={success_rate:.1f}%")
        
        success_rate = 100.0 * np.sum(self.episode_success) / num_episodes
        print(f"\nTraining completed. Final success rate: {success_rate:.1f}%")
    
    def get_path(self, max_steps=1000):
        """Extract path using trained policy
        Args:
            max_steps: maximum steps to extract path
        Returns: list of states in the path """
        state, _ = self.env.reset()
        path = [state]
        visited = {tuple(state)}
        
        for step in range(max_steps):
            action, _ = self.choose_action(state, training=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            state_tuple = tuple(next_state)
            if state_tuple in visited:
                print(f"WARNING: Loop detected at step {step}")
                break
            
            path.append(next_state)
            visited.add(state_tuple)
            state = next_state
            
            if terminated:
                # Check if it's a success or failure
                if reward > 0:
                    print(f"Goal reached in {len(path)} steps!")
                else:
                    print(f"Failed (Collision) in {len(path)} steps.")
                break
        
        return path