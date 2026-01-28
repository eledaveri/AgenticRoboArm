import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
from collections import deque


class SoftActorCriticNetwork(nn.Module):
    """Actor and Critic networks for SAC"""
    
    def __init__(self, state_size=2, action_size=4, hidden_size=64):
        super(SoftActorCriticNetwork, self).__init__()
        
        # Actor
        self.actor_fc1 = nn.Linear(state_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_out = nn.Linear(hidden_size, action_size)
        
        # Q-networks
        self.q1_fc1 = nn.Linear(state_size + 1, hidden_size)
        self.q1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q1_out = nn.Linear(hidden_size, 1)
        
        self.q2_fc1 = nn.Linear(state_size + 1, hidden_size)
        self.q2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.q2_out = nn.Linear(hidden_size, 1)
        
        # V-network
        self.v_fc1 = nn.Linear(state_size, hidden_size)
        self.v_fc2 = nn.Linear(hidden_size, hidden_size)
        self.v_out = nn.Linear(hidden_size, 1)
        
        self.relu = nn.ReLU()
    
    def actor(self, state):
        """Policy network"""
        x = self.relu(self.actor_fc1(state))
        x = self.relu(self.actor_fc2(x))
        action_logits = self.actor_out(x)
        return torch.softmax(action_logits, dim=-1)
    
    def q1(self, state, action):
        """Q-function 1"""
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.q1_fc1(x))
        x = self.relu(self.q1_fc2(x))
        return self.q1_out(x)
    
    def q2(self, state, action):
        """Q-function 2"""
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.q2_fc1(x))
        x = self.relu(self.q2_fc2(x))
        return self.q2_out(x)
    
    def value(self, state):
        """Value function"""
        x = self.relu(self.v_fc1(state))
        x = self.relu(self.v_fc2(x))
        return self.v_out(x)


class SAC:
    """Soft Actor-Critic agent"""
    
    def __init__(self, env, actor_lr=0.001, q_lr=0.001, alpha_lr=0.001,
                 gamma=0.95, tau=0.005, alpha=0.2, memory_size=10000,
                 batch_size=32, device=None):
        """
        Args:
            env: Gymnasium environment
            actor_lr: actor learning rate
            q_lr: Q-function learning rate
            alpha_lr: entropy coefficient learning rate
            gamma: discount factor
            tau: target network soft update parameter
            alpha: entropy coefficient
            memory_size: replay buffer size
            batch_size: batch size
            device: 'cuda' or 'cpu'
        """
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.network = SoftActorCriticNetwork(state_size=2, action_size=env.action_space.n).to(self.device)
        self.target_value = SoftActorCriticNetwork(state_size=2, action_size=env.action_space.n).to(self.device)
        self.target_value.load_state_dict(self.network.state_dict())
        
        # Optimizers
        self.actor_opt = optim.Adam(
            [self.network.actor_fc1.weight, self.network.actor_fc1.bias,
             self.network.actor_fc2.weight, self.network.actor_fc2.bias,
             self.network.actor_out.weight, self.network.actor_out.bias],
            lr=actor_lr
        )
        
        self.q_opt = optim.Adam(
            list(self.network.q1.parameters()) + list(self.network.q2.parameters()),
            lr=q_lr
        )
        
        self.value_opt = optim.Adam(self.network.value.parameters(), lr=q_lr)
        
        # Entropy coefficient (learnable)
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -np.log(1.0 / env.action_space.n)
        
        # Replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Statistics
        self.episode_rewards = []
        self.episode_success = []
    
    @property
    def alpha(self):
        return torch.exp(self.log_alpha).item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state, training=True):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.network.actor(state_tensor)
        
        if training:
            dist = Categorical(action_probs)
            action = dist.sample()
        else:
            action = torch.argmax(action_probs, dim=1)
        
        return action.item()
    
    def replay(self):
        """Update networks"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Value update
        with torch.no_grad():
            next_action_probs = self.network.actor(next_states)
            next_dist = Categorical(next_action_probs)
            next_log_probs = next_dist.logits - torch.logsumexp(next_dist.logits, dim=1, keepdim=True)
            next_values = self.target_value.value(next_states)
        
        target_values = rewards + (1 - dones) * self.gamma * next_values
        values = self.network.value(states)
        value_loss = 0.5 * ((values - target_values) ** 2).mean()
        
        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()
        
        # Q-functions update
        with torch.no_grad():
            action_probs = self.network.actor(states)
            dist = Categorical(action_probs)
            action_log_probs = dist.logits - torch.logsumexp(dist.logits, dim=1, keepdim=True)
        
        q1_values = self.network.q1(states, actions.float())
        q2_values = self.network.q2(states, actions.float())
        
        target_q = rewards + (1 - dones) * self.gamma * target_values
        q1_loss = 0.5 * ((q1_values - target_q) ** 2).mean()
        q2_loss = 0.5 * ((q2_values - target_q) ** 2).mean()
        q_loss = q1_loss + q2_loss
        
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()
        
        # Actor update
        action_probs = self.network.actor(states)
        dist = Categorical(action_probs)
        sampled_actions = dist.sample()
        action_log_probs = dist.logits - torch.logsumexp(dist.logits, dim=1, keepdim=True)
        
        q_values = torch.min(
            self.network.q1(states, sampled_actions.unsqueeze(1).float()),
            self.network.q2(states, sampled_actions.unsqueeze(1).float())
        )
        
        actor_loss = (self.alpha * action_log_probs.squeeze() - q_values.squeeze()).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # Entropy coefficient update
        entropy = -(action_log_probs * action_probs).sum(dim=1).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
        
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        
        # Soft update target network
        for target_param, param in zip(self.target_value.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train(self, num_episodes=5000, max_steps=500, verbose=True):
        """Training loop"""
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
                
                if terminated:
                    success = (reward > 0)
                    break
            
            self.episode_rewards.append(episode_reward)
            self.episode_success.append(success)
            
            if verbose and (episode < 10 or episode % 100 == 0):
                success_rate = 100.0 * np.sum(self.episode_success) / (episode + 1)
                print(f"Episode {episode}/{num_episodes}: reward={episode_reward:.2f}, "
                      f"success_rate={success_rate:.1f}%")
        
        success_rate = 100.0 * np.sum(self.episode_success) / num_episodes
        print(f"\nTraining completed. Final success rate: {success_rate:.1f}%")
    
    def get_path(self, max_steps=1000):
        """Extract path using trained policy"""
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