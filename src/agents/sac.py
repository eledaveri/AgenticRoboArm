import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
from collections import deque

class SACNetwork(nn.Module):
    def __init__(self, state_size=2, action_size=4, hidden_size=64):
        super(SACNetwork, self).__init__()
        
        # Actor: State -> Action Probabilities
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Softmax(dim=-1)
        )
        
        # Critic 1: State -> Q-values for each action
        self.q1 = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Critic 2: State -> Q-values for each action
        self.q2 = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, state):
        """Given a state, return action probabilities and Q-values from both critics"""
        probs = self.actor(state)
        q1 = self.q1(state)
        q2 = self.q2(state)
        return probs, q1, q2

class SAC:
    """Discrete Soft Actor-Critic agent"""
    def __init__(self, env, actor_lr=0.0003, q_lr=0.0003, alpha_lr=0.001,
                 gamma=0.95, tau=0.005, alpha=0.2, memory_size=10000,
                 batch_size=64, device=None):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # Networks
        self.model = SACNetwork(state_size=2, action_size=env.action_space.n).to(self.device)
        self.target_model = SACNetwork(state_size=2, action_size=env.action_space.n).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # Optimizers
        self.actor_opt = optim.Adam(self.model.actor.parameters(), lr=actor_lr)
        self.q_opt = optim.Adam(list(self.model.q1.parameters()) + list(self.model.q2.parameters()), lr=q_lr)
        
        # Entropy automatica
        self.target_entropy = -np.log(1.0 / env.action_space.n) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        
        self.memory = deque(maxlen=memory_size)
        self.episode_rewards = []
        self.episode_success = []

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def remember(self, state, action, reward, next_state, done):
        scaled_reward = reward / 20.0
        self.memory.append((state, action, scaled_reward, next_state, done))

    def choose_action(self, state, training=True):
        """Sample action from policy
        Args:
            state: current state
            training: if True, sample stochastically; if False, choose action with highest probability
        Returns: action: chosen action
        """
        # Input normalization
        state_norm = np.array(state, dtype=np.float32) / self.env.n_discretization
        state_t = torch.FloatTensor(state_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = self.model.actor(state_t)
        
        if training:
            dist = Categorical(probs)
            action = dist.sample().item()
        else:
            action = torch.argmax(probs, dim=1).item()
        return action

    def replay(self):
        """Update networks using a batch of experiences from memory"""
        if len(self.memory) < self.batch_size: return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Batch normalization
        states = torch.FloatTensor(np.array(states, dtype=np.float32) / self.env.n_discretization).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # Next states normalization
        next_states = torch.FloatTensor(np.array(next_states, dtype=np.float32) / self.env.n_discretization).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Critic Update ---
        with torch.no_grad():
            # Compute target Q-values
            next_probs = self.model.actor(next_states) 
            # Avoid log(0) by adding a small constant
            z = (next_probs == 0.0).float() * 1e-8
            log_next_probs = torch.log(next_probs + z)
            
            next_q1 = self.target_model.q1(next_states)
            next_q2 = self.target_model.q2(next_states)
            min_next_q = torch.min(next_q1, next_q2)

            target_v = (next_probs * (min_next_q - self.alpha * log_next_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * target_v

        # Current Q-values for the actions taken
        current_q1 = self.model.q1(states).gather(1, actions)
        current_q2 = self.model.q2(states).gather(1, actions)
        
        q_loss = 0.5 * nn.MSELoss()(current_q1, target_q) + 0.5 * nn.MSELoss()(current_q2, target_q)
        
        self.q_opt.zero_grad()
        q_loss.backward()

        torch.nn.utils.clip_grad_norm_(list(self.model.q1.parameters()) + list(self.model.q2.parameters()), 1.0)
        
        self.q_opt.step()

        # --- Actor Update ---
        probs = self.model.actor(states)
        z = (probs == 0.0).float() * 1e-8
        log_probs = torch.log(probs + z)
        
        with torch.no_grad():
            q1 = self.model.q1(states)
            q2 = self.model.q2(states)
            min_q = torch.min(q1, q2)
        
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), 1.0)
        
        self.actor_opt.step()

        entropy = -(probs * log_probs).sum(dim=1).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
        
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft Update Targets
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, num_episodes=2000, verbose=True, max_steps=500):
        """Training loop
        Args:
            num_episodes: total number of episodes to train
            verbose: if True, print progress every 100 episodes
            max_steps: maximum steps per episode to prevent infinite loops
        Returns: training results dictionary and the last trajectory
        """
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            steps = 0  # Steo counter to prevent infinite loops
            
            # Stop training if it takes too long (e.g., due to loops)
            while not (terminated or truncated) and steps < max_steps:
                action = self.choose_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                self.remember(state, action, reward, next_state, terminated)
                self.replay()
                
                state = next_state
                steps += 1  
            
            self.episode_rewards.append(episode_reward)
            # Consider success only if terminated with a positive reward (reaching the goal) 
            self.episode_success.append(terminated and episode_reward > 0)
            
            if verbose and (episode + 1) % 100 == 0:
                avg = np.mean(self.episode_rewards[-50:])
                success_rate = 100.0 * np.sum(self.episode_success[-100:]) / min(len(self.episode_success), 100)
                print(f"Episode {episode+1}: SAC Reward {avg:.2f}, Success {success_rate:.1f}%")

    def get_path(self, max_steps=500):
        """Extract path using learned policy
        Args:
            max_steps: maximum steps to extract path
        Returns: list of states in the path """
        path = []
        state, _ = self.env.reset()
        path.append(tuple(state))
        
        for _ in range(max_steps):
            action = self.choose_action(state, training=False)
            state, reward, term, trunc, _ = self.env.step(action) 
            path.append(tuple(state))
            
            if term or trunc:
                if reward > 0:
                    print(f"Goal reached in {len(path)} steps!")
                else:
                    print(f"Failed (Collision) in {len(path)} steps.")
                break
        return path