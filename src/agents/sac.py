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
        
        # Critic 1: State -> Q-values per ogni azione
        self.q1 = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
        # Critic 2: State -> Q-values per ogni azione
        self.q2 = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )
        
    def forward(self, state):
        # Ritorna tutto per comodità, ma si possono chiamare singolarmente
        probs = self.actor(state)
        q1 = self.q1(state)
        q2 = self.q2(state)
        return probs, q1, q2

class SAC:
    """Discrete Soft Actor-Critic agent"""
    def __init__(self, env, actor_lr=0.001, q_lr=0.001, alpha_lr=0.001,
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
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, training=True):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.model.actor(state_t)
        
        if training:
            dist = Categorical(probs)
            action = dist.sample().item()
        else:
            action = torch.argmax(probs, dim=1).item()
        return action

    def replay(self):
        if len(self.memory) < self.batch_size: return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # --- Critic Update ---
        with torch.no_grad():
            # Calcolo V(s') usando la policy target e Q target
            next_probs = self.model.actor(next_states) # Usa actor corrente (o target, variante)
            # Evita log(0)
            z = (next_probs == 0.0).float() * 1e-8
            log_next_probs = torch.log(next_probs + z)
            
            next_q1 = self.target_model.q1(next_states)
            next_q2 = self.target_model.q2(next_states)
            min_next_q = torch.min(next_q1, next_q2)
            
            # V(s') = sum_a [ pi(a|s') * (Q(s',a) - alpha * log pi(a|s')) ]
            target_v = (next_probs * (min_next_q - self.alpha * log_next_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * target_v

        # Q correnti
        current_q1 = self.model.q1(states).gather(1, actions)
        current_q2 = self.model.q2(states).gather(1, actions)
        
        q_loss = 0.5 * nn.MSELoss()(current_q1, target_q) + 0.5 * nn.MSELoss()(current_q2, target_q)
        
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # --- Actor Update ---
        # Qui usiamo tutte le probabilità, non solo quelle campionate
        probs = self.model.actor(states)
        z = (probs == 0.0).float() * 1e-8
        log_probs = torch.log(probs + z)
        
        with torch.no_grad():
            q1 = self.model.q1(states)
            q2 = self.model.q2(states)
            min_q = torch.min(q1, q2)
        
        # Loss = sum_a [ pi(a|s) * (alpha * log pi(a|s) - Q(s,a)) ]
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # --- Alpha Update ---
        # Loss = -alpha * (entropy - target_entropy)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
        
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft Update Targets
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self, num_episodes=2000, verbose=True):
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = self.choose_action(state, training=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                
                self.remember(state, action, reward, next_state, terminated)
                self.replay()
                
                state = next_state
            
            self.episode_rewards.append(episode_reward)
            self.episode_success.append(terminated and episode_reward > 0)
            
            if verbose and (episode + 1) % 100 == 0:
                avg = np.mean(self.episode_rewards[-50:])
                print(f"Episode {episode+1}: SAC Reward {avg:.2f}")

    def get_path(self, max_steps=500):
        path = []
        state, _ = self.env.reset()
        path.append(tuple(state))
        for _ in range(max_steps):
            action = self.choose_action(state, training=False)
            state, _, term, trunc, _ = self.env.step(action)
            path.append(tuple(state))
            if term or trunc: break
        return path