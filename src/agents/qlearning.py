import numpy as np
from src.arm import PlanarArm2DOF
from src.obstacle import check_collision

import numpy as np
import random

class QLearning2DOF:
    """Tabular Q-learning for 2DOF planar arm in C-space"""

    def __init__(self, cspace, start=(0, 0), goal=None, num_actions=4,
                 alpha=0.1, gamma=0.95, epsilon=0.9):
        self.cspace = cspace
        self.N1 = cspace.N1
        self.N2 = cspace.N2
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_initial = epsilon
        self.epsilon = epsilon
        self.epsilon_min = 0.01  

        self.start = start
        self.goal = goal if goal else (self.N1-1, self.N2-1)

        # Check whether start and goal are in free space
        if self.cspace.grid[self.start[0], self.start[1]] == 1:
            raise ValueError("Start position is inside an obstacle!")
        
        if self.cspace.grid[self.goal[0], self.goal[1]] == 1:
            raise ValueError("Goal position is inside an obstacle!")

        # Initialize Q-table
        self.Q = np.random.randn(self.N1, self.N2, num_actions) * 0.01
        
        # Counters for analysis
        self.visit_count = np.zeros((self.N1, self.N2))
        self.collision_count = 0
        self.episode_count = 0

    def step(self, state, action):
        """Executes action in the environment, returns next_state, reward, done"""
        i, j = state
        
        # Compute new state based on action (with periodic boundary conditions)
        if action == 0:    # theta1+
            i_new = (i + 1) % self.N1
            j_new = j
        elif action == 1:  # theta1-
            i_new = (i - 1) % self.N1
            j_new = j
        elif action == 2:  # theta2+
            i_new = i
            j_new = (j + 1) % self.N2
        elif action == 3:  # theta2-
            i_new = i
            j_new = (j - 1) % self.N2
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check collision
        if self.cspace.grid[i_new, j_new] == 1:
            return state, -100.0, False
        
        # Check if goal reached
        if (i_new, j_new) == self.goal:
            return (i_new, j_new), 100.0, True
        
        dist_to_goal = abs(i_new - self.goal[0]) + abs(j_new - self.goal[1])
        reward = -1.0 - 0.01 * dist_to_goal
        
        return (i_new, j_new), reward, False

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions-1)
        else:
            i, j = state
            q_values = self.Q[i, j, :]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def train(self, num_episodes=5000, max_steps=500, verbose=True):
        """Training loop con epsilon decay migliorato"""
        success_count = 0
        collision_count = 0
        
        # Exponential decay factor for epsilon 
        epsilon_decay = (self.epsilon_min / self.epsilon_initial) ** (1.0 / num_episodes)
        
        for episode in range(num_episodes):
            state = self.start
            total_reward = 0
            had_collision = False
            
            for step_count in range(max_steps):
                self.visit_count[state[0], state[1]] += 1
                
                action = self.choose_action(state)
                next_state, reward, done = self.step(state, action)
                total_reward += reward
                
                # Check for collision
                if reward == -100.0:
                    had_collision = True
                
                # Q-learning update
                i, j = state
                i_next, j_next = next_state
                
                best_next_q = np.max(self.Q[i_next, j_next, :])
                td_target = reward + self.gamma * best_next_q
                td_error = td_target - self.Q[i, j, action]
                self.Q[i, j, action] += self.alpha * td_error
                
                state = next_state
                
                if done:
                    if reward > 0:
                        success_count += 1
                    break
            
            if had_collision:
                collision_count += 1
            
            # Verbose output 
            if verbose and (episode < 10 or episode % 100 == 0):
                success_rate = 100.0 * success_count / (episode + 1)
                collision_rate = 100.0 * collision_count / (episode + 1)
                print(f"Episode {episode}: steps={step_count+1}, reward={total_reward:.2f}, "
                      f"epsilon={self.epsilon:.3f}, success={success_rate:.1f}%, collisions={collision_rate:.1f}%")
            
            # Epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * epsilon_decay)
        
        print(f"\nTraining completed. Final success rate: {100.0*success_count/num_episodes:.1f}%")
        print(f"Collision rate: {100.0*collision_count/num_episodes:.1f}%")
        print(f"Visited states: {np.sum(self.visit_count > 0)} / {self.N1 * self.N2}")

    def get_path(self, start=None, max_steps=1000):
        """Extract the greedy path from the Q-table"""
        state = start if start else self.start
        path = [state]
        visited = {state}
        
        for step in range(max_steps):
            i, j = state
            
            # Choose best action
            action = np.argmax(self.Q[i, j, :])
            next_state, reward, done = self.step(state, action)
            
            # Check loop
            if next_state in visited:
                print(f"WARNING: Loop detected at step {step}, state {next_state}")
                print(f"State {state} Q-values: {self.Q[i, j, :]}")
                break
            
            path.append(next_state)
            visited.add(next_state)
            state = next_state
            
            if done:
                if reward > 0:
                    print(f"Goal reached in {len(path)} steps!")
                break
        
        if state != self.goal:
            print(f"WARNING: Goal NOT reached. Last state: {state}, Goal: {self.goal}")
        
        return path

