import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .arm import PlanarArm2DOF
from .obstacle import check_collision

class ArmNavigationEnv(gym.Env):
    """
    Gymnasium environment for 2-DOF planar arm navigation in C-space.
    
    The agent navigates in configuration space (theta1, theta2) to reach a goal
    while avoiding obstacles.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, arm, theta1_range, theta2_range, n_discretization, 
                 obstacles=None, start=None, goal=None, continuous=False):
        """
        Args:
            arm: PlanarArm2DOF object
            theta1_range: (min, max) for theta1
            theta2_range: (min, max) for theta2
            n_discretization: number of discrete states per dimension
            obstacles: list of obstacle objects
            start: (i, j) starting state in discretized space
            goal: (i, j) goal state in discretized space
            continuous: if True, use continuous action space; else discrete
        """
        super().__init__()
        
        self.arm = arm
        self.theta1_range = theta1_range
        self.theta2_range = theta2_range
        self.n_discretization = n_discretization
        self.obstacles = obstacles if obstacles else []
        
        # Create discretization
        self.theta1_vals = np.linspace(theta1_range[0], theta1_range[1], n_discretization)
        self.theta2_vals = np.linspace(theta2_range[0], theta2_range[1], n_discretization)
        
        # Set start and goal
        self.start = start if start else (0, 0)
        self.goal = goal if goal else (n_discretization - 1, n_discretization - 1)
        self.current_state = self.start
        
        # Check validity
        if not self._is_free_state(self.start):
            raise ValueError("Start state is in collision!")
        if not self._is_free_state(self.goal):
            raise ValueError("Goal state is in collision!")
        
        self.continuous = continuous
        
        if continuous:
            # Continuous action space: [delta_theta1, delta_theta2]
            self.action_space = spaces.Box(
                low=-0.2, high=0.2, shape=(2,), dtype=np.float32
            )
        else:
            # Discrete action space: 4 directions
            self.action_space = spaces.Discrete(4)
        
        # Observation space: (i, j) in discretized space
        self.observation_space = spaces.Box(
            low=0, high=n_discretization - 1, shape=(2,), dtype=np.int32
        )
    
    def _is_free_state(self, state):
        """Check if a state (in discretized space) is collision-free"""
        i, j = state
        if not (0 <= i < self.n_discretization and 0 <= j < self.n_discretization):
            return False
        
        theta1 = self.theta1_vals[i]
        theta2 = self.theta2_vals[j]
        segments = self.arm.get_segments(theta1, theta2)
        
        return not check_collision(segments, self.obstacles)
    
    def _discretize_continuous_action(self, action):
        """Convert continuous action to discrete movement"""
        delta_theta1, delta_theta2 = action
        
        i, j = self.current_state
        
        # Determine which dimension and direction changed most
        if abs(delta_theta1) > abs(delta_theta2):
            if delta_theta1 > 0:
                return 0  # theta1+
            else:
                return 1  # theta1-
        else:
            if delta_theta2 > 0:
                return 2  # theta2+
            else:
                return 3  # theta2-
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: either discrete (0-3) or continuous (2D array)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Handle continuous actions
        if self.continuous:
            action = self._discretize_continuous_action(action)
        
        i, j = self.current_state
        
        # Apply action with periodic boundary conditions
        if action == 0:  # theta1+
            i_new = (i + 1) % self.n_discretization
            j_new = j
        elif action == 1:  # theta1-
            i_new = (i - 1) % self.n_discretization
            j_new = j
        elif action == 2:  # theta2+
            i_new = i
            j_new = (j + 1) % self.n_discretization
        elif action == 3:  # theta2-
            i_new = i
            j_new = (j - 1) % self.n_discretization
        else:
            raise ValueError(f"Invalid action: {action}")
        
        new_state = (i_new, j_new)
        
        # Check collision
        if not self._is_free_state(new_state):
            reward = -100.0
            terminated = True
        else:
            # Check if goal reached
            if new_state == self.goal:
                reward = 100.0
                terminated = True
            else:
                # Distance-based reward (Manhattan distance in discrete space)
                dist_to_goal = abs(i_new - self.goal[0]) + abs(j_new - self.goal[1])
                reward = -0.1 - 0.01 * dist_to_goal     #-1.0 - 0.01 * dist_to_goal
                terminated = False
        
        self.current_state = new_state
        
        observation = np.array(self.current_state, dtype=np.int32)
        truncated = False
        info = {"state": self.current_state}
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        self.current_state = self.start
        observation = np.array(self.current_state, dtype=np.int32)
        info = {}
        return observation, info
    
    def render(self):
        """Render the environment (placeholder for compatibility)"""
        pass
    
    def get_theta_from_state(self, state):
        """Convert discrete state to actual angles"""
        i, j = state
        return self.theta1_vals[i], self.theta2_vals[j]
    
    def get_state_from_theta(self, theta1, theta2):
        """Convert angles to nearest discrete state"""
        i = np.argmin(np.abs(self.theta1_vals - theta1))
        j = np.argmin(np.abs(self.theta2_vals - theta2))
        return (i, j)
