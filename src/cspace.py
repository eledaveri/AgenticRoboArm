import numpy as np
from .arm import PlanarArm2DOF            # Import relativo
from .obstacle import check_collision     # Import relativo

class ConfigurationSpace:
    """C-space for a 2-DOF planar arm"""
    def __init__(self, arm, theta1_range, theta2_range, N1, N2, obstacles=[]):
        self.arm = arm
        self.theta1_vals = np.linspace(theta1_range[0], theta1_range[1], N1)
        self.theta2_vals = np.linspace(theta2_range[0], theta2_range[1], N2)
        self.grid = np.zeros((N1, N2), dtype=int)  # 0=free, 1=obstacle
        self.obstacles = obstacles
        self.N1, self.N2 = N1, N2

    def build(self):
        """ Builds the C-space grid """
        print("Building C-Space grid... (this might take a moment)")
        for i, th1 in enumerate(self.theta1_vals):
            for j, th2 in enumerate(self.theta2_vals):
                segments = self.arm.get_segments(th1, th2)

                if check_collision(segments, self.obstacles):
                    self.grid[i, j] = 1   # obstacle
                else:
                    self.grid[i, j] = 0   # free
        return self.grid