import numpy as np

class PlanarArm2DOF:
    """Model of a planar 2-DOF robotic arm"""
    def __init__(self, link_lengths):
        self.l1, self.l2 = link_lengths


    def forward_kinematics(self, theta1, theta2):
        """Returns the (x,y) position of the end-effector"""
        x = self.l1 * np.cos(theta1) + self.l2 * np.cos(theta1 + theta2)
        y = self.l1 * np.sin(theta1) + self.l2 * np.sin(theta1 + theta2)
        return np.array([x, y])

    def get_segments(self, theta1, theta2):
        """Returns the line segments representing the arm's links"""
        p0 = np.array([0.0, 0.0])
        p1 = np.array([self.l1 * np.cos(theta1), self.l1 * np.sin(theta1)])
        p2 = self.forward_kinematics(theta1, theta2)
        return [(p0, p1), (p1, p2)]