# AgenticRoboArm: Deep RL for C-Space Navigation

A Python implementation of motion planning for a 2-DOF planar robotic arm in configuration space. This project serves as a comparative benchmark to evaluate the performance of four different Reinforcement Learning algorithms (Q-Learning, DQL, PPO, and SAC) in finding collision-free paths in robotics, particularly handling sparse reward environments.

## Table of Contents
- [Overview](#overview)
- [Theoretical Background](#theoretical-background)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Configuration](#configuration)
- [Common Issues](#common-issues)
- [License](#license)
- [Author](#author)

## Overview

This project implements robot motion planning for a two-link planar robotic arm operating in a workspace with obstacles. The robot learns to navigate from a start configuration to a goal configuration using a discrete action space. It compares traditional tabular reinforcement learning against modern deep reinforcement learning approaches.

### Key Features
- **2-DOF Planar Arm**: Forward kinematics for a two-link manipulator.
- **Configuration Space (C-Space)**: Discretized representation (e.g., 100×100 grid) of valid arm configurations.
- **Collision Detection**: Uses the `shapely` library for robust geometric collision checking.
- **Multi-Agent RL Benchmark**: Implements and compares Q-Learning, Deep Q-Network (DQL), Proximal Policy Optimization (PPO), and Soft Actor-Critic (SAC).
- **Auto-Topology Analysis**: Uses `scipy` to analyze C-Space connectivity, automatically finding valid, mutually reachable Start and Goal points within the largest free-space island.
- **Visualization**: Multiple visualization tools including workspace path plotting, C-space connectivity maps, and animated `.gif` trajectories.

## Theoretical Background

### Configuration Space
The **configuration space** (C-space) is a mathematical representation where each point corresponds to a unique configuration of the robot. For a 2-DOF planar arm:
- **Configuration**: (θ1, θ2) representing the angles of the two joints.
- **Free Space**: Configurations where the robot does not intersect any obstacles.
- **Obstacle Region**: Configurations where the robot collides with workspace obstacles.
Planning in C-space reduces the complex robot body to a single point navigating through a 2D grid.

### Reinforcement Learning Approaches
The environment features **sparse rewards** (a heavy penalty for collision, and a large +100 reward only upon reaching the exact goal). To solve this, the project explores:
1. **Q-Learning**: A tabular baseline.
2. **Deep Q-Learning (DQL)**: Uses a neural network to approximate Q-values, combined with a Replay Buffer.
3. **PPO (Proximal Policy Optimization)**: An on-policy actor-critic algorithm balancing sample complexity and stability.
4. **Discrete SAC (Soft Actor-Critic)**: Adapted from continuous domains, this off-policy algorithm uses an entropy-regularized framework to encourage exploration. The actor network outputs a categorical distribution (probabilities via Softmax), enabling the calculation of exact expected Q-values without the reparameterization trick.

## Project Structure

```
AgenticRoboArm/
├── src/                      # Core environment and robot logic
│   ├── arm.py                # 2-DOF planar arm kinematics
│   ├── cspace.py             # Configuration space generation
│   ├── obstacle.py           # Collision detection logic (Shapely)
│   ├── arm_env.py            # Gymnasium-compatible RL environment
│   └── visualize.py          # Plotting and GIF animation utilities
├── src/agents/               # PyTorch RL Agent implementations
│   ├── qlearning.py          # Tabular Q-Learning
│   ├── dql.py                # Deep Q-Network
│   ├── ppo.py                # Proximal Policy Optimization
│   └── sac.py                # Discrete Soft Actor-Critic
├── scripts/                  # Executable training scripts
│   ├── main_comparison.py    # Sequential multi-agent training & benchmark
│   └── main_single_agent.py  # Script for testing/tuning a single algorithm
├── results/                  # Auto-generated output directory
└── requirements.txt          # Python dependencies

```

## Installation

1. Clone the repository:

```bash
git clone [https://github.com/your-username/AgenticRoboArm.git](https://github.com/your-username/AgenticRoboArm.git)
cd AgenticRoboArm

```

2. Install the required dependencies:

```bash
pip install -r requirements.txt

```

*Main dependencies include `gymnasium`, `torch`, `numpy`, `matplotlib`, `shapely`, `scipy`, and `Pillow`.*

## Usage

To run the complete benchmark comparing all four agents:

```bash
python scripts/main_comparison.py

```

*Note: The script will automatically build the C-Space grid, find the largest connected component, select a reachable Start/Goal pair, and train the agents sequentially.*

To test and tune a specific agent individually:

```bash
python scripts/main_single_agent.py

```

## Results

After training, the `/results` directory will be populated with visualizations:

* **`cspace_connectivity.png`**: Highlights the free space islands and the chosen Start/Goal points.
* **`comparison_reward.png`**: A smoothed learning curve comparing the episodic rewards of all tested agents.
* **`anim_[AGENT].gif`**: Animated representations of the trained policies navigating the physical workspace.
* **`path_[AGENT].png`**: Static workspace overlays of the final trajectory.

## Configuration

To effectively handle the sparse reward structure of the 100×100 grid, the Deep RL agents (DQL, SAC) require specific hyperparameters:

* **Discount Factor (γ)**: High value (`0.99` or `0.995`) to prevent severe reward discounting over long paths (>60 steps).
* **Replay Buffer**: Large capacity (`memory_size=100000`) to prevent catastrophic forgetting once the sparse goal is finally found.
* **Max Steps per Episode**: Hard limit (e.g., `500` steps) to truncate episodes and prevent infinite loops during early, blind exploration.

## Common Issues

**Start/Goal in Obstacle**:

```
ValueError: Start state is in collision!

```

*Solution*: The `main_comparison.py` script uses `scipy.ndimage.label` to automatically prevent this. If setting coordinates manually in `main_single_agent.py`, ensure they fall within the free C-space.

**Agent Stuck in Infinite Loop / No Terminal Output**:
*Solution*: Ensure the training loop contains a `max_steps` counter (e.g., `for step in range(500):`) so the episode forcibly truncates if the agent neither collides nor finds the goal.

**Exploding Gradients / NaN Loss (Deep RL)**:

```
RuntimeError: found invalid values: tensor([[nan, nan, nan, nan]])

```

*Solution*: This occurs when network outputs approach zero, causing `log(0)` in entropy calculations. Ensure probabilities are clamped (e.g., `torch.clamp(probs, min=1e-8)`) and apply gradient clipping (`torch.nn.utils.clip_grad_norm_`).

**Understanding Periodic Boundary Effects**:

* Paths may appear discontinuous in C-space visualizations when crossing the 0 or 2π boundary.
* In the physical workspace, these paths represent continuous, valid robot motions (a full 360-degree rotation of a joint).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

eledaveri

## Acknowledgments

This project was developed as an educational implementation of path planning using deep reinforcement learning for robotic manipulators, extending foundational tabular approaches to modern continuous/discrete actor-critic frameworks.

