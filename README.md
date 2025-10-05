# PPO from Scratch

A complete implementation of the Proximal Policy Optimization (PPO) algorithm from scratch, compatible with OpenAI Gym/Gymnasium API.

## Features

- **Complete PPO Implementation**: Includes actor-critic networks, experience buffer, and the full PPO algorithm
- **Gymnasium Compatible**: Compatible with any Gymnasium environment (discrete or continuous)
- **GAE (Generalized Advantage Estimation)**: Implements GAE for better advantage estimation
- **Numerical Stability**: Advanced NaN/infinity detection and prevention mechanisms
- **Flexible Architecture**: Configurable network architecture and hyperparameters
- **Training Utilities**: Includes training scripts with progress tracking and visualization
- **Model Persistence**: Save and load trained models with automatic compatibility checking

## Installation

1. Clone the repository:
```bash
git clone https://github.com/minsoo0926/PPO_scratch.git
cd PPO_scratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

#### Automatic Agent Selection (Recommended)

```python
import gymnasium as gym
from ppo import create_ppo_agent

# Create environment (works with both discrete and continuous action spaces)
env = gym.make('CartPole-v1')  # or 'Pendulum-v1', 'Ant-v5', etc.

# Automatically creates the appropriate agent based on action space
agent = create_ppo_agent(
    env,
    lr=3e-4,
    gamma=0.99,
    hidden_dim=64,
    buffer_size=2048
)

# Training loop (same for both discrete and continuous)
for episode in range(100):
    state, _ = env.reset()
    episode_reward = 0
    
    while True:
        action, log_prob, value = agent.get_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.store_experience(state, action, reward, value, log_prob, done)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    # Update agent when buffer is ready
    if len(agent.memory) >= agent.batch_size:
        agent.update()
```

### Using the Training Script
- Modify config.py as your training environment
```bash
make train
make test
```

## Key Features

### PPO Algorithm
- **Actor-Critic Architecture** with clipped surrogate objective
- **GAE (Generalized Advantage Estimation)** for variance reduction
- **Multiple training epochs** per batch for sample efficiency

### Numerical Stability
- **NaN/Infinity Detection** with automatic batch skipping
- **Log Probability Clipping** (±20 range) to prevent overflow
- **Gradient Clipping** and detailed debugging output

### Hyperparameters
- `lr=3e-4`, `gamma=0.99`, `lam=0.95`, `clip_ratio=0.2`
- `value_coef=0.5`, `entropy_coef=0.01`, `hidden_dim=64`

## API Reference

### create_ppo_agent (Recommended)

```python
create_ppo_agent(
    env,                 # Gymnasium environment
    lr=3e-4,            # Learning rate
    gamma=0.99,         # Discount factor
    lam=0.95,           # GAE lambda
    clip_ratio=0.2,     # PPO clip ratio
    value_coef=0.5,     # Value loss coefficient
    entropy_coef=0.01,  # Entropy coefficient
    max_grad_norm=0.5,  # Gradient clipping
    hidden_dim=64,      # Hidden layer size
    buffer_size=2048,   # Experience buffer size
    batch_size=64,      # Mini-batch size
    epochs=10,          # Training epochs per update
    device='cpu'        # Device to run on
)
```

Automatically creates `DiscretePPOAgent` for discrete action spaces or `ContinuousPPOAgent` for continuous action spaces based on the environment's action space type.

### Manual Agent Classes

For advanced users who need direct control:

**DiscretePPOAgent**: For `gym.spaces.Discrete` action spaces
**ContinuousPPOAgent**: For `gym.spaces.Box` action spaces (requires `action_low` and `action_high` parameters)

#### Methods

- `get_action(state, deterministic=False)`: Get action for given state
- `store_experience(state, action, reward, value, log_prob, done)`: Store experience in buffer
- `update(next_state=None)`: Update policy using PPO algorithm
- `save(filepath)`: Save model to file
- `load(filepath)`: Load model from file

## File Structure

```
PPO_scratch/
├── ppo/                     # Main PPO implementation
├── models/                  # Saved models
├── train.py                 # Training script  
├── example.py               # Usage example
└── config.py                # Configuration
```

## Supported Environments

**Automatic compatibility** with Gymnasium environments:
- **Discrete**: `gym.spaces.Discrete` → `DiscretePPOAgent`  
- **Continuous**: `gym.spaces.Box` → `ContinuousPPOAgent`

**Tested**: CartPole-v1, LunarLander-v2, Pendulum-v1, BipedalWalker-v3, Ant-v5, Humanoid-v5
