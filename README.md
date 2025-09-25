# PPO from Scratch

A complete implementation of the Proximal Policy Optimization (PPO) algorithm from scratch, compatible with OpenAI Gym/Gymnasium API.

## Features

- **Complete PPO Implementation**: Includes actor-critic networks, experience buffer, and the full PPO algorithm
- **Gymnasium Compatible**: Works with any Gymnasium environment with discrete action spaces
- **GAE (Generalized Advantage Estimation)**: Implements GAE for better advantage estimation
- **Flexible Architecture**: Configurable network architecture and hyperparameters
- **Training Utilities**: Includes training scripts with progress tracking and visualization
- **Model Persistence**: Save and load trained models

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

```python
import gymnasium as gym
from ppo import PPOAgent

# Create environment
env = gym.make('CartPole-v1')

# Initialize PPO agent
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr=3e-4,
    gamma=0.99
)

# Training loop
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
    if agent.memory.is_full():
        agent.update()
```

### Training with the Provided Script

```bash
python train.py
```

This will train a PPO agent on CartPole-v1 and save the model periodically.

### Running the Simple Example

```bash
python example.py
```

## Algorithm Details

### PPO (Proximal Policy Optimization)

PPO is a policy gradient method that aims to improve training stability by constraining policy updates. Key components include:

1. **Actor-Critic Architecture**: Shared network with separate heads for policy (actor) and value function (critic)
2. **Clipped Surrogate Objective**: Prevents large policy updates by clipping the probability ratio
3. **Generalized Advantage Estimation (GAE)**: Reduces variance in advantage estimates
4. **Multiple Epochs**: Updates the policy multiple times on each batch of data

### Key Hyperparameters

- `lr`: Learning rate (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- `lam`: GAE lambda parameter (default: 0.95)
- `clip_ratio`: PPO clipping parameter (default: 0.2)
- `value_coef`: Value function loss coefficient (default: 0.5)
- `entropy_coef`: Entropy regularization coefficient (default: 0.01)

## API Reference

### PPOAgent

```python
PPOAgent(
    state_dim,           # Dimension of state space
    action_dim,          # Dimension of action space (discrete)
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

#### Methods

- `get_action(state, deterministic=False)`: Get action for given state
- `store_experience(state, action, reward, value, log_prob, done)`: Store experience in buffer
- `update(next_state=None)`: Update policy using PPO algorithm
- `save(filepath)`: Save model to file
- `load(filepath)`: Load model from file

## File Structure

```
PPO_scratch/
├── ppo/
│   ├── __init__.py          # Package initialization
│   ├── ppo_agent.py         # Main PPO agent implementation
│   ├── networks.py          # Neural network architectures
│   └── memory.py            # Experience buffer
├── train.py                 # Training script with visualization
├── example.py               # Simple usage example
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Supported Environments

This implementation works with any Gymnasium environment that has:
- Discrete action space (`gym.spaces.Discrete`)
- Box observation space (`gym.spaces.Box`)

Tested environments:
- CartPole-v1
- LunarLander-v2
- Acrobot-v1

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.
