"""Neural network architectures for PPO agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from abc import ABC, abstractmethod

LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0


class BaseActorCritic(nn.Module, ABC):
    """Base class for Actor-Critic networks."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1)

    def get_value(self, state):
        """Get state value."""
        shared_features = self.shared_layers(state)
        return self.critic(shared_features)

    @abstractmethod
    def get_action_and_value(self, state, action=None):
        """Get action, log probability, entropy, and value."""
        pass

    @abstractmethod
    def evaluate(self, state, action):
        """Evaluate actions for given states."""
        pass


class DiscreteActorCritic(BaseActorCritic):
    """Actor-Critic network for discrete action spaces."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__(state_dim, action_dim, hidden_dim)
        
        # Actor head (policy network) - outputs logits for categorical distribution
        self.actor = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """Forward pass through both actor and critic."""
        shared_features = self.shared_layers(state)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_logits, state_value

    def get_action_and_value(self, state, action=None):
        """Get action, log probability, entropy, and value for discrete actions."""
        action_logits, value = self.forward(state)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, log_prob, entropy, value

    def evaluate(self, state, action):
        """Evaluate discrete actions for given states."""
        action_logits, value = self.forward(state)
        probs = Categorical(logits=action_logits)
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return log_prob, entropy, value


class ContinuousActorCritic(BaseActorCritic):
    """Actor-Critic network for continuous action spaces with tanh squashing."""

    def __init__(self, state_dim, action_dim, hidden_dim=64, action_low=-1.0, action_high=1.0):
        super().__init__(state_dim, action_dim, hidden_dim)
        
        # Store action bounds as scalars for now, convert to tensors in forward pass
        self.action_low_val = action_low
        self.action_high_val = action_high
        
        # Actor head - outputs mean for normal distribution
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # Learnable log standard deviation
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """Forward pass through both actor and critic."""
        shared_features = self.shared_layers(state)
        action_mean = self.actor_mean(shared_features)
        state_value = self.critic(shared_features)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return action_mean, state_value, log_std

    def get_action_and_value(self, state, action=None):
        """Get action, log probability, entropy, and value for continuous actions."""
        action_mean, value, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(action_mean, std)
        
        # Create action space tensors on the same device as the input
        device = action_mean.device
        action_low = torch.tensor(self.action_low_val, device=device, dtype=torch.float32)
        action_high = torch.tensor(self.action_high_val, device=device, dtype=torch.float32)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        
        if action is None:
            # Sample from normal distribution
            raw_action = dist.rsample()
        else:
            # Convert action back to raw space for log_prob calculation
            normalized_action = (action - action_bias) / action_scale
            raw_action = torch.atanh(torch.clamp(normalized_action, -0.95, 0.95))
        
        # Calculate log probability in raw space
        log_prob = dist.log_prob(raw_action)
        entropy = dist.entropy()
        
        # Apply tanh squashing and scale to action bounds
        tanh_raw = torch.tanh(raw_action)
        action = tanh_raw * action_scale + action_bias
        
        # Apply Jacobian correction for tanh transformation
        log_prob = log_prob - torch.log(action_scale * (1 - tanh_raw.pow(2)) + 1e-6)
        
        return action, log_prob, entropy, value

    def evaluate(self, state, action):
        """Evaluate continuous actions for given states."""
        action_mean, value, log_std = self.forward(state)
        std = torch.exp(log_std)
        dist = Normal(action_mean, std)
        
        # Create action space tensors on the same device as the input
        device = action.device
        action_low = torch.tensor(self.action_low_val, device=device, dtype=torch.float32)
        action_high = torch.tensor(self.action_high_val, device=device, dtype=torch.float32)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        
        # Convert action back to raw space
        normalized_action = (action - action_bias) / action_scale
        raw_action = torch.atanh(torch.clamp(normalized_action, -0.95, 0.95))
        
        # Calculate log probability and entropy in raw space
        log_prob = dist.log_prob(raw_action)
        entropy = dist.entropy()
        
        # Apply Jacobian correction for tanh transformation
        tanh_raw = torch.tanh(raw_action)
        log_prob = log_prob - torch.log(action_scale * (1 - tanh_raw.pow(2)) + 1e-6)
        
        return log_prob, entropy, value


# Legacy class for backward compatibility
class ActorCritic(DiscreteActorCritic):
    """Legacy ActorCritic class - defaults to discrete actions for backward compatibility."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64, continuous_space=None):
        if continuous_space is not None:
            # If continuous_space is provided, create ContinuousActorCritic instead
            raise ValueError("Use ContinuousActorCritic directly for continuous action spaces")
        super().__init__(state_dim, action_dim, hidden_dim)