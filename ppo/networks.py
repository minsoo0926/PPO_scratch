"""Neural network architectures for PPO agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from abc import ABC, abstractmethod
from ppo.normalizer import RunningMeanStd, RunningStd

LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0


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
        self.obs_rms = RunningMeanStd(shape=(state_dim,))
        self.rew_rs = RunningStd(shape=(1,))
        self.obs_norm = True

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1)

    def get_value(self, state):
        """Get state value."""
        shared_features = self.shared_layers(state)
        return self.critic(shared_features)
    
    def update_obs_rms(self, state):
        """Update observation running mean and std."""
        self.obs_rms.update(state)

    def forward_rew_rs(self, reward):
        """Update reward running mean and std."""
        self.rew_rs.update(reward)
        return self.rew_rs.normalize(reward)

    @abstractmethod
    def get_action_and_value(self, state, action=None, deterministic=False):
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
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        """Forward pass through both actor and critic."""
        if self.obs_norm:
            state = self.obs_rms.normalize(state)
        shared_features = self.shared_layers(state)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        return action_logits, state_value

    def get_action_and_value(self, state, action=None, deterministic=False):
        """Get action, log probability, entropy, and value for discrete actions."""
        action_logits, value = self.forward(state)
        probs = Categorical(logits=action_logits)

        if deterministic:
            action = torch.argmax(probs.probs, dim=-1)
            batch_size = state.shape[0] if state.dim() > 1 else 1
            zero = torch.zeros((batch_size,), device=state.device)
            return action, zero, zero, value
        
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

    def __init__(self, state_dim, action_dim, hidden_dim=64, action_low=np.array([-1.0]), action_high=np.array([1.0])):
        super().__init__(state_dim, action_dim, hidden_dim)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if np.isscalar(action_low) or (isinstance(action_low, np.ndarray) and action_low.size == 1):
            action_low = np.full(action_dim, float(np.array(action_low).item()), dtype=np.float32)
        if np.isscalar(action_high) or (isinstance(action_high, np.ndarray) and action_high.size == 1):
            action_high = np.full(action_dim, float(np.array(action_high).item()), dtype=np.float32)

        self.action_low_val = np.asarray(action_low, dtype=np.float32)
        self.action_high_val = np.asarray(action_high, dtype=np.float32)

        self.action_scale = torch.tensor(
            (self.action_high_val - self.action_low_val) / 2.0,
            device=self.device,
            dtype=torch.float32,
        )
        self.action_bias = torch.tensor(
            (self.action_high_val + self.action_low_val) / 2.0,
            device=self.device,
            dtype=torch.float32,
        )
        
        # Actor head - outputs raw mean (pre-tanh)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Learnable log standard deviation parameter
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """Forward pass through both actor and critic."""
        state = state.to(torch.float32)
        if self.obs_norm:
            state = self.obs_rms.normalize(state)
        shared_features = self.shared_layers(state)
        raw_action_mean = self.actor_mean(shared_features)
        state_value = self.critic(shared_features)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return raw_action_mean, state_value, log_std

    def get_action_and_value(self, state, action=None, deterministic=False):
        """Get action, log probability, entropy, and value for continuous actions."""
        raw_action_mean, value, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(raw_action_mean) * self.action_scale + self.action_bias
            batch_size = state.shape[0] if state.dim() > 1 else 1
            zero = torch.zeros((batch_size, 1), device=self.device)
            return action, zero, zero, value

        # Create normal distribution in raw space
        dist = Normal(raw_action_mean, std)
        
        if action is None:
            # Sample from normal distribution in raw space
            raw_action = dist.rsample()
        else:
            # Convert provided action back to raw space for log_prob calculation
            # First normalize to [-1, 1]
            normalized_action = (action - self.action_bias) / self.action_scale
            # Then apply inverse tanh (clamp to avoid numerical issues)
            raw_action = torch.atanh(torch.clamp(normalized_action, -0.999, 0.999))
        
        # Calculate log probability and entropy from raw space
        # log(1 - tanh(raw_action)^2)
        squash_correction = (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1, keepdim=True)
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True) - squash_correction
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        # Apply tanh to constrain to [-1, 1] then scale to actual action bounds
        assert self.action_scale is not None and self.action_bias is not None
        squashed_action = torch.tanh(raw_action)
        final_action = squashed_action * self.action_scale + self.action_bias
        
        return final_action, log_prob, entropy, value

    def evaluate(self, state, action):
        """Evaluate continuous actions for given states."""
        raw_action_mean, value, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        # Create normal distribution in raw (pre-tanh) space
        dist = Normal(raw_action_mean, std)
        
        # Convert action back to raw space
        # First normalize to [-1, 1]
        assert self.action_scale is not None and self.action_bias is not None
        normalized_action = (action - self.action_bias) / self.action_scale
        # Then apply inverse tanh (clamp to avoid numerical issues)
        raw_action = torch.atanh(torch.clamp(normalized_action, -0.999, 0.999))
        
        # Calculate log probability and entropy in raw space with tanh correction
        squash_correction = (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(dim=-1, keepdim=True)
        log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True) - squash_correction
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, value