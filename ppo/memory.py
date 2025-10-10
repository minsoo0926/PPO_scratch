"""Memory buffer for storing experiences in PPO."""

import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Base class for PPO memory buffers."""

    def __init__(self, n_envs, buffer_size, state_dim, device='cpu'):
        """
        Initialize memory buffer.
        
        Args:
            buffer_size (int): Maximum size of the buffer
            state_dim (int): Dimension of state space
            device (str): Device to store tensors on
        """
        self.n_envs = n_envs
        self.buffer_size = buffer_size
        self.buffer_length = buffer_size // n_envs
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Initialize common buffers
        self.states = torch.zeros((n_envs, self.buffer_length, state_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((n_envs, self.buffer_length), dtype=torch.float32, device=device)
        self.values = torch.zeros((n_envs, self.buffer_length), dtype=torch.float32, device=device)
        self.dones = torch.zeros((n_envs, self.buffer_length), dtype=torch.bool, device=device)

    def is_full(self):
        """Check if buffer is full."""
        return self.ptr == self.buffer_length - 1
    
    def __len__(self):
        """Return current size of buffer."""
        return self.size

    def clear(self):
        """Clear the memory buffer."""
        self.ptr = 0
        self.size = 0

    @abstractmethod
    def store(self, idx_env, state, action, reward, value, log_prob, done):
        """Store a single experience."""
        pass

    @abstractmethod
    def get(self) -> tuple:
        """Get all stored experiences."""
        pass


class DiscreteMemory(BaseMemory):
    """Memory buffer for discrete action spaces."""

    def __init__(self, n_envs, buffer_size, state_dim, device='cpu'):
        super().__init__(n_envs, buffer_size, state_dim, device)

        # Discrete action specific buffers
        self.actions = torch.zeros((self.n_envs, self.buffer_length), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((self.n_envs, self.buffer_length), dtype=torch.float32, device=device)

    def store(self, idx_env, state, action, reward, value, log_prob, done):
        """Store a discrete action experience."""
        self.states[idx_env, self.ptr] = state
        self.actions[idx_env, self.ptr] = int(action)
        self.rewards[idx_env, self.ptr] = float(reward)
        self.values[idx_env, self.ptr] = float(value)
        self.log_probs[idx_env, self.ptr] = float(log_prob)
        self.dones[idx_env, self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.buffer_length if idx_env == self.n_envs - 1 else self.ptr
        self.size = min(self.size + self.n_envs, self.buffer_length)

    def get(self):
        """Get all stored discrete experiences."""
        assert self.size > 0, "Memory buffer is empty"

        if self.size < self.buffer_length:
            return (
                self.states[:, :self.size],
                self.actions[:, :self.size],
                self.rewards[:, :self.size],
                self.values[:, :self.size],
                self.log_probs[:, :self.size],
                self.dones[:, :self.size]
            )
        else:
            indices = torch.arange(self.ptr, self.ptr + self.buffer_length, device=self.device) % self.buffer_length
            return (
                self.states[:, indices],
                self.actions[:, indices],
                self.rewards[:, indices],
                self.values[:, indices],
                self.log_probs[:, indices],
                self.dones[:, indices]
            )


class ContinuousMemory(BaseMemory):
    """Memory buffer for continuous action spaces."""
    
    def __init__(self, n_envs, buffer_size, state_dim, action_dim, device='cpu'):
        super().__init__(n_envs, buffer_size, state_dim, device)

        # Continuous action specific buffers
        self.action_dim = action_dim
        self.actions = torch.zeros((n_envs, self.buffer_size, action_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((n_envs, self.buffer_size, 1), dtype=torch.float32, device=device)

    def store(self, idx_env, state, action, reward, value, log_prob, done):
        """Store a continuous action experience."""
        self.states[idx_env, self.ptr] = state
        self.actions[idx_env, self.ptr] = action
        self.rewards[idx_env, self.ptr] = float(reward)
        self.values[idx_env, self.ptr] = float(value)
        self.log_probs[idx_env, self.ptr] = log_prob
        self.dones[idx_env, self.ptr] = bool(done)

        self.ptr = (self.ptr + 1) % self.buffer_length if idx_env == self.n_envs - 1 else self.ptr
        self.size = min(self.size + self.n_envs, self.buffer_length)

    def get(self):
        """Get all stored continuous experiences."""
        assert self.size > 0, "Memory buffer is empty"

        if self.size < self.buffer_length:
            return (
                self.states[:, :self.size],
                self.actions[:, :self.size],
                self.rewards[:, :self.size],
                self.values[:, :self.size],
                self.log_probs[:, :self.size],
                self.dones[:, :self.size]
            )
        else:
            indices = torch.arange(self.ptr, self.ptr + self.buffer_length, device=self.device) % self.buffer_length
            return (
                self.states[:, indices],
                self.actions[:, indices],
                self.rewards[:, indices],
                self.values[:, indices],
                self.log_probs[:, indices],
                self.dones[:, indices]
            )


# Legacy class for backward compatibility
class PPOMemory(DiscreteMemory):
    """Legacy PPOMemory class - defaults to discrete actions for backward compatibility."""
    
    def __init__(self, buffer_size, state_dim, device='cpu', continuous_dim=None):
        if continuous_dim is not None:
            raise ValueError("Use ContinuousMemory directly for continuous action spaces")
        super().__init__(buffer_size, state_dim, device)