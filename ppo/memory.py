"""Memory buffer for storing experiences in PPO."""

import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseMemory(ABC):
    """Base class for PPO memory buffers."""
    
    def __init__(self, buffer_size, state_dim, device='cpu'):
        """
        Initialize memory buffer.
        
        Args:
            buffer_size (int): Maximum size of the buffer
            state_dim (int): Dimension of state space
            device (str): Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Initialize common buffers
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)

    def is_full(self):
        """Check if buffer is full."""
        return self.size == self.buffer_size
    
    def __len__(self):
        """Return current size of buffer."""
        return self.size

    def clear(self):
        """Clear the memory buffer."""
        self.ptr = 0
        self.size = 0

    @abstractmethod
    def store(self, state, action, reward, value, log_prob, done):
        """Store a single experience."""
        pass

    @abstractmethod
    def get(self):
        """Get all stored experiences."""
        pass


class DiscreteMemory(BaseMemory):
    """Memory buffer for discrete action spaces."""
    
    def __init__(self, buffer_size, state_dim, device='cpu'):
        super().__init__(buffer_size, state_dim, device)
        
        # Discrete action specific buffers
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
    
    def store(self, state, action, reward, value, log_prob, done):
        """Store a discrete action experience."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        
        self.states[self.ptr] = state.to(self.device)
        self.actions[self.ptr] = int(action)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.dones[self.ptr] = bool(done)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get(self):
        """Get all stored discrete experiences."""
        assert self.size > 0, "Memory buffer is empty"
        
        if self.size < self.buffer_size:
            return (
                self.states[:self.size],
                self.actions[:self.size],
                self.rewards[:self.size],
                self.values[:self.size],
                self.log_probs[:self.size],
                self.dones[:self.size]
            )
        else:
            indices = torch.arange(self.ptr, self.ptr + self.buffer_size, device=self.device) % self.buffer_size
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.values[indices],
                self.log_probs[indices],
                self.dones[indices]
            )


class ContinuousMemory(BaseMemory):
    """Memory buffer for continuous action spaces."""
    
    def __init__(self, buffer_size, state_dim, action_dim, device='cpu'):
        super().__init__(buffer_size, state_dim, device)
        
        # Continuous action specific buffers
        self.action_dim = action_dim
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
    
    def store(self, state, action, reward, value, log_prob, done):
        """Store a continuous action experience."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float()
        if isinstance(log_prob, np.ndarray):
            log_prob = torch.from_numpy(log_prob).float()
        
        self.states[self.ptr] = state.to(self.device)
        self.actions[self.ptr] = action.to(self.device)
        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = log_prob.to(self.device)
        self.dones[self.ptr] = bool(done)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get(self):
        """Get all stored continuous experiences."""
        assert self.size > 0, "Memory buffer is empty"
        
        if self.size < self.buffer_size:
            return (
                self.states[:self.size],
                self.actions[:self.size],
                self.rewards[:self.size],
                self.values[:self.size],
                self.log_probs[:self.size],
                self.dones[:self.size]
            )
        else:
            indices = torch.arange(self.ptr, self.ptr + self.buffer_size, device=self.device) % self.buffer_size
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.values[indices],
                self.log_probs[indices],
                self.dones[indices]
            )


# Legacy class for backward compatibility
class PPOMemory(DiscreteMemory):
    """Legacy PPOMemory class - defaults to discrete actions for backward compatibility."""
    
    def __init__(self, buffer_size, state_dim, device='cpu', continuous_dim=None):
        if continuous_dim is not None:
            raise ValueError("Use ContinuousMemory directly for continuous action spaces")
        super().__init__(buffer_size, state_dim, device)