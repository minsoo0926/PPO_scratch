"""Memory buffer for storing experiences in PPO."""

import torch
import numpy as np


class PPOMemory:
    """Memory buffer for PPO agent."""
    
    def __init__(self, buffer_size, state_dim, device='cpu', continuous_dim=None):
        """
        Initialize memory buffer.
        
        Args:
            buffer_size (int): Maximum size of the buffer
            state_dim (int): Dimension of state space
            device (str): Device to store tensors on
            continuous_dim (int): Dimension of continuous action space if applicable
        """
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        self.continuous_dim = continuous_dim
        
        # Initialize buffers
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        if continuous_dim:
            self.actions = torch.zeros((buffer_size, continuous_dim), dtype=torch.float32, device=device)
            self.log_probs = torch.zeros((buffer_size, continuous_dim), dtype=torch.float32, device=device)
        else:
            self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
            self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=device)
        
    def store(self, state, action, reward, value, log_prob, done):
        """
        Store a single experience.
        
        Args:
            state (torch.Tensor or np.ndarray): State
            action (int): Action taken
            reward (float): Reward received
            value (float): Value estimate
            log_prob (float): Log probability of action
            done (bool): Episode termination flag
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).float().to(self.device)
        if isinstance(log_prob, np.ndarray):
            log_prob = torch.from_numpy(log_prob).float().to(self.device)
        reward = float(reward)
        self.states[self.ptr] = state.to(self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def get(self):
        """
        Get all stored experiences.
        
        Returns:
            tuple: (states, actions, rewards, values, log_probs, dones)
        """
        assert self.size > 0, "Memory buffer is empty"
        
        if self.size < self.buffer_size:
            # Return only filled portion
            return (
                self.states[:self.size],
                self.actions[:self.size],
                self.rewards[:self.size],
                self.values[:self.size],
                self.log_probs[:self.size],
                self.dones[:self.size]
            )
        else:
            # Return full buffer in correct order
            indices = torch.arange(self.ptr, self.ptr + self.buffer_size, device=self.device) % self.buffer_size
            return (
                self.states[indices],
                self.actions[indices],
                self.rewards[indices],
                self.values[indices],
                self.log_probs[indices],
                self.dones[indices]
            )
    
    def clear(self):
        """Clear the memory buffer."""
        self.ptr = 0
        self.size = 0
    
    def is_full(self):
        """Check if buffer is full."""
        return self.size == self.buffer_size
    
    def __len__(self):
        """Return current size of buffer."""
        return self.size