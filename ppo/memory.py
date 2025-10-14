"""Rollout buffer for PPO, using PyTorch tensors."""

import torch
from torch.utils.data.dataset import Dataset


class RolloutBuffer:
    """
    Rollout buffer implemented with PyTorch tensors, inspired by Stable Baselines 3.
    """

    def __init__(self, buffer_size, state_dim, action_dim, device, gae_lambda, gamma, n_envs):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs
        self.action_dim_storage = action_dim if action_dim is not None else 1

        # Determine action dtype based on action_dim
        self.is_discrete = action_dim is None
        self.action_dtype = torch.long if self.is_discrete else torch.float32

        self.observations = torch.zeros((self.buffer_size, self.n_envs, self.state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim_storage), dtype=self.action_dtype, device=device)
        self.rewards = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.full = False

    def __len__(self):
        return self.buffer_size * self.n_envs

    def add(self, obs, action, reward, done, value, log_prob):
        """
        Adds a transition to the buffer.
        All inputs are expected to be PyTorch tensors on the correct device.
        """
        if self.full:
            raise ValueError("Buffer is full. Call .clear() before adding new data.")

        self.observations[self.ptr] = obs
        if self.is_discrete:
            action = action.reshape(-1, 1)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done.float()
        self.values[self.ptr] = value.flatten()
        self.log_probs[self.ptr] = log_prob
        
        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values, dones):
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.
        """
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.float()
                next_values = last_values.flatten()
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values

    def get(self):
        """
        Returns all data in the buffer and flattens it.
        """
        assert self.full, "Rollout buffer is not full. Call compute_returns_and_advantage first."
        
        # Flatten the data
        # Shape: (n_steps, n_envs, *shape) -> (n_steps * n_envs, *shape)
        observations = self.observations.swapaxes(0, 1).reshape(-1, self.state_dim)
        actions = self.actions.swapaxes(0, 1).reshape(-1, self.action_dim_storage)
        if self.is_discrete:
            actions = actions.long().flatten()
        values = self.values.swapaxes(0, 1).reshape(-1)
        log_probs = self.log_probs.swapaxes(0, 1).reshape(-1)
        advantages = self.advantages.swapaxes(0, 1).reshape(-1)
        returns = self.returns.swapaxes(0, 1).reshape(-1)

        return RolloutBufferSamples(
            observations=observations,
            actions=actions,
            old_values=values,
            old_log_prob=log_probs,
            advantages=advantages,
            returns=returns,
        )

    def clear(self):
        self.ptr = 0
        self.full = False


class RolloutBufferSamples(Dataset):
    def __init__(self, observations, actions, old_values, old_log_prob, advantages, returns):
        self.observations = observations
        self.actions = actions
        self.old_values = old_values
        self.old_log_prob = old_log_prob
        self.advantages = advantages
        self.returns = returns

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.old_values[idx],
            self.old_log_prob[idx],
            self.advantages[idx],
            self.returns[idx],
        )