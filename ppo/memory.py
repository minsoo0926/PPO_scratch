"""
Rollout buffer for PPO, inspired by Stable Baselines 3.
"""
import torch
from torch.utils.data.dataset import Dataset
from gymnasium import spaces
from typing import NamedTuple

class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

class RolloutBuffer:
    """
    Rollout buffer implemented with PyTorch tensors, inspired by Stable Baselines 3.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device,
        gae_lambda: float = 0.95,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.n_envs = n_envs

        self.obs_shape = observation_space.shape
        self.action_dim = action_space.shape[0] if isinstance(action_space, spaces.Box) else 1

        self.observations = torch.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=torch.float32, device=device)
        if isinstance(action_space, spaces.Box):
            self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.float32, device=device)
        else: # Discrete
            self.actions = torch.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=torch.long, device=device)
        
        self.rewards = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.returns = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.values = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.log_probs = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((self.buffer_size, self.n_envs), dtype=torch.float32, device=device)
        
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward, done, value, log_prob):
        """ Adds a transition to the buffer. """
        self.observations[self.pos] = obs.clone()
        if isinstance(self.action_space, spaces.Discrete):
            action = action.reshape(-1, 1)
        self.actions[self.pos] = action.clone()
        self.rewards[self.pos] = reward.clone()
        self.dones[self.pos] = done.clone()
        self.values[self.pos] = value.clone().flatten()
        self.log_probs[self.pos] = log_prob.clone()
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: torch.Tensor, dones: torch.Tensor):
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.
        """
        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.float()
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        
        self.returns = self.advantages + self.values

    def get(self, batch_size: int) -> Dataset:
        """
        Returns all data in the buffer and flattens it for training.
        """
        assert self.full, "Rollout buffer is not full"
        
        # Flatten the data
        # (n_steps, n_envs, *shape) -> (n_steps * n_envs, *shape)
        observations = self.observations.swapaxes(0, 1).reshape(-1, *self.obs_shape)
        actions = self.actions.swapaxes(0, 1).reshape(-1, self.action_dim)
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
        self.pos = 0
        self.full = False
