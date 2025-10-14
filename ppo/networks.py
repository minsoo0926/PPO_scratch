"""
Neural network policies for PPO, inspired by Stable Baselines 3.
"""
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Categorical, Normal
from typing import Tuple, Union

class Policy(nn.Module):
    """
    A standard actor-critic policy network (MLP).

    :param observation_space: The observation space of the environment
    :param action_space: The action space of the environment
    :param hidden_dim: Dimension of the hidden layers
    :param net_arch: The specification of the policy and value networks.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = observation_space.shape[0]
        
        # Shared network
        mlp_extractor = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.latent_pi = mlp_extractor
        self.latent_vf = mlp_extractor

        # Action and value heads
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
            self.action_net = nn.Linear(hidden_dim, self.action_dim)
            self.action_dist = "categorical"
        elif isinstance(action_space, spaces.Box):
            self.action_dim = action_space.shape[0]
            self.action_net = nn.Linear(hidden_dim, self.action_dim)
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
            self.action_dist = "gaussian"
        else:
            raise NotImplementedError(f"Action space {type(action_space)} not supported")

        self.value_net = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in the actor-critic network.

        :param obs: Observation
        :param deterministic: Whether to sample or take the mode of the distribution
        :return: action, value, log_probability_of_action
        """
        latent_pi = self.latent_pi(obs)
        latent_vf = self.latent_vf(obs)
        
        value = self.value_net(latent_vf)
        
        if self.action_dist == "categorical":
            action_logits = self.action_net(latent_pi)
            distribution = Categorical(logits=action_logits)
        elif self.action_dist == "gaussian":
            mean = self.action_net(latent_pi)
            std = torch.exp(self.log_std)
            distribution = Normal(mean, std)
        else:
            raise NotImplementedError
        
        if deterministic:
            if self.action_dist == "categorical":
                action = torch.argmax(distribution.probs, dim=1)
            else:
                action = distribution.mean
        else:
            action = distribution.sample()
            
        log_prob = distribution.log_prob(action)
        if self.action_dist == "gaussian":
            log_prob = log_prob.sum(axis=-1)
            
        return action, value, log_prob

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param action: Action
        :return: value, log_probability_of_action, entropy
        """
        latent_pi = self.latent_pi(obs)
        latent_vf = self.latent_vf(obs)
        
        value = self.value_net(latent_vf)
        
        if self.action_dist == "categorical":
            action_logits = self.action_net(latent_pi)
            distribution = Categorical(logits=action_logits)
        elif self.action_dist == "gaussian":
            mean = self.action_net(latent_pi)
            std = torch.exp(self.log_std)
            distribution = Normal(mean, std)
        else:
            raise NotImplementedError
            
        log_prob = distribution.log_prob(action)
        if self.action_dist == "gaussian":
            log_prob = log_prob.sum(axis=-1)
            
        entropy = distribution.entropy()
        if self.action_dist == "gaussian":
            entropy = entropy.sum(axis=-1)
            
        return value, log_prob, entropy
