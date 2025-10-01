"""Neural network architectures for PPO agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        """
        Initialize Actor-Critic network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden layer dimension
        """
        super().__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy network)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head (value network)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        Forward pass through both actor and critic.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (action_logits, state_value)
        """
        shared_features = self.shared_layers(state)
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        
        return action_logits, state_value
    
    def get_action_and_value(self, state, action=None):
        """
        Get action, log probability, and value for given state.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor, optional): Action tensor for evaluation
            
        Returns:
            tuple: (action, log_prob, entropy, value)
        """
        action_logits, value = self.forward(state)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, log_prob, entropy, value
    
    def evaluate(self, state, action):
        """
        Evaluate actions for given states.
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            tuple: (log_prob, entropy, value)
        """
        action_logits, value = self.forward(state)
        probs = Categorical(logits=action_logits)
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return log_prob, entropy, value