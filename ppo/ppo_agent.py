"""PPO Agent implementation with separate discrete and continuous versions."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
from .networks import DiscreteActorCritic, ContinuousActorCritic
from .memory import DiscreteMemory, ContinuousMemory


class BasePPOAgent(ABC):
    """Base class for PPO agents."""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        hidden_dim=64,
        buffer_size=2048,
        batch_size=64,
        epochs=10,
        device='cpu'
    ):
        """Initialize base PPO agent."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': []
        }
        
        # Initialize network, optimizer and memory (implemented in subclasses)
        self.network = self._create_network(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = self._create_memory(buffer_size, state_dim)

    @abstractmethod
    def _create_network(self, state_dim, action_dim, hidden_dim):
        """Create the appropriate network architecture."""
        pass

    @abstractmethod
    def _create_memory(self, buffer_size, state_dim):
        """Create the appropriate memory buffer."""
        pass

    @abstractmethod
    def get_action(self, state, deterministic=False):
        """Select action for given state."""
        pass

    def store_experience(self, state, action, reward, value, log_prob, done):
        """Store experience in memory buffer."""
        self.memory.store(state, action, reward, value, log_prob, done)
    
    def compute_advantages(self, rewards, values, dones, next_value=0):
        """Compute GAE (Generalized Advantage Estimation) advantages."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0
        next_val = next_value
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = next_val
            else:
                next_non_terminal = 1.0 - float(dones[t])
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns

    @abstractmethod
    def update(self, next_state=None):
        """Update the policy using PPO algorithm."""
        pass

    def save(self, filepath):
        """Save the model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
    
    def load(self, filepath):
        """Load the model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)


class DiscretePPOAgent(BasePPOAgent):
    """PPO Agent for discrete action spaces."""

    def _create_network(self, state_dim, action_dim, hidden_dim):
        """Create discrete action network."""
        return DiscreteActorCritic(state_dim, action_dim, hidden_dim).to(self.device)

    def _create_memory(self, buffer_size, state_dim):
        """Create discrete action memory buffer."""
        return DiscreteMemory(buffer_size, state_dim, self.device)

    def get_action(self, state, deterministic=False):
        """Select action for given state."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action_logits, value = self.network(state.unsqueeze(0))
                action = torch.argmax(action_logits, dim=-1)
                log_prob = torch.zeros(1, device=self.device)
            else:
                action, log_prob, _, value = self.network.get_action_and_value(state.unsqueeze(0))
        
        return action.item(), log_prob.item(), value.item()

    def update(self, next_state=None):
        """Update the policy using PPO algorithm."""
        if len(self.memory) < self.batch_size:
            return
        
        # Get all experiences from memory
        states, actions, rewards, values, old_log_probs, dones = self.memory.get()
        
        # Compute next value for advantage calculation
        next_value = 0
        if next_state is not None:
            if isinstance(next_state, np.ndarray):
                next_state = torch.from_numpy(next_state).float().to(self.device)
            with torch.no_grad():
                _, next_value = self.network(next_state.unsqueeze(0))
                next_value = next_value.item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        old_log_probs = old_log_probs.detach()
        
        # Training loop
        for epoch in range(self.epochs):
            # Create mini-batches
            indices = torch.randperm(len(states), device=self.device)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                if len(batch_indices) < self.batch_size:
                    continue
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                new_log_probs, entropy, new_values = self.network.evaluate(batch_states, batch_actions)
                
                # Compute policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.MSELoss()(new_values.squeeze(), batch_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store training statistics
                self.training_stats['policy_loss'].append(policy_loss.item())
                self.training_stats['value_loss'].append(value_loss.item())
                self.training_stats['entropy_loss'].append(entropy_loss.item())
                self.training_stats['total_loss'].append(total_loss.item())
        
        # Clear memory after update
        self.memory.clear()


class ContinuousPPOAgent(BasePPOAgent):
    """PPO Agent for continuous action spaces."""

    def __init__(self, state_dim, action_dim, action_low=-1.0, action_high=1.0, **kwargs):
        """Initialize continuous PPO agent with action bounds."""
        self.action_low = action_low
        self.action_high = action_high
        super().__init__(state_dim, action_dim, **kwargs)

    def _create_network(self, state_dim, action_dim, hidden_dim):
        """Create continuous action network."""
        return ContinuousActorCritic(
            state_dim, action_dim, hidden_dim, 
            self.action_low, self.action_high
        ).to(self.device)

    def _create_memory(self, buffer_size, state_dim):
        """Create continuous action memory buffer."""
        return ContinuousMemory(buffer_size, state_dim, self.action_dim, self.device)

    def get_action(self, state, deterministic=False):
        """Select action for given state."""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action_mean, value, _ = self.network(state.unsqueeze(0))
                # For deterministic actions, use the mean and compute action bounds  
                device = action_mean.device
                action_low = torch.tensor(self.action_low, device=device, dtype=torch.float32)
                action_high = torch.tensor(self.action_high, device=device, dtype=torch.float32)
                action_scale = (action_high - action_low) / 2.0
                action_bias = (action_high + action_low) / 2.0
                action = torch.tanh(action_mean) * action_scale + action_bias
                log_prob = torch.zeros(action.shape, device=self.device)
            else:
                action, log_prob, _, value = self.network.get_action_and_value(state.unsqueeze(0))
        
        return action.cpu().numpy().flatten(), log_prob.cpu().numpy().flatten(), value.item()

    def update(self, next_state=None):
        """Update the policy using PPO algorithm."""
        if len(self.memory) < self.batch_size:
            return
        
        # Get all experiences from memory
        states, actions, rewards, values, old_log_probs, dones = self.memory.get()
        
        # Compute next value for advantage calculation
        next_value = 0
        if next_state is not None:
            if isinstance(next_state, np.ndarray):
                next_state = torch.from_numpy(next_state).float().to(self.device)
            with torch.no_grad():
                _, next_value, _ = self.network(next_state.unsqueeze(0))
                next_value = next_value.item()
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors and sum log probs over action dimensions
        old_log_probs = old_log_probs.sum(dim=-1, keepdim=False).detach()
        
        # Training loop
        for epoch in range(self.epochs):
            # Create mini-batches
            indices = torch.randperm(len(states), device=self.device)
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                if len(batch_indices) < self.batch_size:
                    continue
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Forward pass
                new_log_probs, entropy, new_values = self.network.evaluate(batch_states, batch_actions)
                
                # Sum over action dimensions
                new_log_probs = new_log_probs.sum(dim=-1)
                entropy = entropy.sum(dim=-1)
                
                # Compute policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.MSELoss()(new_values.squeeze(), batch_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store training statistics
                self.training_stats['policy_loss'].append(policy_loss.item())
                self.training_stats['value_loss'].append(value_loss.item())
                self.training_stats['entropy_loss'].append(entropy_loss.item())
                self.training_stats['total_loss'].append(total_loss.item())
        
        # Clear memory after update
        self.memory.clear()


# Legacy class for backward compatibility
class PPOAgent:
    """Factory class that creates appropriate PPO agent based on action space."""
    
    def __new__(cls, state_dim, action_dim, continuous_space=None, action_low=-1.0, action_high=1.0, **kwargs):
        if continuous_space is not None:
            # Extract bounds from continuous_space if provided
            if hasattr(continuous_space, 'low') and hasattr(continuous_space, 'high'):
                action_low = continuous_space.low
                action_high = continuous_space.high
            return ContinuousPPOAgent(state_dim, action_dim, action_low, action_high, **kwargs)
        else:
            return DiscretePPOAgent(state_dim, action_dim, **kwargs)