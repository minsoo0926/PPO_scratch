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
        
        # Store hyperparameters for recreation during loading
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.buffer_size = buffer_size
        
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
        """Save the model with metadata."""
        save_dict = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            # Save model metadata for dimension validation and recreation
            'model_metadata': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                'lr': self.lr,
                'buffer_size': self.buffer_size,
                'agent_type': self.__class__.__name__,
                'device': str(self.device),
                'gamma': self.gamma,
                'lam': self.lam,
                'clip_ratio': self.clip_ratio,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            }
        }
        
        if isinstance(self, ContinuousPPOAgent):
            save_dict['model_metadata']['action_low'] = self.action_low
            save_dict['model_metadata']['action_high'] = self.action_high

        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
        print(f"  - State dim: {self.state_dim}")
        print(f"  - Action dim: {self.action_dim}")
        print(f"  - Agent type: {self.__class__.__name__}")
    
    @classmethod
    def load(cls, filepath, strict=True):
        """
        Load the model and automatically adjust current model to match saved model dimensions.
        
        Args:
            filepath (str): Path to the saved model
            strict (bool): Whether to use strict loading for state dict
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Check if metadata exists (for backward compatibility)
        if 'model_metadata' in checkpoint:
            metadata = checkpoint['model_metadata']
            saved_state_dim = metadata.get('state_dim')
            saved_action_dim = metadata.get('action_dim')
            saved_agent_type = metadata.get('agent_type')
            
            print(f"Loading model from {filepath}")
            print(f"  - Saved state dim: {saved_state_dim}")
            print(f"  - Saved action dim: {saved_action_dim}")
            print(f"  - Saved agent type: {saved_agent_type}")

        # Create new agent instance with saved dimensions
        if metadata.get('agent_type') == 'DiscretePPOAgent':
            agent = DiscretePPOAgent(
                state_dim=metadata['state_dim'],
                action_dim=metadata['action_dim'],
                hidden_dim=metadata['hidden_dim'],
                lr=metadata['lr'],
                buffer_size=metadata['buffer_size'],
                device=metadata.get('device', 'cpu'),
                gamma=metadata.get('gamma', 0.99),
                lam=metadata.get('lam', 0.95),
                clip_ratio=metadata.get('clip_ratio', 0.2),
                value_coef=metadata.get('value_coef', 0.5),
                entropy_coef=metadata.get('entropy_coef', 0.01),
                max_grad_norm=metadata.get('max_grad_norm', 0.5),
                batch_size=metadata.get('batch_size', 64),
                epochs=metadata.get('epochs', 10)
            )
        elif metadata.get('agent_type') == 'ContinuousPPOAgent':
            agent = ContinuousPPOAgent(
                state_dim=metadata['state_dim'],
                action_dim=metadata['action_dim'],
                action_low=metadata['action_low'],
                action_high=metadata['action_high'],
                hidden_dim=metadata['hidden_dim'],
                lr=metadata['lr'],
                buffer_size=metadata['buffer_size'],
                device=metadata.get('device', 'cpu'),
                gamma=metadata.get('gamma', 0.99),
                lam=metadata.get('lam', 0.95),
                clip_ratio=metadata.get('clip_ratio', 0.2),
                value_coef=metadata.get('value_coef', 0.5),
                entropy_coef=metadata.get('entropy_coef', 0.01),
                max_grad_norm=metadata.get('max_grad_norm', 0.5),
                batch_size=metadata.get('batch_size', 64),
                epochs=metadata.get('epochs', 10)
            )
        else:
            raise ValueError(f"Unknown agent type: {metadata.get('agent_type')}")

        # Load network state
        try:
            agent.network.load_state_dict(checkpoint['network_state_dict'], strict=strict)
            print("✓ Network state loaded successfully")
        except RuntimeError as e:
            print(f"WARNING: Failed to load network state with strict=True: {e}")
            print("Trying with strict=False...")
            try:
                agent.network.load_state_dict(checkpoint['network_state_dict'], strict=False)
                print("✓ Network state loaded with strict=False (some parameters may be missing/extra)")
            except RuntimeError as e2:
                raise RuntimeError(f"Failed to load network state even with strict=False: {e2}")
        
        # Load optimizer state (with error handling)
        try:
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✓ Optimizer state loaded successfully")
        except Exception as e:
            print(f"WARNING: Failed to load optimizer state: {e}")
            print("Continuing without optimizer state (this is usually fine for inference)")
        
        # Load training stats
        agent.training_stats = checkpoint.get('training_stats', agent.training_stats)
        print("✓ Training stats loaded successfully")
    
    @classmethod
    def get_model_info(cls, filepath):
        """
        Get model information without loading the full model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            dict: Model metadata if available, None otherwise
        """
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            if 'model_metadata' in checkpoint:
                return checkpoint['model_metadata']
            else:
                return None
        except Exception as e:
            print(f"Error reading model file: {e}")
            return None
    
    @classmethod
    def is_compatible(cls, filepath, state_dim, action_dim, agent_type=None):
        """
        Check if saved model is compatible with given dimensions and agent type.
        
        Args:
            filepath (str): Path to the saved model
            state_dim (int): Required state dimension
            action_dim (int): Required action dimension
            agent_type (str, optional): Required agent type
            
        Returns:
            tuple: (is_compatible: bool, info: str)
        """
        info = cls.get_model_info(filepath)
        
        if info is None:
            return False, "No metadata found (old model format)"
        
        issues = []
        
        if info.get('state_dim') != state_dim:
            issues.append(f"State dim mismatch: saved={info.get('state_dim')}, required={state_dim}")
        
        if info.get('action_dim') != action_dim:
            issues.append(f"Action dim mismatch: saved={info.get('action_dim')}, required={action_dim}")
        
        if agent_type and info.get('agent_type') != agent_type:
            issues.append(f"Agent type mismatch: saved={info.get('agent_type')}, required={agent_type}")
        
        if issues:
            return False, "; ".join(issues)
        else:
            return True, "Model is compatible"
    
    def load_network_only(self, filepath, strict=True):
        """
        Load only the network weights, skipping optimizer and other data.
        Useful for inference or transfer learning.
        
        Args:
            filepath (str): Path to the saved model
            strict (bool): Whether to use strict loading for state dict
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        try:
            self.network.load_state_dict(checkpoint['network_state_dict'], strict=strict)
            print(f"✓ Network weights loaded from {filepath}")
        except RuntimeError as e:
            if not strict:
                raise
            print(f"Failed with strict=True, trying strict=False: {e}")
            self.network.load_state_dict(checkpoint['network_state_dict'], strict=False)
            print(f"✓ Network weights loaded from {filepath} (some parameters ignored)")


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