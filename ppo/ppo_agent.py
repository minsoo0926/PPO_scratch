"""PPO Agent implementation with separate discrete and continuous versions."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
from .networks import DiscreteActorCritic, ContinuousActorCritic, BaseActorCritic
from .memory import DiscreteMemory, ContinuousMemory, BaseMemory
from config import ENV_CONFIG


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
        kl_coef=0.0,
        target_kl=0.01,
        adaptive_kl=True,
        max_grad_norm=0.5,
        hidden_dim=64,
        buffer_size=2048,
        batch_size=64,
        epochs=10,
        n_envs=1,
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
        self.kl_coef = kl_coef
        self.target_kl = target_kl
        self.adaptive_kl = adaptive_kl
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.n_envs = n_envs
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'kl_loss': [],
            'kl_divergence': [],
            'total_loss': []
        }
        
        # Store hyperparameters for recreation during loading
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.buffer_size = buffer_size
        
        # Initialize network, optimizer and memory (implemented in subclasses)
        self.network = self._create_network(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = self._create_memory(n_envs, buffer_size, state_dim)

    @abstractmethod
    def _create_network(self, state_dim, action_dim, hidden_dim) -> BaseActorCritic:
        """Create the appropriate network architecture."""
        pass

    @abstractmethod
    def _create_memory(self, n_envs, buffer_size, state_dim) -> BaseMemory:
        """Create the appropriate memory buffer."""
        pass

    @abstractmethod
    def get_action(self, states, deterministic=False):
        """Select action for given state."""
        pass

    def store_experience(self, idx_env, state, action, reward, value, log_prob, done):
        """Store experience in memory buffer."""
        self.memory.store(idx_env, state, action, reward, value, log_prob, done)

    def normalize_rewards(self, rewards):
        """Normalize rewards across the batch."""
        with torch.no_grad():
            shape = rewards.shape
            rewards = rewards.to(self.device)
            rewards = rewards.view(-1)
            rewards = self.network.forward_rew_rs(rewards)
            rewards = rewards.view(shape)
        return rewards

    @torch.no_grad()
    def compute_advantages(self, rewards, values, dones, next_value):
        """Compute GAE (Generalized Advantage Estimation) advantages."""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        batch_len = len(rewards[0])

        gae = torch.zeros(size=(self.n_envs, 1), device=self.device)

        for t in reversed(range(batch_len)):
            next_val = next_value if t == batch_len - 1 else values[:, t + 1:t + 2]
            next_non_terminal = torch.full_like(rewards[:, t:t+1], 1.0) - dones[:, t:t+1].float()
            delta = rewards[:, t:t+1] + self.gamma * next_val * next_non_terminal - values[:, t:t+1]
            gae = delta + self.gamma * self.lam * next_non_terminal * gae
            advantages[:, t:t+1] = gae
            returns[:, t:t+1] = advantages[:, t:t+1] + values[:, t:t+1]

        advantages = (advantages - advantages.mean(-1, keepdim=True)) / (advantages.std(-1, keepdim=True) + 1e-8)
        return advantages, returns

    def compute_kl_divergence(self, old_log_probs, new_log_probs):
        log_ratio = new_log_probs - old_log_probs
        kl_div = (torch.exp(log_ratio) - 1 - log_ratio).mean().clamp(min=0.0)
        return kl_div

    @abstractmethod
    def update(self, next_state=None):
        """Update the policy using PPO algorithm."""
        pass

    def save(self, filepath, episode_rewards=None, episode_lengths=None):
        """Save the model with metadata."""
        save_dict = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            # Save episode statistics for resume
            'episode_rewards': episode_rewards if episode_rewards is not None else [],
            'episode_lengths': episode_lengths if episode_lengths is not None else [],
            # Save normalizer states explicitly
            'obs_rms_mean': self.network.obs_rms.mean.clone(),
            'obs_rms_var': self.network.obs_rms.var.clone(), 
            'obs_rms_count': self.network.obs_rms.count.clone(),
            'rew_rs_var': self.network.rew_rs.var.clone(),
            'rew_rs_count': self.network.rew_rs.count.clone(),

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
                'kl_coef': self.kl_coef,
                'target_kl': self.target_kl,
                'adaptive_kl': self.adaptive_kl,
                'max_grad_norm': self.max_grad_norm,
                'batch_size': self.batch_size,
                'epochs': self.epochs,
                'n_envs': self.n_envs,
            }
        }
        
        if isinstance(self, ContinuousPPOAgent):
            save_dict['model_metadata']['action_low'] = self.action_low
            save_dict['model_metadata']['action_high'] = self.action_high

        torch.save(save_dict, filepath)
    
    @classmethod
    def load(cls, filepath, strict=True):
        """
        Load the model and automatically adjust current model to match saved model dimensions.
        
        Args:
            filepath (str): Path to the saved model
            strict (bool): Whether to use strict loading for state dict
        """
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
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
                lr=ENV_CONFIG.get('lr', 1e-4) if 'lr' in ENV_CONFIG else metadata['lr'],
                buffer_size=ENV_CONFIG.get('buffer_size', 2048) if 'buffer_size' in ENV_CONFIG else metadata['buffer_size'],
                device=ENV_CONFIG.get('device', 'cpu') if 'device' in ENV_CONFIG else metadata.get('device', 'cpu'),
                gamma=ENV_CONFIG.get('gamma', 0.99) if 'gamma' in ENV_CONFIG else metadata.get('gamma', 0.99),
                lam=ENV_CONFIG.get('lam', 0.95) if 'lam' in ENV_CONFIG else metadata.get('lam', 0.95),
                clip_ratio=ENV_CONFIG.get('clip_ratio', 0.2) if 'clip_ratio' in ENV_CONFIG else metadata.get('clip_ratio', 0.2),
                value_coef=ENV_CONFIG.get('value_coef', 0.5) if 'value_coef' in ENV_CONFIG else metadata.get('value_coef', 0.5),
                entropy_coef=ENV_CONFIG.get('entropy_coef', 0.01) if 'entropy_coef' in ENV_CONFIG else metadata.get('entropy_coef', 0.01),
                kl_coef=ENV_CONFIG.get('kl_coef', 0.0) if 'kl_coef' in ENV_CONFIG else metadata.get('kl_coef', 0.0),
                target_kl=ENV_CONFIG.get('target_kl', 0.01) if 'target_kl' in ENV_CONFIG else metadata.get('target_kl', 0.01),
                adaptive_kl=ENV_CONFIG.get('adaptive_kl', True) if 'adaptive_kl' in ENV_CONFIG else metadata.get('adaptive_kl', True),
                max_grad_norm=ENV_CONFIG.get('max_grad_norm', 0.5) if 'max_grad_norm' in ENV_CONFIG else metadata.get('max_grad_norm', 0.5),
                batch_size=ENV_CONFIG.get('batch_size', 64) if 'batch_size' in ENV_CONFIG else metadata.get('batch_size', 64),
                epochs=ENV_CONFIG.get('epochs', 10) if 'epochs' in ENV_CONFIG else metadata.get('epochs', 10),
                n_envs=ENV_CONFIG.get('n_envs', 16) if 'n_envs' in ENV_CONFIG else metadata.get('n_envs', 1)
            )
        elif metadata.get('agent_type') == 'ContinuousPPOAgent':
            agent = ContinuousPPOAgent(
                state_dim=metadata['state_dim'],
                action_dim=metadata['action_dim'],
                action_low=metadata['action_low'],
                action_high=metadata['action_high'],
                hidden_dim=metadata['hidden_dim'],
                lr=ENV_CONFIG.get('lr', 1e-4) if 'lr' in ENV_CONFIG else metadata['lr'],
                buffer_size=ENV_CONFIG.get('buffer_size', 2048) if 'buffer_size' in ENV_CONFIG else metadata['buffer_size'],
                device=ENV_CONFIG.get('device', 'cpu') if 'device' in ENV_CONFIG else metadata.get('device', 'cpu'),
                gamma=ENV_CONFIG.get('gamma', 0.99) if 'gamma' in ENV_CONFIG else metadata.get('gamma', 0.99),
                lam=ENV_CONFIG.get('lam', 0.95) if 'lam' in ENV_CONFIG else metadata.get('lam', 0.95),
                clip_ratio=ENV_CONFIG.get('clip_ratio', 0.2) if 'clip_ratio' in ENV_CONFIG else metadata.get('clip_ratio', 0.2),
                value_coef=ENV_CONFIG.get('value_coef', 0.5) if 'value_coef' in ENV_CONFIG else metadata.get('value_coef', 0.5),
                entropy_coef=ENV_CONFIG.get('entropy_coef', 0.01) if 'entropy_coef' in ENV_CONFIG else metadata.get('entropy_coef', 0.01),
                kl_coef=ENV_CONFIG.get('kl_coef', 0.0) if 'kl_coef' in ENV_CONFIG else metadata.get('kl_coef', 0.0),
                target_kl=ENV_CONFIG.get('target_kl', 0.01) if 'target_kl' in ENV_CONFIG else metadata.get('target_kl', 0.01),
                adaptive_kl=ENV_CONFIG.get('adaptive_kl', True) if 'adaptive_kl' in ENV_CONFIG else metadata.get('adaptive_kl', True),
                max_grad_norm=ENV_CONFIG.get('max_grad_norm', 0.5) if 'max_grad_norm' in ENV_CONFIG else metadata.get('max_grad_norm', 0.5),
                batch_size=ENV_CONFIG.get('batch_size', 64) if 'batch_size' in ENV_CONFIG else metadata.get('batch_size', 64),
                epochs=ENV_CONFIG.get('epochs', 10) if 'epochs' in ENV_CONFIG else metadata.get('epochs', 10),
                n_envs=ENV_CONFIG.get('n_envs', 16) if 'n_envs' in ENV_CONFIG else metadata.get('n_envs', 1)
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
        
        # Restore normalizer states explicitly
        try:
            if 'obs_rms_mean' in checkpoint:
                agent.network.obs_rms.mean.copy_(checkpoint['obs_rms_mean'])
                agent.network.obs_rms.var.copy_(checkpoint['obs_rms_var'])
                agent.network.obs_rms.count.copy_(checkpoint['obs_rms_count'])
                print(f"✓ Observation normalizer restored (count: {agent.network.obs_rms.count.item():.0f})")
            
            if 'rew_rs_var' in checkpoint:
                agent.network.rew_rs.var.copy_(checkpoint['rew_rs_var'])
                agent.network.rew_rs.count.copy_(checkpoint['rew_rs_count'])
                print(f"✓ Reward normalizer restored (count: {agent.network.rew_rs.count.item():.0f}, std: {torch.sqrt(agent.network.rew_rs.var).item():.4f})")
        except Exception as e:
            print(f"WARNING: Failed to restore normalizer states: {e}")
        first_param = next(iter(agent.optimizer.state))
        opt_step = agent.optimizer.state[first_param].get('step', 0)
        print(f"✓ DEBUG- optimizer step: {opt_step}")
        return agent
    
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
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        try:
            self.network.load_state_dict(checkpoint['network_state_dict'], strict=strict)
            print(f"✓ Network weights loaded from {filepath}")
        except RuntimeError as e:
            if not strict:
                raise
            print(f"Failed with strict=True, trying strict=False: {e}")
            self.network.load_state_dict(checkpoint['network_state_dict'], strict=False)
            print(f"✓ Network weights loaded from {filepath} (some parameters ignored)")
        
        # Restore normalizer states for consistent behavior
        try:
            if 'obs_rms_mean' in checkpoint:
                self.network.obs_rms.mean.copy_(checkpoint['obs_rms_mean'])
                self.network.obs_rms.var.copy_(checkpoint['obs_rms_var'])
                self.network.obs_rms.count.copy_(checkpoint['obs_rms_count'])
                print(f"✓ Observation normalizer restored (count: {self.network.obs_rms.count.item():.0f})")
            
            if 'rew_rs_var' in checkpoint:
                self.network.rew_rs.var.copy_(checkpoint['rew_rs_var'])
                self.network.rew_rs.count.copy_(checkpoint['rew_rs_count'])
                print(f"✓ Reward normalizer restored (count: {self.network.rew_rs.count.item():.0f}, std: {torch.sqrt(self.network.rew_rs.var).item():.4f})")
        except Exception as e:
            print(f"WARNING: Failed to restore normalizer states: {e}")


class DiscretePPOAgent(BasePPOAgent):
    """PPO Agent for discrete action spaces."""

    def _create_network(self, state_dim, action_dim, hidden_dim) -> DiscreteActorCritic:
        """Create discrete action network."""
        return DiscreteActorCritic(state_dim, action_dim, hidden_dim).to(self.device)

    def _create_memory(self, n_envs, buffer_size, state_dim) -> DiscreteMemory:
        """Create discrete action memory buffer."""
        return DiscreteMemory(n_envs, buffer_size, state_dim, self.device)

    def get_action(self, states, deterministic=False):
        """Select actions for batch of states (for vectorized environments)."""
        # Convert to tensor safely
        if not isinstance(states, torch.Tensor):
            # Convert any array-like input to tensor directly
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        else:
            states = states.to(self.device)
        
        with torch.no_grad():
            actions, log_probs, _, values = self.network.get_action_and_value(states, deterministic=deterministic)
        
        return actions, log_probs, values

    def update(self, next_state=None):
        """Update the policy using PPO algorithm."""
        if len(self.memory) < self.batch_size:
            return
        
        # Get all experiences from memory
        states, actions, rewards, values, old_log_probs, dones = self.memory.get()
        rewards = self.normalize_rewards(rewards)

        # Compute next value for advantage calculation
        next_value = torch.zeros(size=(self.n_envs, 1), device=self.device)
        if next_state is not None:
            with torch.no_grad():
                _, next_value = self.network(next_state)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones, next_value)
        
        # Convert to tensors
        old_log_probs = old_log_probs.detach()
        
        # Flatten the first dimension
        states = states.view(-1, self.state_dim)
        actions = actions.view(-1)
        old_log_probs = old_log_probs.view(-1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        values = values.view(-1)

        # Update observation normalization
        self.network.update_obs_rms(states)

        # Training loop
        epoch_kl_divs = []
        early_stop = False
        
        for epoch in range(self.epochs):
            if early_stop:
                break
                
            # Create mini-batches
            indices = torch.randperm(len(states), device=self.device)
            
            batch_kl_divs = []
            
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
                
                # Compute KL divergence
                kl_div = self.compute_kl_divergence(batch_old_log_probs, new_log_probs.squeeze(-1))
                batch_kl_divs.append(kl_div.item())
                
                # Early stopping based on KL divergence
                if self.adaptive_kl and kl_div > 1.5 * self.target_kl:
                    early_stop = True
                    break
                
                # Compute policy loss with clipped log probability ratio
                log_ratio = new_log_probs.squeeze(-1) - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.MSELoss()(new_values.squeeze(-1), batch_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Compute KL penalty loss
                kl_loss = self.kl_coef * kl_div
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss + kl_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store training statistics
                self.training_stats['policy_loss'].append(policy_loss.item())
                self.training_stats['value_loss'].append(value_loss.item())
                self.training_stats['entropy_loss'].append(entropy_loss.item())
                self.training_stats['kl_loss'].append(kl_loss.item())
                self.training_stats['kl_divergence'].append(kl_div.item())
                self.training_stats['total_loss'].append(total_loss.item())
            
            # Update KL coefficient adaptively
            if batch_kl_divs and self.adaptive_kl:
                epoch_kl = np.mean(batch_kl_divs)
                epoch_kl_divs.append(epoch_kl)
                
                if epoch_kl < self.target_kl / 1.5:
                    self.kl_coef *= 0.5
                elif epoch_kl > self.target_kl * 1.5:
                    self.kl_coef *= 2.0
                
                # Clamp KL coefficient to reasonable bounds
                self.kl_coef = np.clip(self.kl_coef, 1e-4, 1.0)
        
        # Clear memory after update
        self.memory.clear()


class ContinuousPPOAgent(BasePPOAgent):
    """PPO Agent for continuous action spaces."""

    def __init__(self, state_dim, action_dim, action_low=np.array([-1.0]), action_high=np.array([1.0]), **kwargs):
        """Initialize continuous PPO agent with action bounds."""
        self.action_low = action_low
        self.action_high = action_high
        super().__init__(state_dim, action_dim, **kwargs)

    def _create_network(self, state_dim, action_dim, hidden_dim) -> ContinuousActorCritic:
        """Create continuous action network."""
        return ContinuousActorCritic(
            state_dim, action_dim, hidden_dim, 
            self.action_low, self.action_high
        ).to(self.device)

    def _create_memory(self, n_envs, buffer_size, state_dim) -> ContinuousMemory:
        """Create continuous action memory buffer."""
        return ContinuousMemory(n_envs, buffer_size, state_dim, self.action_dim, self.device)

    def get_action(self, states, deterministic=False):
        """Select actions for batch of states (for vectorized environments)."""
        # Convert to tensor safely
        if not isinstance(states, torch.Tensor):
            # Convert any array-like input to tensor directly
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        else:
            states = states.to(self.device)
        
        with torch.no_grad():
            actions, log_probs, _, values = self.network.get_action_and_value(states, deterministic=deterministic)
        
        return actions, log_probs, values

    def update(self, next_state=None):
        """Update the policy using PPO algorithm."""
        if len(self.memory) < self.batch_size:
            return
        
        # Get all experiences from memory
        states, actions, rewards, values, old_log_probs, dones = self.memory.get()
        rewards = self.normalize_rewards(rewards)

        # Compute next value for advantage calculation
        next_value = torch.zeros(size=(self.n_envs, 1), device=self.device)
        if next_state is not None:
            with torch.no_grad():
                _, next_value, _ = self.network(next_state)
        
        # Compute advantages and returns
        advantages, returns = self.compute_advantages(rewards, values, dones, next_value)
        
        # Convert to tensors and sum log probs over action dimensions
        old_log_probs = old_log_probs.detach()

        # Flatten the first dimension
        states = states.view(-1, self.state_dim)
        actions = actions.view(-1, self.action_dim)
        old_log_probs = old_log_probs.view(-1)
        advantages = advantages.view(-1)
        returns = returns.view(-1)
        values = values.view(-1)

        # Update observation normalization
        self.network.update_obs_rms(states)

        # Training loop
        epoch_kl_divs = []
        early_stop = False
        
        for epoch in range(self.epochs):
            if early_stop:
                break
                
            # Create mini-batches
            indices = torch.randperm(len(states), device=self.device)
            
            batch_kl_divs = []
            
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
                
                # Compute KL divergence
                kl_div = self.compute_kl_divergence(batch_old_log_probs, new_log_probs.squeeze(-1))
                batch_kl_divs.append(kl_div.item())
                
                # Early stopping based on KL divergence
                if self.adaptive_kl and kl_div > 1.5 * self.target_kl:
                    early_stop = True
                    break
                
                # Compute policy loss with clipped log probability ratio
                log_ratio = new_log_probs.squeeze(-1) - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = nn.MSELoss()(new_values.squeeze(-1), batch_returns)
                
                # Compute entropy loss
                entropy_loss = -entropy.mean()
                
                # Compute KL penalty loss
                kl_loss = self.kl_coef * kl_div
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss + kl_loss

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Store training statistics
                self.training_stats['policy_loss'].append(policy_loss.item())
                self.training_stats['value_loss'].append(value_loss.item())
                self.training_stats['entropy_loss'].append(entropy_loss.item())
                self.training_stats['kl_loss'].append(kl_loss.item())
                self.training_stats['kl_divergence'].append(kl_div.item())
                self.training_stats['total_loss'].append(total_loss.item())
            
            # Update KL coefficient adaptively
            if batch_kl_divs and self.adaptive_kl:
                epoch_kl = np.mean(batch_kl_divs)
                epoch_kl_divs.append(epoch_kl)
                
                if epoch_kl < self.target_kl / 1.5:
                    self.kl_coef *= 0.5
                elif epoch_kl > self.target_kl * 1.5:
                    self.kl_coef *= 2.0
                
                # Clamp KL coefficient to reasonable bounds
                self.kl_coef = np.clip(self.kl_coef, 1e-4, 1.0)
        
        # Clear memory after update
        self.memory.clear()


# Legacy class for backward compatibility
class PPOAgent:
    """Factory class that creates appropriate PPO agent based on action space."""

    def __new__(cls, state_dim, action_dim, continuous_space=None, action_low=np.array([-1.0]), action_high=np.array([1.0]), **kwargs):
        if continuous_space is not None:
            # Extract bounds from continuous_space if provided
            if hasattr(continuous_space, 'low') and hasattr(continuous_space, 'high'):
                action_low = continuous_space.low
                action_high = continuous_space.high
            return ContinuousPPOAgent(state_dim, action_dim, action_low, action_high, **kwargs)
        else:
            return DiscretePPOAgent(state_dim, action_dim, **kwargs)