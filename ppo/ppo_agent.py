"""PPO Agent implementation, inspired by SB3."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from abc import ABC, abstractmethod
from .networks import DiscreteActorCritic, ContinuousActorCritic, BaseActorCritic
from .memory import RolloutBuffer
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
        clip_vf=0.2,
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
        self.clip_vf = clip_vf
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
        
        # Initialize network, optimizer and memory
        self.network = self._create_network(state_dim, self._get_network_action_dim(), hidden_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.memory = RolloutBuffer(
            self.buffer_size, self.state_dim, self._get_buffer_action_dim(), self.device, self.lam, self.gamma, self.n_envs
        )

    @abstractmethod
    def _create_network(self, state_dim, action_dim, hidden_dim) -> BaseActorCritic:
        """Create the appropriate network architecture."""
        pass

    @abstractmethod
    def _get_network_action_dim(self) -> int:
        """Get the action dimension for the network (can differ from env action_dim)."""
        pass

    @abstractmethod
    def _get_buffer_action_dim(self):
        """Get the action dimension for the rollout buffer."""
        pass

    @abstractmethod
    def get_action(self, states, deterministic=False):
        """Select action for given state."""
        pass

    def compute_kl_divergence(self, old_log_probs, new_log_probs):
        log_ratio = new_log_probs - old_log_probs
        kl_div = (torch.exp(log_ratio) - 1 - log_ratio).mean().clamp(min=0.0)
        return kl_div

    def update(self, last_values, dones):
        """Update the policy using PPO algorithm."""
        
        self.memory.compute_returns_and_advantage(last_values, dones)
        rollout_data = self.memory.get()
        dataloader = DataLoader(rollout_data, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        epoch_kl_divs = []
        
        for epoch in range(self.epochs):
            early_stop = False
            batch_kl_divs = []

            for batch in dataloader:
                batch_states, batch_actions, batch_old_values, batch_old_log_probs, batch_advantages, batch_returns = batch

                # Normalize advantages within the batch
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # Forward pass
                new_log_probs, entropy, new_values = self.network.evaluate(batch_states, batch_actions)
                
                # Compute KL divergence
                kl_div = self.compute_kl_divergence(batch_old_log_probs, new_log_probs.squeeze(-1))
                batch_kl_divs.append(kl_div.item())
                
                # Early stopping based on KL divergence
                if self.adaptive_kl and kl_div > 1.5 * self.target_kl:
                    early_stop = True
                    break

                # Compute value loss
                if self.clip_vf > 0:
                    # Clipped value loss
                    value_loss_unclipped = (new_values.squeeze(-1) - batch_returns) ** 2
                    value_clipped = batch_old_values + torch.clamp(
                        new_values.squeeze(-1) - batch_old_values,
                        -self.clip_vf,
                        self.clip_vf,
                    )
                    value_loss_clipped = (value_clipped - batch_returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * ((new_values.squeeze(-1) - batch_returns) ** 2).mean()

                # Compute policy loss with clipped log probability ratio
                log_ratio = new_log_probs.squeeze(-1) - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
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
            
            if early_stop:
                break

            # Update KL coefficient adaptively
            if batch_kl_divs and self.adaptive_kl:
                epoch_kl = np.mean(batch_kl_divs)
                epoch_kl_divs.append(epoch_kl)
                
                if epoch_kl < self.target_kl / 1.5:
                    self.kl_coef = max(1e-4, self.kl_coef / 1.5)
                elif epoch_kl > self.target_kl * 1.5:
                    self.kl_coef = min(100.0, self.kl_coef * 1.5)
        
        # Clear memory after update
        self.memory.clear()

    def save(self, filepath, episode_rewards=None, episode_lengths=None):
        """Save the model with metadata."""
        save_dict = {
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'episode_rewards': episode_rewards if episode_rewards is not None else [],
            'episode_lengths': episode_lengths if episode_lengths is not None else [],
            'obs_rms_mean': self.network.obs_rms.mean.clone(),
            'obs_rms_var': self.network.obs_rms.var.clone(), 
            'obs_rms_count': self.network.obs_rms.count.clone(),
            'rew_rs_var': self.network.rew_rs.var.clone(),
            'rew_rs_count': self.network.rew_rs.count.clone(),
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
                'clip_vf': self.clip_vf,
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
        """Load the model."""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        metadata = checkpoint['model_metadata']
        
        agent_class = DiscretePPOAgent if metadata['agent_type'] == 'DiscretePPOAgent' else ContinuousPPOAgent
        
        # Override saved metadata with any new config from ENV_CONFIG
        for key in ENV_CONFIG:
            if key in metadata:
                metadata[key] = ENV_CONFIG[key]

        agent = agent_class(**metadata)
        
        agent.network.load_state_dict(checkpoint['network_state_dict'], strict=strict)
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.training_stats = checkpoint.get('training_stats', agent.training_stats)
        
        if 'obs_rms_mean' in checkpoint:
            agent.network.obs_rms.mean.copy_(checkpoint['obs_rms_mean'])
            agent.network.obs_rms.var.copy_(checkpoint['obs_rms_var'])
            agent.network.obs_rms.count.copy_(checkpoint['obs_rms_count'])
        
        if 'rew_rs_var' in checkpoint:
            agent.network.rew_rs.var.copy_(checkpoint['rew_rs_var'])
            agent.network.rew_rs.count.copy_(checkpoint['rew_rs_count'])
            
        return agent


class DiscretePPOAgent(BasePPOAgent):
    """PPO Agent for discrete action spaces."""

    def __init__(self, state_dim, action_dim=1, **kwargs):
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)

    def _create_network(self, state_dim, action_dim, hidden_dim) -> DiscreteActorCritic:
        return DiscreteActorCritic(state_dim, action_dim, hidden_dim).to(self.device)

    def _get_network_action_dim(self) -> int:
        # For discrete actions, the network output corresponds to the number of actions
        return self.action_dim

    def _get_buffer_action_dim(self):
        return None

    def get_action(self, states, deterministic=False):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        else:
            states = states.to(self.device)
        
        with torch.no_grad():
            actions, log_probs, _, values = self.network.get_action_and_value(states, deterministic=deterministic)
        
        return actions, log_probs, values


class ContinuousPPOAgent(BasePPOAgent):
    """PPO Agent for continuous action spaces."""

    def __init__(self, state_dim, action_dim, action_low=np.array([-1.0]), action_high=np.array([1.0]), **kwargs):
        self.action_low = action_low
        self.action_high = action_high
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)

    def _create_network(self, state_dim, action_dim, hidden_dim) -> ContinuousActorCritic:
        return ContinuousActorCritic(
            state_dim, action_dim, hidden_dim, 
            self.action_low, self.action_high
        ).to(self.device)

    def _get_network_action_dim(self) -> int:
        return self.action_dim

    def _get_buffer_action_dim(self):
        return self.action_dim

    def get_action(self, states, deterministic=False):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        else:
            states = states.to(self.device)
        
        with torch.no_grad():
            actions, log_probs, _, values = self.network.get_action_and_value(states, deterministic=deterministic)
        
        return actions, log_probs, values


# Factory class for convenience
class PPOAgent:
    def __new__(cls, state_dim, action_dim, continuous_space=None, **kwargs):
        if continuous_space:
            return ContinuousPPOAgent(state_dim=state_dim, action_dim=action_dim, **kwargs)
        else:
            # The action_dim for discrete is the number of possible actions
            return DiscretePPOAgent(state_dim=state_dim, action_dim=action_dim, **kwargs)
