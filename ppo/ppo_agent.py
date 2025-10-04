"""PPO Agent implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .networks import ActorCritic
from .memory import PPOMemory


class PPOAgent:
    """Proximal Policy Optimization Agent."""
    
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
        device='cpu',
        continuous_space=None
    ):
        """
        Initialize PPO agent.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            lr (float): Learning rate
            gamma (float): Discount factor
            lam (float): GAE lambda parameter
            clip_ratio (float): PPO clipping parameter
            value_coef (float): Value function loss coefficient
            entropy_coef (float): Entropy regularization coefficient
            max_grad_norm (float): Maximum gradient norm for clipping
            hidden_dim (int): Hidden layer dimension
            buffer_size (int): Size of experience buffer
            batch_size (int): Mini-batch size for training
            epochs (int): Number of training epochs per update
            device (str): Device to run on
            continuous_space (Gym.spaces.Box): Continuous action space if applicable
        """
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
        self.continuous_space = continuous_space
        
        # Initialize network and optimizer
        self.network = ActorCritic(state_dim, action_dim, hidden_dim, continuous_space).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Initialize memory buffer
        if continuous_space:
            continuous_dim = action_dim
        else:
            continuous_dim = None
        self.memory = PPOMemory(buffer_size, state_dim, device, continuous_dim)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': []
        }
    
    def get_action(self, state, deterministic=False):
        """
        Select action for given state.
        
        Args:
            state (np.ndarray): Current state
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            tuple: (action, log_prob, value)
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        
        with torch.no_grad():
            if deterministic:
                action_logits, value, _ = self.network(state.unsqueeze(0))
                if self.continuous_space:
                    action = action_logits
                    log_prob = torch.zeros(action_logits.shape, device=self.device)
                else:
                    action = torch.argmax(action_logits, dim=-1)
                    log_prob = torch.zeros(1, device=self.device)
            else:
                action, log_prob, _, value = self.network.get_action_and_value(state.unsqueeze(0))
        
        if self.continuous_space:
            action = action.cpu().numpy().flatten()
            log_prob = log_prob.cpu().numpy().flatten()
        else:
            action = action.item()
            log_prob = log_prob.item()
        return action, log_prob, value.item()

    def store_experience(self, state, action, reward, value, log_prob, done):
        """Store experience in memory buffer."""
        self.memory.store(state, action, reward, value, log_prob, done)
    
    def compute_advantages(self, rewards, values, dones, next_value=0):
        """
        Compute GAE (Generalized Advantage Estimation) advantages.
        
        Args:
            rewards (torch.Tensor): Rewards
            values (torch.Tensor): Value estimates
            dones (torch.Tensor): Done flags
            next_value (float): Next state value for last step
            
        Returns:
            tuple: (advantages, returns)
        """
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
    
    def update(self, next_state=None):
        """
        Update the policy using PPO algorithm.
        
        Args:
            next_state (np.ndarray, optional): Next state for advantage computation
        """
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
        
        # Convert to tensors and handle continuous actions
        old_log_probs = old_log_probs.detach()
        if self.continuous_space is not None and len(old_log_probs.shape) > 1:
            old_log_probs = old_log_probs.sum(dim=-1, keepdim=True)  # Sum over action dimensions
        
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
                
                # Handle continuous actions - sum log probs over action dimensions
                if self.continuous_space is not None and len(new_log_probs.shape) > 1:
                    new_log_probs = new_log_probs.sum(dim=-1, keepdim=True)
                    entropy = entropy.sum(dim=-1, keepdim=True)
                
                # Compute policy loss
                ratio = torch.exp(new_log_probs.squeeze() - batch_old_log_probs.squeeze())
                    
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