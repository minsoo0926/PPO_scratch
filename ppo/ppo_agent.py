"""
PPO Agent, inspired by Stable Baselines 3.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import time

from .networks import Policy
from .memory import RolloutBuffer

class PPO:
    """
    Proximal Policy Optimization (PPO) agent.

    :param env: The environment to learn from.
    :param learning_rate: The learning rate for the optimizer.
    :param n_steps: The number of steps to run for each environment per update.
    :param batch_size: The size of the batches for the update.
    :param n_epochs: The number of epochs to train on the collected data.
    :param gamma: The discount factor.
    :param gae_lambda: Factor for trade-off of bias vs variance for GAE.
    :param clip_range: Clipping parameter, it can be a function.
    :param vf_coef: Value function coefficient for the loss calculation.
    :param ent_coef: Entropy coefficient for the loss calculation.
    :param max_grad_norm: The maximum value for the gradient clipping.
    :param device: The device to use for training.
    """
    def __init__(
        self,
        env: gym.vector.VectorEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.0,
        max_grad_norm: float = 0.5,
        device: str = "auto",
    ):
        self.env = env
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.num_timesteps = 0
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._setup_model()

    def _setup_model(self):
        self.policy = Policy(
            self.env.single_observation_space,
            self.env.single_action_space
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.env.single_observation_space,
            self.env.single_action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.env.num_envs,
        )

    def train(self) -> None:
        """Train the agent on the collected rollout data."""
        self.policy.train()
        rollout_data = self.rollout_buffer.get(self.batch_size)
        
        # Normalize advantages
        advantages = rollout_data.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.n_epochs):
            # Create a new permutation for each epoch
            indices = np.random.permutation(len(rollout_data.observations))
            for start_idx in range(0, len(rollout_data.observations), self.batch_size):
                end_idx = start_idx + self.batch_size
                minibatch_indices = indices[start_idx:end_idx]

                # Get minibatch data
                obs_batch = rollout_data.observations[minibatch_indices]
                actions_batch = rollout_data.actions[minibatch_indices]
                old_values_batch = rollout_data.old_values[minibatch_indices]
                old_log_probs_batch = rollout_data.old_log_prob[minibatch_indices]
                advantages_batch = advantages[minibatch_indices]
                returns_batch = rollout_data.returns[minibatch_indices]

                values, log_prob, entropy = self.policy.evaluate_actions(obs_batch, actions_batch)
                values = values.flatten()

                # Policy loss
                ratio = torch.exp(log_prob - old_log_probs_batch)
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(returns_batch, values)

                # Entropy loss
                entropy_loss = -torch.mean(entropy)

                # Total loss
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def learn(self, total_timesteps: int):
        """Train the agent for a total of `total_timesteps`."""
        self._last_obs, _ = self.env.reset()
        self._last_obs = torch.tensor(self._last_obs, dtype=torch.float32).to(self.device)
        self._last_done = torch.zeros(self.env.num_envs, device=self.device)
        start_time = time.time()

        while self.num_timesteps < total_timesteps:
            self.collect_rollouts()
            self.train()
            print(f"Timesteps: {self.num_timesteps}/{total_timesteps}, Time: {time.time() - start_time:.2f}s")

        return self

    def collect_rollouts(self) -> None:
        """Collect experiences and store them in the rollout buffer."""
        self.policy.eval()
        self.rollout_buffer.clear()

        for _ in range(self.n_steps):
            with torch.no_grad():
                action, value, log_prob = self.policy(self._last_obs)
            
            next_obs, reward, terminated, truncated, infos = self.env.step(action.cpu().numpy())
            done = terminated | truncated

            self.num_timesteps += self.env.num_envs

            # Convert to torch tensors
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            done_tensor = torch.tensor(done, dtype=torch.float32).to(self.device)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32).to(self.device)

            self.rollout_buffer.add(self._last_obs, action, reward, self._last_done, value, log_prob)

            self._last_obs = next_obs_tensor
            self._last_done = done_tensor

        with torch.no_grad():
            # Compute value for the last observation
            last_values, _, _ = self.policy(self._last_obs)
            last_values = last_values.flatten()

        self.rollout_buffer.compute_returns_and_advantage(last_values, self._last_done)

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
