"""
Main training script for the PPO agent, SB3 style.
"""
import gymnasium as gym
import torch
import os
import pickle
import numpy as np

from ppo.ppo_agent import PPO
from ppo.wrappers import VecNormalize

def main():
    # --- Hyperparameters ---
    ENV_ID = "LunarLander-v3"
    TOTAL_TIMESTEPS = 100000
    N_ENVS = 8
    N_STEPS = 2048 // N_ENVS # Rollout buffer size per env

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Environment ---
    env = gym.make_vec(ENV_ID, num_envs=N_ENVS)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # --- Agent ---
    agent = PPO(
        env=env,
        n_steps=N_STEPS,
        n_epochs=10,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        device=device,
    )

    # --- Training ---
    agent.learn(total_timesteps=TOTAL_TIMESTEPS)

    # --- Save ---
    model_path = f"ppo_{ENV_ID}.pth"
    stats_path = f"vec_normalize_{ENV_ID}.pkl"
    agent.save(model_path)
    env.save(stats_path)
    print(f"Saved agent to {model_path} and env stats to {stats_path}")

    # --- Testing ---
    print("--- Testing a trained agent ---")
    
    # Load normalization stats
    with open(stats_path, "rb") as f:
        data = pickle.load(f)
        obs_rms = data['obs_rms']

    # Create a single environment for testing
    test_env = gym.make(ENV_ID, render_mode="human")

    # Use the policy from the trained agent
    test_policy = agent.policy
    test_policy.eval() # Set to evaluation mode

    obs, _ = test_env.reset()
    for _ in range(1000):
        # Manual normalization
        norm_obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10., 10.)
        
        with torch.no_grad():
            obs_tensor = torch.tensor(norm_obs, dtype=torch.float32).to(device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            action, _, _ = test_policy(obs_tensor)
        obs, _, _, _, _ = test_env.step(action.item())
    
    test_env.close()

if __name__ == "__main__":
    main()
