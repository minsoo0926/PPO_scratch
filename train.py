"""Training script for PPO agent with Gymnasium environments."""

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from ppo import PPOAgent

ENV_CONFIG = {
    "id": "BipedalWalker-v3", # "LunarLander-v3",
    # "continuous": True,
}


def train_ppo(env_config=ENV_CONFIG, total_timesteps=100000, save_freq=10000):
    """
    Train PPO agent on given environment.
    
    Args:
        env_name (str): Name of the Gymnasium environment
        total_timesteps (int): Total number of training timesteps
        save_freq (int): Frequency of saving the model (in timesteps)
    """
    # Create environment
    env = gym.make(**env_config, render_mode=None)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
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
        device=device,
        continuous_space=env.action_space if isinstance(env.action_space, gym.spaces.Box) else None
    )
    
    # Training variables
    timestep = 0
    episode = 0
    episode_rewards = []
    episode_lengths = []
    
    print(f"Starting training on {env_config['id']}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print("-" * 50)
    
    while timestep < total_timesteps:
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and timestep < total_timesteps:
            # Get action from agent
            action, log_prob, value = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(state, action, reward, value, log_prob, done)
            
            # Update counters
            timestep += 1
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            
            # Update agent when buffer is full
            if agent.memory.is_full():
                agent.update(next_state if not done else None)
        
        # Store episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode += 1
        
        # Print progress
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode}, Timestep {timestep}")
            print(f"Average Reward (last 100): {avg_reward:.2f}")
            print(f"Average Length (last 100): {avg_length:.2f}")
            print("-" * 30)
        
        # Save model periodically
        if timestep % save_freq == 0:
            agent.save(f"ppo_model_{timestep}.pth")
            print(f"Model saved at timestep {timestep}")
    
    # Final update if there are remaining experiences
    if len(agent.memory) > 0:
        agent.update()
    
    # Save final model
    agent.save("ppo_model_final.pth")
    
    # Close environment
    env.close()
    
    return agent, episode_rewards, episode_lengths


def plot_training_results(episode_rewards, episode_lengths, window=100):
    """Plot training results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot episode rewards
    ax1.plot(episode_rewards, alpha=0.3, color='blue')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, color='red', linewidth=2)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Progress - Rewards')
    ax1.grid(True)
    
    # Plot episode lengths
    ax2.plot(episode_lengths, alpha=0.3, color='green')
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), moving_avg, color='red', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Training Progress - Episode Length')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_agent(env_config=ENV_CONFIG, model_path='ppo_model_final.pth', num_episodes=10):
    """Test trained agent."""
    # Create environment
    env = gym.make(**env_config, render_mode='human')

    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize agent
    agent = PPOAgent(
        state_dim=state_dim, 
        action_dim=action_dim, 
        device=device, 
        continuous_space=env.action_space if isinstance(env.action_space, gym.spaces.Box) else None
    )

    # Load trained model
    agent.load(model_path)

    print(f"Testing agent on {env_config['id']} for {num_episodes} episodes")

    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Get deterministic action
            action, _, _ = agent.get_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
    
    print(f"\nAverage test reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or test PPO agent.")
    parser.add_argument('--mode', choices=['train', 'test'], default='train', help='Mode: train or test the agent')
    parser.add_argument('--model_path', type=str, default='ppo_model_final.pth', help='Path to the trained model for testing')
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of episodes to test the agent')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    args = parser.parse_args()

    if args.mode == 'train':
        # Train the agent
        agent, rewards, lengths = train_ppo(
            env_config=ENV_CONFIG,
            total_timesteps=args.timesteps,
            save_freq=10000
        )
        
        # Plot results
        plot_training_results(rewards, lengths)

    # Test the trained agent
    test_agent(model_path=args.model_path, num_episodes=args.test_episodes)