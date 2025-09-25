"""Simple example of using PPO agent."""

import numpy as np
import torch
import gymnasium as gym
from ppo import PPOAgent


def simple_example():
    """Simple example showing how to use the PPO agent."""
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Get environment specifications
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        buffer_size=1024,
        batch_size=32,
        epochs=5
    )
    
    # Training loop for a few episodes
    num_episodes = 10
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        
        while True:
            # Get action from agent
            action, log_prob, value = agent.get_action(state)
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store experience
            agent.store_experience(state, action, reward, value, log_prob, done)
            
            episode_reward += reward
            step += 1
            state = next_state
            
            if done:
                break
        
        # Update agent if buffer has enough experiences
        if len(agent.memory) >= agent.batch_size:
            agent.update()
        
        print(f"Episode {episode + 1}: Reward = {episode_reward}, Steps = {step}")
    
    # Save the trained model
    agent.save("example_model.pth")
    print("Model saved as example_model.pth")
    
    # Demonstrate loading and using the saved model
    print("\nTesting saved model...")
    
    # Create new agent and load saved model
    test_agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    test_agent.load("example_model.pth")
    
    # Test for one episode
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        action, _, _ = test_agent.get_action(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"Test episode reward: {total_reward}")
    
    env.close()


if __name__ == "__main__":
    simple_example()