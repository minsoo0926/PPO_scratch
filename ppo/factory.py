"""Factory functions and utilities for creating PPO agents."""

import gymnasium as gym
from .ppo_agent import DiscretePPOAgent, ContinuousPPOAgent


def create_ppo_agent(env, **kwargs):
    """
    Create appropriate PPO agent based on environment action space.
    
    Args:
        env: Gymnasium environment
        **kwargs: Additional arguments for PPO agent
        
    Returns:
        PPO agent (either DiscretePPOAgent or ContinuousPPOAgent)
    """
    state_dim = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
        return DiscretePPOAgent(state_dim=state_dim, action_dim=action_dim, **kwargs)
    
    elif isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high
        return ContinuousPPOAgent(
            state_dim=state_dim, 
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unsupported action space: {type(env.action_space)}")


def get_action_space_info(env):
    """
    Get information about environment action space.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        dict: Action space information
    """
    state_dim = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        return {
            'type': 'discrete',
            'state_dim': state_dim,
            'action_dim': env.action_space.n,
            'action_space': env.action_space
        }
    
    elif isinstance(env.action_space, gym.spaces.Box):
        return {
            'type': 'continuous',
            'state_dim': state_dim,
            'action_dim': env.action_space.shape[0],
            'action_low': env.action_space.low,
            'action_high': env.action_space.high,
            'action_space': env.action_space
        }
    
    else:
        raise ValueError(f"Unsupported action space: {type(env.action_space)}")


def print_action_space_info(env):
    """Print detailed information about environment action space."""
    info = get_action_space_info(env)
    
    print(f"Environment: {env.spec.id}")
    print(f"State dimension: {info['state_dim']}")
    print(f"Action space type: {info['type']}")
    
    if info['type'] == 'discrete':
        print(f"Number of actions: {info['action_dim']}")
    else:
        print(f"Action dimension: {info['action_dim']}")
        print(f"Action bounds: [{info['action_low']}, {info['action_high']}]")
    
    print("-" * 50)