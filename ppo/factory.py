"""Factory functions and utilities for creating PPO agents."""

import gymnasium as gym
from gymnasium import vector
from .ppo_agent import DiscretePPOAgent, ContinuousPPOAgent


def create_ppo_agent(env, **kwargs):
    """
    Create appropriate PPO agent based on environment action space.
    
    Args:
        env: Gymnasium environment (single or vectorized)
        **kwargs: Additional arguments for PPO agent
        
    Returns:
        PPO agent (either DiscretePPOAgent or ContinuousPPOAgent)
    """
    # Handle both single and vectorized environments
    obs_space = env.single_observation_space
    action_space = env.single_action_space
    n_envs = env.num_envs if isinstance(env, vector.VectorEnv) else 1

    # Ensure obs_space.shape is available
    if hasattr(obs_space, 'shape') and obs_space.shape is not None:
        state_dim = obs_space.shape[0]
    else:
        raise ValueError(f"Cannot determine state dimension from observation space: {obs_space}")
    
    # Filter out environment-specific kwargs that shouldn't be passed to PPO agent
    ppo_kwargs = {k: v for k, v in kwargs.items() 
                  if k not in ['id', 'n_envs']}  # Filter out env-specific keys
    
    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
        return DiscretePPOAgent(state_dim=state_dim, action_dim=action_dim, n_envs=n_envs, **ppo_kwargs)
    
    elif isinstance(action_space, gym.spaces.Box):
        return ContinuousPPOAgent(
            state_dim=state_dim, 
            action_dim=action_space.shape[0],
            action_low=action_space.low,
            action_high=action_space.high,
            n_envs=n_envs,
            **ppo_kwargs
        )
    
    else:
        raise ValueError(f"Unsupported action space: {type(action_space)}")


def get_action_space_info(env):
    """
    Get information about environment action space.
    
    Args:
        env: Gymnasium environment (single or vectorized)
        
    Returns:
        dict: Action space information
    """
    # Handle both single and vectorized environments
    if isinstance(env, vector.VectorEnv):
        obs_space = env.single_observation_space
        action_space = env.single_action_space
        env_id = 'VectorEnv'  # Simplified for now
    else:
        obs_space = env.observation_space
        action_space = env.action_space
        env_id = getattr(env.spec, 'id', 'Unknown') if env.spec else 'Unknown'
    
    if hasattr(obs_space, 'shape') and obs_space.shape is not None:
        state_dim = obs_space.shape[0]
    else:
        raise ValueError(f"Cannot determine state dimension from observation space: {obs_space}")
    
    if isinstance(action_space, gym.spaces.Discrete):
        return {
            'type': 'discrete',
            'state_dim': state_dim,
            'action_dim': action_space.n,
            'action_space': action_space,
            'env_id': env_id
        }
    
    elif isinstance(action_space, gym.spaces.Box):
        return {
            'type': 'continuous',
            'state_dim': state_dim,
            'action_dim': action_space.shape[0],
            'action_low': action_space.low,
            'action_high': action_space.high,
            'action_space': action_space,
            'env_id': env_id
        }
    
    else:
        raise ValueError(f"Unsupported action space: {type(action_space)}")


def print_action_space_info(env):
    """Print detailed information about environment action space."""
    info = get_action_space_info(env)
    
    print(f"Environment: {info['env_id']}")
    print(f"State dimension: {info['state_dim']}")
    print(f"Action space type: {info['type']}")
    
    if isinstance(env, vector.VectorEnv):
        print(f"Vectorized environment with {env.num_envs} parallel environments")
    
    if info['type'] == 'discrete':
        print(f"Number of actions: {info['action_dim']}")
    else:
        print(f"Action dimension: {info['action_dim']}")
        print(f"Action bounds: [{info['action_low']}, {info['action_high']}]")
    
    print("-" * 50)