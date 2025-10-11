"""Training script for PPO agent with Gymnasium environments."""

import os
import numpy as np
import torch
import gymnasium as gym
from gymnasium import vector
from gymnasium.wrappers.numpy_to_torch import NumpyToTorch as SingleNumpyToTorch
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch as VectorNumpyToTorch
import matplotlib.pyplot as plt
from ppo import create_ppo_agent, print_action_space_info
from config import ENV_CONFIG


def create_env_model_dir(env_id):
    """
    Create directory for environment-specific models.

    Args:
        env_id (str): Environment ID

    Returns:
        str: Path to the environment directory
    """
    env_dir = os.path.join("models", env_id.replace("-", "_"))
    os.makedirs(env_dir, exist_ok=True)
    return env_dir


def get_model_path(env_id, filename):
    """
    Get full path for model file in environment directory.

    Args:
        env_id (str): Environment ID
        filename (str): Model filename

    Returns:
        str: Full path to the model file
    """
    env_dir = create_env_model_dir(env_id)
    return os.path.join(env_dir, filename)


def list_saved_models(env_id):
    """
    List all saved models for a given environment.

    Args:
        env_id (str): Environment ID

    Returns:
        list: List of available model files
    """
    env_dir = os.path.join("models", env_id.replace("-", "_"))
    if not os.path.exists(env_dir):
        return []

    model_files = [f for f in os.listdir(env_dir) if f.endswith('.pth')]
    return sorted(model_files)


def find_latest_model(env_id):
    """
    Find the latest saved model for a given environment.

    Args:
        env_id (str): Environment ID

    Returns:
        str or None: Path to the latest model file, or None if no models found
    """
    model_files = list_saved_models(env_id)
    if not model_files:
        return None

    # Try to find final model first
    final_model = "ppo_model_final.pth"
    if final_model in model_files:
        return get_model_path(env_id, final_model)

    # Otherwise, find the model with highest timestep
    timestep_models = []
    for model_file in model_files:
        if "ppo_model_" in model_file and model_file.endswith(".pth"):
            try:
                # Extract timestep from filename
                timestep_str = model_file.replace("ppo_model_", "").replace(".pth", "")
                if timestep_str.isdigit():
                    timestep_models.append((int(timestep_str), model_file))
            except:
                continue

    if timestep_models:
        # Return model with highest timestep
        latest_model = max(timestep_models, key=lambda x: x[0])[1]
        return get_model_path(env_id, latest_model)

    # Fallback to any available model
    return get_model_path(env_id, model_files[0])


def train_ppo(env_config=ENV_CONFIG, total_timesteps=100000, save_freq=10000, resume_from=None):
    """
    Train PPO agent on given environment.

    Args:
        env_config (dict): Environment configuration
        total_timesteps (int): Total number of training timesteps
        save_freq (int): Frequency of saving the model (in timesteps)
        resume_from (str, optional): Model filename to resume training from
    """
    # Create environment - support both single and vectorized environments
    n_envs = env_config.get('n_envs', 1)

    print(f"Creating vectorized environment with {n_envs} parallel environments")
    # Create multiple environments for vectorization
    env = gym.make_vec(env_config['id'], render_mode=None, num_envs=n_envs)
    env = VectorNumpyToTorch(env)  # Convert observations to PyTorch tensors

    # Print action space information
    print_action_space_info(env)

    # Set device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_str}")

    # Initialize PPO agent using factory function
    agent = create_ppo_agent(
        env,
        lr=env_config.get('lr', 3e-4),
        gamma=env_config.get('gamma', 0.99),
        lam=env_config.get('lam', 0.95),
        clip_ratio=env_config.get('clip_ratio', 0.2),
        value_coef=env_config.get('value_coef', 0.5),
        entropy_coef=env_config.get('entropy_coef', 0.01),
        max_grad_norm=env_config.get('max_grad_norm', 0.5),
        hidden_dim=env_config.get('hidden_dim', 64),
        buffer_size=env_config.get('buffer_size', 2048),
        batch_size=env_config.get('batch_size', 64),
        epochs=env_config.get('epochs', 10),
        device=device_str
    )

    # Resume from existing model if specified
    start_timestep = 0
    checkpoint = None
    
    if resume_from:
        if not os.path.exists(resume_from):
            # Try to find it in the environment directory
            env_model_path = get_model_path(env_config['id'], resume_from)
            if os.path.exists(env_model_path):
                resume_from = env_model_path
            else:
                print(f"Resume model not found: {resume_from}")
                print(f"Available models: {list_saved_models(env_config['id'])}")
                return None, [], []

        print(f"Resuming training from: {resume_from}")
        
        # Load checkpoint to get episode statistics
        checkpoint = torch.load(resume_from, map_location='cpu', weights_only=False)
        agent.load(resume_from)

        # Extract timestep from filename if possible
        try:
            filename = os.path.basename(resume_from)
            if "ppo_model_" in filename and filename.endswith(".pth"):
                timestep_str = filename.replace("ppo_model_", "").replace(".pth", "")
                if timestep_str.isdigit():
                    start_timestep = int(timestep_str)
                    print(f"Resuming from timestep: {start_timestep}")
        except:
            pass

    # Training variables
    timestep = start_timestep
    prior_logging = 0
    prior_save = 0
    
    # Initialize episode statistics
    if checkpoint is not None:
        episode_rewards = checkpoint.get('episode_rewards', [])
        episode_lengths = checkpoint.get('episode_lengths', [])
        episode = len(episode_rewards)
        print(f"Resumed with {len(episode_rewards)} previous episodes")
        if episode_rewards:
            recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"Recent average reward: {recent_avg:.2f}")
    else:
        episode_rewards = []
        episode_lengths = []
        episode = 0

    print(f"Starting training on {env_config['id']}")
    print("-" * 50)

    # Vectorized environment training with batch processing
    while timestep < total_timesteps:
        states, _ = env.reset()  # Shape: (n_envs, obs_dim)
        episode_rewards_batch = np.zeros(n_envs)
        episode_lengths_batch = np.zeros(n_envs)
        dones = np.array([False] * n_envs)

        while not np.all(dones) and timestep < total_timesteps:
            # Get actions for all environments at once (batch processing)
            actions, log_probs, values = agent.get_action(states)

            # Take step in all environments
            next_states, rewards, terminateds, truncateds, _ = env.step(actions)
            new_dones = terminateds | truncateds

            # Store experiences for all environments
            for i in range(n_envs):
                if not dones[i]:
                    # Convert tensor values to appropriate types for storage
                    state = states[i]
                    action_val = actions[i]
                    reward_val = float(rewards[i])
                    value_val = values[i]
                    # Keep log_prob as tensor for memory compatibility
                    log_prob_val = log_probs[i]
                    done_val = bool(new_dones[i])

                    agent.store_experience(
                        i, state, action_val, reward_val,
                        value_val, log_prob_val, done_val
                    )
                    episode_rewards_batch[i] += float(rewards[i])
                    episode_lengths_batch[i] += 1

            # Update counters
            timestep += np.sum(~dones)  # Count active environments
            new_dones = new_dones.cpu().numpy()

            # Update done status and collect episode stats
            newly_done = new_dones & ~dones
            if np.any(newly_done):
                for i in range(n_envs):
                    if newly_done[i]:
                        episode_rewards.append(episode_rewards_batch[i])
                        episode_lengths.append(episode_lengths_batch[i])
                        episode += 1
                        episode_rewards_batch[i] = 0.0
                        episode_lengths_batch[i] = 0.0

            dones = new_dones
            states = next_states

            # Update agent when buffer is full
            if agent.memory.is_full():
                agent.update(next_states)

            # Print progress
            if episode // 100 != prior_logging // 100:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_length = np.mean(episode_lengths[-100:])
                training_stats = agent.training_stats
                print(f"Episode {episode}, Timestep {timestep}")
                print(f"Average Reward (last 100): {avg_reward:.2f}")
                print(f"Average Length (last 100): {avg_length:.2f}")
                if training_stats:
                    print("Training Stats:")
                    for key, value in training_stats.items():
                        if value:
                            print(f"  {key}: {value[-1]:.4f}")
                print("-" * 30)
                prior_logging = episode

            # Save model periodically
            if timestep // save_freq != prior_save // save_freq:
                model_path = get_model_path(env_config['id'], f"ppo_model_{timestep // save_freq * save_freq}.pth")
                agent.save(model_path, episode_rewards, episode_lengths)
                print(f"Model saved at timestep {timestep}: {model_path}")
                prior_save = timestep

    # Final update if there are remaining experiences
    if len(agent.memory) > 0:
        agent.update()

    # Save final model
    final_model_path = get_model_path(env_config['id'], "ppo_model_final.pth")
    agent.save(final_model_path, episode_rewards, episode_lengths)
    print(f"Final model saved: {final_model_path}")

    # Close environment
    env.close()

    return agent, episode_rewards, episode_lengths


def plot_training_results(episode_rewards, episode_lengths, agent=None, window=100):
    """Plot training results including all training statistics."""
    # Determine the number of subplots needed
    num_stats = 0
    training_stats = {}
    
    if agent and hasattr(agent, 'training_stats') and agent.training_stats:
        training_stats = agent.training_stats
        num_stats = len(training_stats)
    
    # Calculate total number of plots (rewards + lengths + training stats)
    total_plots = 2 + num_stats
    
    # Determine grid layout
    if total_plots <= 2:
        rows, cols = 1, 2
    elif total_plots <= 4:
        rows, cols = 2, 2
    elif total_plots <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, (total_plots + 2) // 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    
    # Flatten axes for easier indexing if multiple subplots
    if total_plots > 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot episode rewards
    axes[plot_idx].plot(episode_rewards, alpha=0.3, color='blue')
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[plot_idx].plot(range(window-1, len(episode_rewards)), moving_avg, color='red', linewidth=2)
    axes[plot_idx].set_xlabel('Episode')
    axes[plot_idx].set_ylabel('Episode Reward')
    axes[plot_idx].set_title('Training Progress - Rewards')
    axes[plot_idx].grid(True)
    plot_idx += 1

    # Plot episode lengths
    axes[plot_idx].plot(episode_lengths, alpha=0.3, color='green')
    if len(episode_lengths) >= window:
        moving_avg = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        axes[plot_idx].plot(range(window-1, len(episode_lengths)), moving_avg, color='red', linewidth=2)
    axes[plot_idx].set_xlabel('Episode')
    axes[plot_idx].set_ylabel('Episode Length')
    axes[plot_idx].set_title('Training Progress - Episode Length')
    axes[plot_idx].grid(True)
    plot_idx += 1
    
    # Plot training statistics
    colors = ['purple', 'orange', 'brown', 'pink', 'gray', 'olive']
    for i, (stat_name, stat_values) in enumerate(training_stats.items()):
        if plot_idx < len(axes):
            color = colors[i % len(colors)]
            axes[plot_idx].plot(stat_values, color=color, linewidth=2)
            axes[plot_idx].set_xlabel('Training Update')
            axes[plot_idx].set_ylabel(stat_name.replace('_', ' ').title())
            axes[plot_idx].set_title(f'Training Stats - {stat_name.replace("_", " ").title()}')
            axes[plot_idx].grid(True)
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def test_agent(env_config=ENV_CONFIG, model_path=None, num_episodes=10):
    """Test trained agent."""
    # Create single environment for testing (no vectorization for testing)
    # Always use single environment for testing regardless of training setup
    test_env_config = env_config.copy()
    test_env_config['n_envs'] = 1  # Force single environment for testing

    env: gym.Env = gym.make(test_env_config['id'], render_mode="human")
    env = SingleNumpyToTorch(env)  # Convert observations to PyTorch tensors

    # Set device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine model path
    if model_path is None:
        # Try to find latest model for this environment
        model_path = find_latest_model(env_config['id'])
        if model_path is None:
            print(f"No saved models found for environment {env_config['id']}")
            print(f"Available models: {list_saved_models(env_config['id'])}")
            env.close()
            return
        print(f"Using latest model: {model_path}")
    else:
        # If a specific path is provided, check if it exists
        if not os.path.exists(model_path):
            # Try to find it in the environment directory
            env_model_path = get_model_path(env_config['id'], os.path.basename(model_path))
            if os.path.exists(env_model_path):
                model_path = env_model_path
                print(f"Using model from environment directory: {model_path}")
            else:
                print(f"Model not found: {model_path}")
                print(f"Available models: {list_saved_models(env_config['id'])}")
                env.close()
                return

    # Create agent using same method as training, then load weights
    agent = create_ppo_agent(
        env,
        lr=test_env_config.get('lr', 3e-4),
        gamma=test_env_config.get('gamma', 0.99),
        lam=test_env_config.get('lam', 0.95),
        clip_ratio=test_env_config.get('clip_ratio', 0.2),
        value_coef=test_env_config.get('value_coef', 0.5),
        entropy_coef=test_env_config.get('entropy_coef', 0.01),
        max_grad_norm=test_env_config.get('max_grad_norm', 0.5),
        hidden_dim=test_env_config.get('hidden_dim', 64),
        buffer_size=test_env_config.get('buffer_size', 2048),
        batch_size=test_env_config.get('batch_size', 64),
        epochs=test_env_config.get('epochs', 10),
        device=device_str
    )

    # Load trained weights
    agent.load_network_only(model_path)

    # Set model to evaluation mode
    agent.network.eval()

    print(f"Testing agent on {env_config['id']} for {num_episodes} episodes")

    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # Get deterministic action (no gradient computation needed)
            with torch.no_grad():
                action, _, _ = agent.get_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    print(f"\nAverage test reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    env.close()


def show_available_models(env_config=ENV_CONFIG):
    """Show all available models for the given environment."""
    env_id = env_config['id']
    models = list_saved_models(env_id)

    if not models:
        print(f"No saved models found for environment '{env_id}'")
        return

    print(f"Available models for environment '{env_id}':")
    print("-" * 50)

    env_dir = os.path.join("models", env_id.replace("-", "_"))
    for model in models:
        model_path = os.path.join(env_dir, model)
        file_size = os.path.getsize(model_path)
        file_size_mb = file_size / (1024 * 1024)

        print(f"  {model:<25} ({file_size_mb:.2f} MB)")

    latest_model = find_latest_model(env_id)
    if latest_model:
        print(f"\nLatest model: {os.path.basename(latest_model)}")


def clean_old_models(env_config=ENV_CONFIG, keep_count=5):
    """
    Clean old model files, keeping only the most recent ones.

    Args:
        env_config (dict): Environment configuration
        keep_count (int): Number of models to keep
    """
    env_id = env_config['id']
    models = list_saved_models(env_id)

    if len(models) <= keep_count:
        print(f"Only {len(models)} models found, no cleanup needed")
        return

    # Separate final model and timestep models
    final_models = [m for m in models if "final" in m]
    timestep_models = []

    for model in models:
        if "ppo_model_" in model and model.endswith(".pth") and "final" not in model:
            try:
                timestep_str = model.replace("ppo_model_", "").replace(".pth", "")
                if timestep_str.isdigit():
                    timestep_models.append((int(timestep_str), model))
            except:
                continue

    # Sort by timestep and keep most recent
    timestep_models.sort(key=lambda x: x[0], reverse=True)
    models_to_keep = [m[1] for m in timestep_models[:keep_count-len(final_models)]]
    models_to_keep.extend(final_models)

    # Delete old models
    env_dir = os.path.join("models", env_id.replace("-", "_"))
    deleted_count = 0

    for model in models:
        if model not in models_to_keep:
            model_path = os.path.join(env_dir, model)
            try:
                os.remove(model_path)
                print(f"Deleted: {model}")
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {model}: {e}")

    print(f"Cleaned up {deleted_count} old models, kept {len(models_to_keep)} models")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train or test PPO agent with environment-specific model management.")
    parser.add_argument('--mode', choices=['train', 'test', 'list', 'clean'], default='train',
                       help='Mode: train, test, list models, or clean old models')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to the trained model for testing (auto-detects latest if not specified)')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='Model filename to resume training from')
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of episodes to test the agent')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--save_freq', type=int, default=10000, help='Model save frequency (in timesteps)')
    parser.add_argument('--keep_models', type=int, default=5, help='Number of models to keep when cleaning')
    args = parser.parse_args()

    if args.mode == 'train':
        # Train the agent
        result = train_ppo(
            env_config=ENV_CONFIG,
            total_timesteps=args.timesteps,
            save_freq=args.save_freq,
            resume_from=args.resume_from
        )

        if result is not None:
            agent, rewards, lengths = result
            # Plot results
            plot_training_results(rewards, lengths)
        else:
            print("Training failed - check error messages above")

    elif args.mode == 'test':
        # Test the trained agent
        test_agent(env_config=ENV_CONFIG, model_path=args.model_path, num_episodes=args.test_episodes)

    elif args.mode == 'list':
        # List available models
        show_available_models(ENV_CONFIG)

    elif args.mode == 'clean':
        # Clean old models
        clean_old_models(ENV_CONFIG, keep_count=args.keep_models)