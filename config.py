import torch

ENV_CONFIG = {
    # "id": "BipedalWalker-v3",
    # "id": "Humanoid-v5", 
    # "id": "MountainCarContinuous-v0",
    # "id": "Pendulum-v1",
    # "id": "LunarLander-v3",
    # "id": "Ant-v5",
    # "id": "HalfCheetah-v5",
    # "id": "HumanoidStandup-v5",
    # "id": "CartPole-v1",
    "id": "Hopper-v5",
    # "id": "ALE/Pong-v5",
    "lr": 3e-4,
    "gamma": 0.99,
    "lam": 0.95,
    "clip_ratio": 0.2,
    "value_coef": 0.5,
    "entropy_coef": 0.01,
    "max_grad_norm": 0.5,
    "buffer_size": 2048,
    "batch_size": 64,
    "epochs": 10,
    "hidden_dim": 64,
    "n_envs": 8,
    "kl_coef": 1.0,
    "target_kl": 0.01,
    "adaptive_kl": True,
}

# torch.autograd.set_detect_anomaly(True)
    