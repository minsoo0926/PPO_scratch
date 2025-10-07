import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import time

# 1️⃣ 환경 생성 (벡터화 환경)
env = make_vec_env("Ant-v5", n_envs=8)

# 2️⃣ PPO 모델 초기화
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
)

# 3️⃣ 학습
model.learn(total_timesteps=100_000)

# 4️⃣ 단일 환경으로 전환 (렌더링용)
env = gym.make("Ant-v5", render_mode="human")

obs, info = env.reset()
for _ in range(2000):  # 2000 step 시뮬레이션
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    time.sleep(0.01)  # 렌더 속도 조절 (optional)
    if done or truncated:
        obs, info = env.reset()

env.close()
