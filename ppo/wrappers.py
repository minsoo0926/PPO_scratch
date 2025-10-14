"""
Environment wrappers, inspired by Stable Baselines 3.
"""
import torch
import gymnasium as gym
import numpy as np
import pickle

# Copied from ppo/normalizer.py and adapted to be a standalone class
class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.power(delta, 2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class VecNormalize(gym.vector.VectorWrapper):
    """
    A vectorized wrapper that normalizes the observations and returns.
    """
    def __init__(self, venv, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10., gamma=0.99, epsilon=1e-8):
        super().__init__(venv)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape) if norm_obs else None
        self.ret_rms = RunningMeanStd(shape=()) if norm_reward else None
        self.returns = np.zeros(self.num_envs)

    def step(self, action):
        obs, rews, terminated, truncated, infos = self.env.step(action)
        self.returns = self.returns * self.gamma + rews
        self.returns[terminated | truncated] = 0

        if self.norm_obs:
            self.obs_rms.update(obs)
            obs = self._normalize_obs(obs)

        if self.norm_reward:
            self.ret_rms.update(self.returns)
            rews = self._normalize_reward(rews)

        return obs, rews, terminated, truncated, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.returns = np.zeros(self.num_envs)
        if self.norm_obs:
            self.obs_rms.update(obs)
            obs = self._normalize_obs(obs)
        return obs, info

    def _normalize_obs(self, obs):
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)

    def _normalize_reward(self, rews):
        return np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({'obs_rms': self.obs_rms, 'ret_rms': self.ret_rms}, f)

    @classmethod
    def load(cls, path, env):
        with open(path, "rb") as f:
            data = pickle.load(f)
        vec_normalize = cls(env)
        vec_normalize.obs_rms = data['obs_rms']
        vec_normalize.ret_rms = data['ret_rms']
        return vec_normalize
