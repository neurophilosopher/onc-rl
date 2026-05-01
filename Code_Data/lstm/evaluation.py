"""Deterministic evaluation rollouts for ARS and PPO policies."""
import numpy as np
import torch


def _rollout_ars_once(policy, env):
    obs, _ = env.reset()
    hidden = policy.reset_hidden(1)
    ep_return, done = 0.0, False
    while not done:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action, hidden = policy(obs_t, hidden)
        obs, r, term, trunc, _ = env.step(action.squeeze(0).numpy())
        ep_return += r; done = term or trunc
    return ep_return


def evaluate_ars_deterministic(policy, env, n_episodes=50):
    return [_rollout_ars_once(policy, env) for _ in range(n_episodes)]


def evaluate_ppo_deterministic(policy, env, obs_normalizer, n_episodes=50):
    """Evaluate PPO policy using the action mean (no Gaussian sampling)."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        hidden = policy.reset_hidden(1)
        ep_return, done = 0.0, False
        while not done:
            with torch.no_grad():
                obs_norm = obs_normalizer.normalize(obs)
                obs_t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
                action_mean, _, hidden = policy(obs_t, hidden)
                action = np.clip(action_mean.squeeze(0).numpy(),
                                 env.action_space.low, env.action_space.high)
            obs, r, term, trunc, _ = env.step(action)
            ep_return += r; done = term or trunc
        returns.append(ep_return)
    return returns