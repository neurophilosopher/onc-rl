"""PPO helpers: observation normalization, rollout collection, GAE, and the update step."""
import numpy as np
import torch
import torch.nn as nn


class ObsNormalizer:
    """Running mean/std normalizer using Welford's online algorithm."""
    def __init__(self, obs_dim, clip=10.0):
        self.mean = np.zeros(obs_dim, dtype=np.float64)
        self.var = np.ones(obs_dim, dtype=np.float64)
        self.count = 1e-4
        self.clip = clip

    def update(self, obs):
        batch_mean = obs.astype(np.float64)
        batch_var = np.zeros_like(self.mean); batch_count = 1
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, obs):
        return np.clip(
            (obs - self.mean.astype(np.float32)) / (np.sqrt(self.var.astype(np.float32)) + 1e-8),
            -self.clip, self.clip,
        )


def collect_rollout(policy, env, obs_normalizer, n_steps=2048):
    """Run n_steps of environment interaction with stochastic action sampling."""
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    all_obs = torch.zeros(n_steps, obs_dim)
    all_actions = torch.zeros(n_steps, action_dim)
    all_log_probs = torch.zeros(n_steps)
    all_values = torch.zeros(n_steps)
    all_rewards = torch.zeros(n_steps)
    all_dones = torch.zeros(n_steps)
    episode_returns = []

    obs, _ = env.reset()
    hidden = policy.reset_hidden(1)
    initial_hidden = (
        (hidden[0][0].detach().clone(), hidden[0][1].detach().clone()),
        (hidden[1][0].detach().clone(), hidden[1][1].detach().clone()),
    )
    ep_return = 0.0

    with torch.no_grad():
        std = torch.exp(policy.actor_log_std)
        for t in range(n_steps):
            obs_normalizer.update(obs)
            obs_norm = obs_normalizer.normalize(obs)
            obs_t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            action_mean, value, hidden = policy(obs_t, hidden)
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            action_clipped = np.clip(action.squeeze(0).numpy(),
                                     env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, _ = env.step(action_clipped)
            done = terminated or truncated
            ep_return += reward

            all_obs[t] = obs_t.squeeze(0)
            all_actions[t] = action.squeeze(0)
            all_log_probs[t] = log_prob.squeeze(0)
            all_values[t] = value.squeeze(0)
            all_rewards[t] = reward
            all_dones[t] = float(done)

            if done:
                episode_returns.append(ep_return); ep_return = 0.0
                next_obs, _ = env.reset()
                hidden = policy.reset_hidden(1)
            obs = next_obs

        obs_normalizer.update(obs)
        obs_norm = obs_normalizer.normalize(obs)
        obs_t = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
        _, last_value, _ = policy(obs_t, hidden)
        last_value = last_value.squeeze(0).item()

    return {"obs": all_obs, "actions": all_actions, "log_probs": all_log_probs,
            "values": all_values, "rewards": all_rewards, "dones": all_dones,
            "initial_hidden": initial_hidden, "last_value": last_value,
            "episode_returns": episode_returns}


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation with terminal masking."""
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(policy, optimizer, obs_seq, actions, old_log_probs,
               advantages, returns, dones, initial_hidden,
               n_epochs=4, clip_ratio=0.1, ent_coef=0.01,
               vf_coef=0.5, max_grad_norm=0.5, target_kl=0.015):
    """Clipped-surrogate PPO update. Hidden state is refreshed across the full sequence each epoch."""
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    for epoch in range(n_epochs):
        new_log_probs, new_values, entropy = policy.evaluate_sequences(
            obs_seq, actions, dones, initial_hidden)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = (new_values - returns).pow(2).mean()
        loss = policy_loss - ent_coef * entropy + vf_coef * value_loss
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean().item()
        if approx_kl > target_kl:
            break