"""Training loops for ARS and PPO with periodic deterministic evaluation."""
import time
import numpy as np
import torch

from envs import make_env
from policies import (
    LSTMPolicyARS, LSTMPolicyARSProjected,
    LSTMPolicyPPO, LSTMPolicyPPOProjected,
)
from ppo_utils import ObsNormalizer, collect_rollout, compute_gae, ppo_update
from evaluation import (
    _rollout_ars_once,
    evaluate_ars_deterministic,
    evaluate_ppo_deterministic,
)


def train_ars_with_eval(env_name, partial, hidden_dim, total_iterations, seed,
                        sigma_init=0.5, alpha=1.2, n_rollouts=5,
                        sigma_min=0.1, sigma_max=1.0,
                        eval_every=50, eval_episodes=50,
                        project_dim=None):
    """Run ARS with periodic deterministic evaluation.
    If project_dim is set, uses LSTMPolicyARSProjected with a learned obs -> project_dim bottleneck."""
    torch.manual_seed(seed); np.random.seed(seed)
    env = make_env(env_name, partial=partial); env.reset(seed=seed)
    eval_env = make_env(env_name, partial=partial); eval_env.reset(seed=seed + 10_000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    if project_dim is not None:
        policy = LSTMPolicyARSProjected(obs_dim, action_dim, hidden_dim, project_dim,
                                         action_low, action_high)
    else:
        policy = LSTMPolicyARS(obs_dim, action_dim, hidden_dim, action_low, action_high)

    theta = policy.flatten_params().clone()
    sigma = sigma_init

    def objective(params):
        policy.unflatten_params(params)
        rets = [_rollout_ars_once(policy, env) for _ in range(n_rollouts)]
        return float(np.mean(rets))

    f_theta = objective(theta)
    stale_count = 0
    best_theta = theta.clone(); best_return = f_theta
    eval_iterations, eval_means = [], []

    t_start = time.time()
    for k in range(1, total_iterations + 1):
        noise = torch.randn_like(theta) * sigma
        theta_prime = theta + noise
        f_prime = objective(theta_prime)
        if f_prime > f_theta:
            theta = theta_prime; f_theta = f_prime; stale_count = 0
            sigma = min(sigma * alpha, sigma_max)
            if f_theta > best_return:
                best_theta = theta.clone(); best_return = f_theta
        else:
            sigma = max(sigma / alpha, sigma_min)

        stale_count += 1
        if stale_count > n_rollouts:
            f_theta = objective(theta); stale_count = 0

        if k % eval_every == 0:
            policy.unflatten_params(best_theta)
            rets = evaluate_ars_deterministic(policy, eval_env, eval_episodes)
            eval_iterations.append(k)
            eval_means.append(float(np.mean(rets)))
            policy.unflatten_params(theta)

    policy.unflatten_params(best_theta)
    final_rets = evaluate_ars_deterministic(policy, eval_env, eval_episodes)
    env.close(); eval_env.close()

    return {"algo": "ARS", "env_name": env_name, "partial": partial, "seed": seed,
            "hidden_dim": hidden_dim,
            "project_dim": project_dim,
            "eval_iterations": eval_iterations, "eval_means": eval_means,
            "final_eval_returns": final_rets,
            "final_eval_mean": float(np.mean(final_rets)),
            "final_eval_std": float(np.std(final_rets)),
            "total_iterations": total_iterations,
            "elapsed_seconds": time.time() - t_start}


def train_ppo_with_eval(env_name, partial, hidden_dim, total_iterations, seed,
                        n_steps=2048, n_epochs=4, clip_ratio=0.1, lr=3e-4,
                        gamma=0.99, gae_lam=0.95, ent_coef=0.01, vf_coef=0.5,
                        max_grad_norm=0.5, target_kl=0.015,
                        eval_every=50, eval_episodes=50,
                        project_dim=None):
    """Run PPO with linear learning rate decay and periodic deterministic evaluation.
    If project_dim is set, uses LSTMPolicyPPOProjected with learned obs -> project_dim bottlenecks."""
    torch.manual_seed(seed); np.random.seed(seed)
    env = make_env(env_name, partial=partial); env.reset(seed=seed)
    eval_env = make_env(env_name, partial=partial); eval_env.reset(seed=seed + 10_000)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if project_dim is not None:
        policy = LSTMPolicyPPOProjected(obs_dim, action_dim, hidden_dim, project_dim)
    else:
        policy = LSTMPolicyPPO(obs_dim, action_dim, hidden_dim)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    obs_normalizer = ObsNormalizer(obs_dim)

    eval_iterations, eval_means = [], []
    t_start = time.time()
    for k in range(1, total_iterations + 1):
        fraction_remaining = 1.0 - (k - 1) / total_iterations
        current_lr = lr * max(fraction_remaining, 0.0)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        rollout = collect_rollout(policy, env, obs_normalizer, n_steps)
        advantages, returns = compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"],
            rollout["last_value"], gamma, gae_lam)
        ppo_update(policy, optimizer, rollout["obs"], rollout["actions"],
                   rollout["log_probs"], advantages, returns, rollout["dones"],
                   rollout["initial_hidden"], n_epochs=n_epochs,
                   clip_ratio=clip_ratio, ent_coef=ent_coef, vf_coef=vf_coef,
                   max_grad_norm=max_grad_norm, target_kl=target_kl)

        if k % eval_every == 0:
            rets = evaluate_ppo_deterministic(policy, eval_env, obs_normalizer, eval_episodes)
            eval_iterations.append(k)
            eval_means.append(float(np.mean(rets)))

    final_rets = evaluate_ppo_deterministic(policy, eval_env, obs_normalizer, eval_episodes)
    env.close(); eval_env.close()

    return {"algo": "PPO", "env_name": env_name, "partial": partial, "seed": seed,
            "hidden_dim": hidden_dim,
            "project_dim": project_dim,
            "eval_iterations": eval_iterations, "eval_means": eval_means,
            "final_eval_returns": final_rets,
            "final_eval_mean": float(np.mean(final_rets)),
            "final_eval_std": float(np.std(final_rets)),
            "total_iterations": total_iterations,
            "elapsed_seconds": time.time() - t_start}