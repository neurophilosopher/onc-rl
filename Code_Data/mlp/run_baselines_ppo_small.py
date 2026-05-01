
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
import argparse

SEEDS = list(range(5))


ENV_CONFIGS = {
    "invpend_full": {
        "env_name": "InvertedPendulum-v5",
        "base_name": "InvertedPendulum",
        "partial_obs": False,
    },
    "invpend_partial": {
        "env_name": "InvertedPendulum-v5",
        "base_name": "InvertedPendulum",
        "partial_obs": True,
    },
    "cheetah_full": {
        "env_name": "HalfCheetah-v5",
        "base_name": "HalfCheetah",
        "partial_obs": False,
    },
    "cheetah_partial": {
        "env_name": "HalfCheetah-v5",
        "base_name": "HalfCheetah",
        "partial_obs": True,
    },
    "swimmer_full": {
        "env_name": "Swimmer-v5",
        "base_name": "Swimmer",
        "partial_obs": False,
    },
}

PARTIAL_KEEP = {
    "HalfCheetah": list(range(0, 8)),
    "InvertedPendulum": [0, 1],
    "Swimmer": [0, 1],
}

BOTTLENECK_ENVS = {"HalfCheetah", "Swimmer"}

class PartialObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, base_name: str):
        super().__init__(env)
        self.keep_indices = PARTIAL_KEEP[base_name]
        low = self.observation_space.low[self.keep_indices]
        high = self.observation_space.high[self.keep_indices]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        return obs[self.keep_indices]


class InvPendShapedReward(gym.Wrapper):
    MAX_BONUS = 200.0 / 1000.0

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        if r > 0.0:
            bonus = (1.0 - abs(float(obs[0]))) * self.MAX_BONUS
            r = float(r) + bonus
        return obs, r, terminated, truncated, info


def make_env(env_key: str):
    cfg = ENV_CONFIGS[env_key]
    env = gym.make(cfg["env_name"])
    if cfg["partial_obs"]:
        env = PartialObsWrapper(env, cfg["base_name"])
    if cfg["base_name"] == "InvertedPendulum":
        env = InvPendShapedReward(env)
    return env


# projeciton
class BottleneckMlpExtractor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 12, bottleneck_dim: int = 2):
        super().__init__()
        self.latent_dim_pi = bottleneck_dim
        self.latent_dim_vf = hidden_dim

        self.w_in = nn.Linear(feature_dim, bottleneck_dim, bias=False)
        self.policy_hidden = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, bottleneck_dim),

        )

        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        x = self.w_in(features)
        x = self.policy_hidden(x)
        return x

    def forward_critic(self, features):
        return self.value_net(features)


class BottleneckPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = BottleneckMlpExtractor(
            feature_dim=self.features_dim,
            hidden_dim=12,
            bottleneck_dim=2,
        )

STANDARD_POLICY_KWARGS = dict(
    net_arch=[12],
    activation_fn=nn.Tanh,
)



def run_baseline(env_key, save_path, total_timesteps=1_000_000):
    cfg = ENV_CONFIGS[env_key]
    use_bottleneck = cfg["base_name"] in BOTTLENECK_ENVS

    ppo_params = dict(
        learning_rate=3e-4, n_steps=2048, batch_size=64,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    )

    if use_bottleneck:
        # Custom policy: w_in bottleneck + [12] tanh + latent_dim_pi=2 -> action_net = w_out
        policy_cls = BottleneckPolicy
        policy_kwargs = {}
    else:
        policy_cls = "MlpPolicy"
        policy_kwargs = STANDARD_POLICY_KWARGS

    all_eval_results = []
    log_base = os.path.join(save_path, f"{env_key}_ppo_logs")

    for seed in SEEDS:
        print(f"\n=== {env_key} | PPO (small MLP) | seed {seed} ===")
        env = make_env(env_key)
        eval_env = make_env(env_key)

        log_path = os.path.join(log_base, f"seed_{seed}")
        eval_callback = EvalCallback(
            eval_env, eval_freq=10_000, n_eval_episodes=50,
            log_path=log_path, verbose=0,
        )

        model = PPO(
            policy_cls, env, seed=seed, verbose=0,
            policy_kwargs=policy_kwargs,
            **ppo_params,
        )
        model.learn(total_timesteps=total_timesteps, callback=eval_callback)

        results = np.load(os.path.join(log_path, "evaluations.npz"))
        all_eval_results.append(results["results"].mean(axis=1))

        env.close()
        eval_env.close()

    # Plot
    eval_curves = np.array(all_eval_results)
    timesteps = results["timesteps"]
    mean_eval = eval_curves.mean(axis=0)
    std_eval = eval_curves.std(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, mean_eval, label=f"small MLP + PPO (n={len(SEEDS)})")
    plt.fill_between(timesteps, mean_eval - std_eval, mean_eval + std_eval, alpha=0.25)
    plt.xlabel("Timesteps")
    plt.ylabel("Average return")
    plt.title(f"{env_key} - small MLP + PPO")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(save_path, f"{env_key}_ppo_small.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    finals = eval_curves[:, -1]
    print(f"{env_key} | small MLP + PPO: {finals.mean():.1f} ± {finals.std():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COMP 579 PPO Baseline (small MLP)")
    parser.add_argument("--save-dir", type=str, default="./results_ppo_small")
    parser.add_argument("--envs", nargs="+",
                        default=list(ENV_CONFIGS.keys()),
                        choices=list(ENV_CONFIGS.keys()),
                        help="Which environments to run")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for env_key in args.envs:
        print(f"\n{'='*60}")
        print(f"  Running: {env_key} | PPO (small MLP)")
        print(f"{'='*60}")
        run_baseline(env_key, args.save_dir, args.timesteps)

    print("\nAll baselines complete!")
