
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback

SEEDS = list(range(5))

# environment configs - the exact same as in run_ars.py

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


BIG_POLICY_KWARGS = dict(
    net_arch=[64, 64],
    activation_fn=nn.Tanh,
)


def run_baseline(algo_name, env_key, save_path, total_timesteps=1_000_000):
    AlgoClass = PPO if algo_name == "ppo" else A2C

    algo_params = {
        "ppo": dict(
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            policy_kwargs=BIG_POLICY_KWARGS,
        ),
        "a2c": dict(
            learning_rate=7e-4, n_steps=5, gamma=0.99,
            gae_lambda=1.0, ent_coef=0.0, vf_coef=0.5,
            policy_kwargs=BIG_POLICY_KWARGS,
        ),
    }

    all_eval_results = []
    log_base = os.path.join(save_path, f"{env_key}_{algo_name}_logs")

    for seed in SEEDS:
        print(f"\n=== {env_key} | {algo_name.upper()} | seed {seed} ===")
        env = make_env(env_key)
        eval_env = make_env(env_key)

        log_path = os.path.join(log_base, f"seed_{seed}")
        os.makedirs(log_path, exist_ok=True)
        eval_callback = EvalCallback(
            eval_env, eval_freq=10_000, n_eval_episodes=50,
            log_path=log_path, verbose=0,
        )

        model = AlgoClass(
            "MlpPolicy", env, seed=seed, verbose=0,
            **algo_params[algo_name],
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
    plt.plot(timesteps, mean_eval, label=f"MLP+{algo_name.upper()} (n={len(SEEDS)})")
    plt.fill_between(timesteps, mean_eval - std_eval, mean_eval + std_eval, alpha=0.25)
    plt.xlabel("Timesteps")
    plt.ylabel("Average return")
    plt.title(f"{env_key} - MLP + {algo_name.upper()}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(save_path, f"{env_key}_{algo_name}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nFigure saved: {fig_path}")

    finals = eval_curves[:, -1]
    print(f"{env_key} | MLP + {algo_name.upper()}: {finals.mean():.1f} ± {finals.std():.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COMP 579 PPO/A2C Big-MLP Baselines")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--algo", nargs="+", default=["ppo", "a2c"],
                        choices=["ppo", "a2c"])
    parser.add_argument("--envs", nargs="+",
                        default=list(ENV_CONFIGS.keys()),
                        choices=list(ENV_CONFIGS.keys()),
                        help="Which environments to run")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    for env_key in args.envs:
        for algo in args.algo:
            print(f"\n{'='*60}")
            print(f"  Running: {env_key} | {algo.upper()}")
            print(f"{'='*60}")
            run_baseline(algo, env_key, args.save_dir, args.timesteps)

    print("\nAll baselines complete!")
