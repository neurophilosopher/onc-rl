
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import gymnasium as gym
import pickle
import argparse


ENV_CONFIGS = {
    "invpend_full": {
        "env_name": "InvertedPendulum-v5",
        "base_name": "InvertedPendulum",
        "partial_obs": False,
        "obs_dim": 4,
        "act_dim": 1,
    },
    "invpend_partial": {
        "env_name": "InvertedPendulum-v5",
        "base_name": "InvertedPendulum",
        "partial_obs": True,
        "obs_dim": 2,
        "act_dim": 1,
    },
    "cheetah_full": {
        "env_name": "HalfCheetah-v5",
        "base_name": "HalfCheetah",
        "partial_obs": False,
        "obs_dim": 17,
        "act_dim": 6,
    },
    "cheetah_partial": {
        "env_name": "HalfCheetah-v5",
        "base_name": "HalfCheetah",
        "partial_obs": True,
        "obs_dim": 8,
        "act_dim": 6,
    },
    "swimmer_full": {
        "env_name": "Swimmer-v5",
        "base_name": "Swimmer",
        "partial_obs": False,
        "obs_dim": 8,
        "act_dim": 2,
    },
}

TOTAL_STEPS = {
    "invpend_full": 20_000,
    "invpend_partial": 20_000,
    "cheetah_full": 20_000,
    "cheetah_partial": 20_000,
    "swimmer_full": 20_000,
}

PARTIAL_KEEP = {
    "HalfCheetah": list(range(0, 8)),
    "InvertedPendulum": [0, 1],
    "Swimmer": [0, 1],
}

SEEDS = list(range(5))



class PartialObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, base_name: str):
        super().__init__(env)
        self.base_name = base_name
        self.keep_indices = PARTIAL_KEEP[base_name]

        low = self.observation_space.low[self.keep_indices]
        high = self.observation_space.high[self.keep_indices]
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float32,
        )

    def observation(self, obs):
        return obs[self.keep_indices]


def make_env(env_key: str, seed: int = 0) -> gym.Env:
    cfg = ENV_CONFIGS[env_key]
    env = gym.make(cfg["env_name"])
    if cfg["partial_obs"]:
        env = PartialObsWrapper(env, cfg["base_name"])
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env

class MLPController:
    def __init__(self, input_dim: int = 2, hidden_dim: int = 12, output_dim: int = 1):
        self.W1 = np.zeros((hidden_dim, input_dim), dtype=np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = np.zeros((output_dim, hidden_dim), dtype=np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)
        self._backup = None

    def reset_state(self):
        pass

    def step(self, obs: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W1 @ obs.astype(np.float32) + self.b1)
        return (self.W2 @ h + self.b2).astype(np.float32)

    def add_noise(self, amplitude: float):
        self._backup = (self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())
        for arr in (self.W1, self.b1, self.W2, self.b2):
            arr += np.random.randn(*arr.shape).astype(np.float32) * amplitude

    def undo_noise(self):
        if self._backup is None:
            return
        self.W1, self.b1, self.W2, self.b2 = self._backup
        self._backup = None


class NNsearchEnv:
    def __init__(self, env_key: str, filter_len: int, mean_len: int,
                 nn_type: str = "mlp", seed: int = 0):
        assert env_key in ENV_CONFIGS, f"Unknown env_key: {env_key}"
        assert nn_type == "mlp", "This runner currently supports only mlp."

        self.env_key = env_key
        self.cfg = ENV_CONFIGS[env_key]
        self.env_name = self.cfg["base_name"]
        self.filter_len = filter_len
        self.mean_len = mean_len
        self.nn_type = nn_type
        self.seed = seed

        self.env = make_env(env_key, seed=seed)
        self.create_nn(nn_type)

        self.history_steps = []
        self.history_eval_avg = []
        self.history_running_perf = []

    def input_size(self) -> int:
        return int(self.env.observation_space.shape[0])

    def output_size(self) -> int:
        return int(self.env.action_space.shape[0])

    def create_nn(self, nn_type: str):
        input_dim = 2
        output_dim = 1

        if self.env_name in ["HalfCheetah", "Swimmer"]:
            output_dim = 2
            self.w_in = np.zeros((self.input_size(), 2), dtype=np.float32)
            self.w_out = np.zeros((2, self.output_size()), dtype=np.float32)
        else:
            self.w_in = None
            self.w_out = None
            if self.env_name == "InvertedPendulum" and not self.cfg["partial_obs"]:
                input_dim = 4

        self.w_backup = None
        self.nn = MLPController(input_dim=input_dim, hidden_dim=12, output_dim=output_dim)

    def preprocess_observations(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)

        if self.env_name == "InvertedPendulum":
            if self.cfg["partial_obs"]:
                if obs.shape[0] == 2:
                    return obs
                if obs.shape[0] >= 4:
                    return np.array([obs[1], obs[0]], dtype=np.float32)
                raise ValueError(f"Unexpected InvertedPendulum obs shape: {obs.shape}")
            return obs  # full 4-dim

        if self.env_name in ["HalfCheetah", "Swimmer"]:
            return np.dot(obs, self.w_in).astype(np.float32)

        raise ValueError(f"Unexpected env_name: {self.env_name}")

    def preprocess_actions(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        if self.env_name in ["HalfCheetah", "Swimmer"]:
            return np.dot(action, self.w_out).astype(np.float32)
        return action.astype(np.float32)

    def run_one_episode(self, do_render: bool = False) -> float:
        total_reward = 0.0
        obs, _ = self.env.reset()
        self.nn.reset_state()

        while True:
            if do_render:
                self.env.render()

            proc_obs = self.preprocess_observations(obs)
            action = self.nn.step(proc_obs)
            action = self.preprocess_actions(action)

            next_obs, r, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if self.env_name == "InvertedPendulum":
                max_bonus = 200.0 / 1000.0
                bonus = (1.0 - abs(float(next_obs[0]))) * max_bonus
                if r > 0.0:
                    total_reward += bonus

            total_reward += float(r)
            obs = next_obs

            if done:
                break

        return float(total_reward)

    def evaluate_avg(self, N: int = 50) -> float:
        returns = np.zeros(N, dtype=np.float32)
        for i in range(N):
            returns[i] = self.run_one_episode()
        return float(np.mean(returns))

    def run_multiple_episodes(self):
        returns = np.zeros(self.filter_len, dtype=np.float32)
        for i in range(self.filter_len):
            returns[i] = self.run_one_episode()

        sort = np.sort(returns)
        worst_cases = sort[:self.mean_len]
        return float(np.mean(worst_cases)), float(np.mean(returns))

    def optimize(self, max_steps: int, log_freq: int = 250):
        self.nn.add_noise(0.01)

        r_values = np.zeros(max_steps + 5, dtype=np.float32)
        r_counter = 0

        current_return, mean_ret = self.run_multiple_episodes()
        r_values[r_counter] = mean_ret
        r_counter += 1

        steps_since_last_improvement = 0
        amplitude = 0.01
        steps = -1

        while steps < max_steps:
            steps += 1

            self.nn.add_noise(amplitude)

            if self.env_name in ["HalfCheetah", "Swimmer"]:
                self.w_backup = [np.copy(self.w_in), np.copy(self.w_out)]
                self.w_in += np.random.normal(0, amplitude, size=self.w_in.shape).astype(np.float32)
                self.w_out += np.random.normal(0, amplitude, size=self.w_out.shape).astype(np.float32)

            new_return, mean_ret = self.run_multiple_episodes()
            r_values[r_counter] = mean_ret
            r_counter += 1

            if new_return > current_return:
                current_return = new_return
                steps_since_last_improvement = 0
                amplitude /= 2.0
                if amplitude < 0.01:
                    amplitude = 0.01
            else:
                self.nn.undo_noise()

                if self.env_name in ["HalfCheetah", "Swimmer"]:
                    self.w_in, self.w_out = self.w_backup

                steps_since_last_improvement += 1

                if steps_since_last_improvement > 50:
                    amplitude *= 5.0
                    if amplitude > 1.0:
                        amplitude = 1.0
                    steps_since_last_improvement = 0
                    current_return, mean_ret = self.run_multiple_episodes()

            if steps % log_freq == 0:
                avg_eval = self.evaluate_avg(N=50)
                performance_r = float(np.mean(r_values[:r_counter]))

                self.history_steps.append(steps)
                self.history_eval_avg.append(avg_eval)
                self.history_running_perf.append(performance_r)

                print(
                    f"step={steps:5d} | "
                    f"objective={current_return:9.3f} | "
                    f"eval_avg={avg_eval:9.3f} | "
                    f"running_perf={performance_r:9.3f} | "
                    f"amp={amplitude:.4f}",
                    flush=True,
                )

    def close(self):
        self.env.close()


def run_single(env_key, nn_type, seed, steps=None,
               filter_len=10, mean_len=5, log_freq=50):
    if steps is None:
        steps = TOTAL_STEPS[env_key]

    np.random.seed(seed)

    runner = NNsearchEnv(
        env_key=env_key, filter_len=filter_len, mean_len=mean_len,
        nn_type=nn_type, seed=seed,
    )

    print(f"Begin return: {runner.run_multiple_episodes()}")
    runner.optimize(max_steps=steps, log_freq=log_freq)
    print(f"End return: {runner.run_multiple_episodes()}")
    return runner


def plot_multi_seed(runners, title, save_path):
    eval_curves = np.array([r.history_eval_avg for r in runners])
    steps = np.array(runners[0].history_steps)

    mean_eval = eval_curves.mean(axis=0)
    std_eval = eval_curves.std(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(steps, mean_eval, label=f"Eval avg (n={len(runners)})")
    plt.fill_between(steps, mean_eval - std_eval, mean_eval + std_eval, alpha=0.25)
    plt.xlabel("Search iterations")
    plt.ylabel("Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_name = title.replace(" ", "_").replace("-", "_") + ".png"
    fig_path = os.path.join(save_path, fig_name)
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Figure saved: {fig_path}")


def run_env_experiment(env_key, save_path, steps=25_000):
    pkl_path = os.path.join(save_path, f"runners_{env_key}.pkl")

    # Resume from checkpoint if it exists
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            runners = pickle.load(f)
        print(f"Resuming {env_key}: {len(runners)}/{len(SEEDS)} seeds done")
    else:
        runners = []

    done_seeds = len(runners)
    for seed in SEEDS[done_seeds:]:
        print(f"\n=== {env_key} | seed {seed} ===")
        r = run_single(env_key, "mlp", seed, steps=steps)
        runners.append(r)

        with open(pkl_path, "wb") as f:
            pickle.dump(runners, f)
        print(f"Saved {len(runners)}/{len(SEEDS)} seeds")

    # Print summary
    final_returns = [r.history_eval_avg[-1] for r in runners]
    arr = np.array(final_returns)
    print(f"\n{env_key} | Final eval: {arr.mean():.1f} ± {arr.std():.1f}")

    # Save plot
    title = env_key.replace("_", " ").title() + " - MLP ARS"
    plot_multi_seed(runners, title, save_path)

    return runners

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="COMP 579 ARS Experiments")
    parser.add_argument(
        "--save-dir", type=str, default="./results",
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--envs", nargs="+",
        default=["invpend_full", "invpend_partial", "cheetah_full",
                 "cheetah_partial", "swimmer_full"],
        choices=list(ENV_CONFIGS.keys()),
        help="Which environments to run",
    )
    parser.add_argument(
        "--steps", type=int, default=25_000,
        help="Number of ARS search steps per seed (default: 25000)",
    )
    args = parser.parse_args()

    save_path = args.save_dir
    os.makedirs(save_path, exist_ok=True)
    print(f"Saving results to: {os.path.abspath(save_path)}")

    for env_key in args.envs:
        print(f"\n{'='*60}")
        print(f"  Running: {env_key}")
        print(f"{'='*60}")
        run_env_experiment(env_key, save_path, steps=args.steps)

    print("\nAll experiments complete!")
