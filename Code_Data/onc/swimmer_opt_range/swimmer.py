import argparse
import datetime
import os
import random as rng
import subprocess
import sys

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYBNN_DIR = os.path.normpath(os.path.join(CURRENT_DIR, "..", "pybnn", "bin"))
if PYBNN_DIR not in sys.path:
    sys.path.insert(0, PYBNN_DIR)

import pybnn

DEFAULT_SENSORY_RANGE_SCALE = 10.0
SENSORY_RANGE_NOISE_STD = 0.1
MIN_SENSORY_RANGE_SCALE = 0.1
MAX_SENSORY_RANGE_SCALE = 1000.0


class TWsearchEnv:
    def __init__(self, env, filter_len, mean_len, record_video=None, partial_obs=False):
        self.env = env
        self.filter_len = filter_len
        self.mean_len = mean_len
        self.record_video = record_video
        self.video_writer = None
        self.partial_obs = partial_obs
        self.log_input_range = False
        self.input_range_path = None
        self.input_mins = None
        self.input_maxs = None
        self.input_sums = None
        self.input_count = 0
        self.sensory_range_scale = DEFAULT_SENSORY_RANGE_SCALE

    def TensorRGBToImage(self, tensor):
        new_im = Image.new("RGB", (tensor.shape[1], tensor.shape[0]))
        pixels = []
        for y in range(tensor.shape[0]):
            for x in range(tensor.shape[1]):
                r = tensor[y][x][0]
                g = tensor[y][x][1]
                b = tensor[y][x][2]
                pixels.append((r, g, b))
        new_im.putdata(pixels)
        return new_im

    def input_size(self):
        if self.partial_obs:
            return 2
        return int(self.env.observation_space.shape[0])

    def output_size(self):
        return int(self.env.action_space.shape[0])

    def get_observation_slice(self, obs):
        if self.partial_obs:
            return obs[:2]
        return obs

    def set_observations_for_lif(self, obs, observations):
        obs_slice = self.get_observation_slice(obs)
        v = np.dot(obs_slice, self.w_in) / self.sensory_range_scale
        if self.log_input_range:
            if self.input_mins is None:
                self.input_mins = np.array(v, dtype=float)
                self.input_maxs = np.array(v, dtype=float)
                self.input_sums = np.array(v, dtype=float)
                self.input_count = 1
            else:
                self.input_mins = np.minimum(self.input_mins, v)
                self.input_maxs = np.maximum(self.input_maxs, v)
                self.input_sums += v
                self.input_count += 1
        observations[0] = float(v[0])
        observations[1] = float(v[1])

    def post_process_action(self, action):
        action = np.array(action)
        actions = np.dot(action, self.w_out)
        return actions

    def run_one_episode(self, do_render=False):
        obs, _ = self.env.reset()
        self.lif.Reset()
        if do_render:
            rewardlog = open("rewardlog.log", "w")
            self.lif.DumpClear("lif-dump.csv")
            if self.record_video and self.video_writer is None:
                self.video_writer = imageio.get_writer(self.record_video, fps=60)

        observations = [float(0), float(0)]
        self.set_observations_for_lif(obs, observations)
        self.lif.Update(observations, 0.01, 10)

        total_reward = np.zeros(1)
        time = 0.0

        while 1:
            action = self.lif.Update(observations, 0.01, 10)
            actions = self.post_process_action(action)
            obs, r, terminated, truncated, info = self.env.step(actions)
            self.set_observations_for_lif(obs, observations)
            total_reward += r
            time += 0.0165

            if do_render:
                rewardlog.write(str(total_reward) + "\n")
                rewardlog.flush()
                self.lif.DumpState("lif-dump.csv")
                frame = self.env.render()
                if self.video_writer is not None:
                    self.video_writer.append_data(frame)
                if time >= 16.5:
                    return
            elif terminated or truncated:
                break

        return np.sum(total_reward)

    def evaluate_avg(self):
        N = 50
        returns = np.zeros(N)
        for i in range(N):
            returns[i] = self.run_one_episode()
        return np.mean(returns)

    def run_multiple_episodes(self):
        returns = np.zeros(self.filter_len)
        for i in range(self.filter_len):
            returns[i] = self.run_one_episode()

        sort = np.sort(returns)
        worst_cases = sort[0:self.mean_len]
        return [np.mean(worst_cases), np.mean(returns)]

    def load_tw(self, filename):
        self.lif = pybnn.LifNet(filename)

        aux_files = filename.replace(".bnn", ".npz")
        if os.path.isfile(aux_files):
            nd = np.load(aux_files)
            self.w_in = nd["w_in"]
            self.w_out = nd["w_out"]
            if "sensory_range_scale" in nd.files:
                self.sensory_range_scale = float(nd["sensory_range_scale"])
            else:
                self.sensory_range_scale = DEFAULT_SENSORY_RANGE_SCALE
            if self.w_in.shape != (self.input_size(), 2):
                self.w_in = np.random.normal(0, 1, size=[self.input_size(), 2])
            if self.w_out.shape != (2, self.output_size()):
                self.w_out = np.random.normal(0, 1, size=[2, self.output_size()])
        else:
            self.w_in = np.random.normal(0, 1, size=[self.input_size(), 2])
            self.w_out = np.random.normal(0, 1, size=[2, self.output_size()])
            self.sensory_range_scale = DEFAULT_SENSORY_RANGE_SCALE

        self.sensory_range_scale = min(
            MAX_SENSORY_RANGE_SCALE,
            max(MIN_SENSORY_RANGE_SCALE, self.sensory_range_scale),
        )
        self.lif.AddBiSensoryNeuron(1, 6, -1, 1)
        self.lif.AddBiSensoryNeuron(7, 0, -1, 1)
        self.lif.AddMotorNeuron(9, 1)
        self.lif.AddMotorNeuron(10, 1)
        self.lif.Reset()

    def store_tw(self, filename):
        aux_files = filename.replace(".bnn", ".npz")
        self.lif.WriteToFile(filename)
        np.savez(
            aux_files,
            w_in=self.w_in,
            w_out=self.w_out,
            sensory_range_scale=self.sensory_range_scale,
        )

    def optimize(self, ts=datetime.timedelta(seconds=60), max_steps=1000000):
        self.lif.AddNoise(0.5, 15)
        self.lif.AddNoiseVleak(8, 8)
        self.lif.AddNoiseGleak(0.2, 8)
        self.lif.AddNoiseSigma(0.2, 10)
        self.lif.AddNoiseCm(0.1, 10)
        self.lif.CommitNoise()

        r_values = np.zeros(1000000)
        r_counter = 0

        current_return, mean_ret = self.run_multiple_episodes()
        r_values[r_counter] = mean_ret
        r_counter += 1

        num_distortions = 4
        num_distortions_sigma = 3
        num_distortions_vleak = 2
        num_distortions_gleak = 2
        num_distortions_cm = 2
        steps_since_last_improvement = 0

        starttime = datetime.datetime.now()
        endtime = starttime + ts
        steps = -1
        log_freq = 50
        while endtime > datetime.datetime.now() and steps < max_steps:
            steps += 1

            self.backup_w_in = np.copy(self.w_in)
            self.backup_w_out = np.copy(self.w_out)
            self.backup_sensory_range_scale = self.sensory_range_scale
            self.w_in = self.w_in + np.random.normal(size=self.w_in.shape)
            self.w_out = self.w_out + np.random.normal(size=self.w_out.shape)
            self.sensory_range_scale = float(
                np.exp(np.log(self.sensory_range_scale) + np.random.normal(0, SENSORY_RANGE_NOISE_STD))
            )
            self.sensory_range_scale = min(
                MAX_SENSORY_RANGE_SCALE,
                max(MIN_SENSORY_RANGE_SCALE, self.sensory_range_scale),
            )

            distortions = rng.randint(0, num_distortions)
            variance = rng.uniform(0.01, 0.4)
            distortions_sigma = rng.randint(0, num_distortions_sigma)
            variance_sigma = rng.uniform(0.01, 0.05)
            distortions_vleak = rng.randint(0, num_distortions_vleak)
            variance_vleak = rng.uniform(0.1, 3)
            distortions_gleak = rng.randint(0, num_distortions_gleak)
            variance_gleak = rng.uniform(0.05, 0.5)
            distortions_cm = rng.randint(0, num_distortions_cm)
            variance_cm = rng.uniform(0.01, 0.1)

            self.lif.AddNoise(variance, distortions)
            self.lif.AddNoiseSigma(variance_sigma, distortions_sigma)
            self.lif.AddNoiseVleak(variance_vleak, distortions_vleak)
            self.lif.AddNoiseCm(variance_cm, distortions_cm)
            self.lif.AddNoiseGleak(variance_gleak, distortions_gleak)

            new_return, mean_ret = self.run_multiple_episodes()
            r_values[r_counter] = mean_ret
            r_counter += 1
            if new_return > current_return:
                if self.logfile is not None:
                    elapsed = datetime.datetime.now() - starttime
                    self.logfile.write(
                        "Improvement after: "
                        + str(steps)
                        + " steps, with return "
                        + str(new_return)
                        + ", Elapsed: "
                        + str(elapsed.total_seconds())
                        + "\n"
                    )
                    self.logfile.flush()

                current_return = new_return
                self.lif.CommitNoise()
                steps_since_last_improvement = 0
                num_distortions = max(4, num_distortions - 1)
                num_distortions_sigma = max(3, num_distortions_sigma - 1)
                num_distortions_vleak = max(2, num_distortions_vleak - 1)
                num_distortions_gleak = max(2, num_distortions_gleak - 1)
                num_distortions_cm = max(2, num_distortions_cm - 1)
            else:
                steps_since_last_improvement += 1
                self.lif.UndoNoise()
                self.w_in = self.backup_w_in
                self.w_out = self.backup_w_out
                self.sensory_range_scale = self.backup_sensory_range_scale

                if steps_since_last_improvement > 50:
                    steps_since_last_improvement = 0
                    current_return, mean_ret = self.run_multiple_episodes()
                    r_values[r_counter] = mean_ret
                    r_counter += 1
                    if self.logfile is not None:
                        self.logfile.write(
                            "Reevaluate after: "
                            + str(steps)
                            + " steps, with return "
                            + str(current_return)
                            + "\n"
                        )
                        self.logfile.flush()

                    num_distortions = min(12, num_distortions + 1)
                    num_distortions_sigma = min(8, num_distortions_sigma + 1)
                    num_distortions_vleak = min(6, num_distortions_vleak + 1)
                    num_distortions_gleak = min(6, num_distortions_gleak + 1)
                    num_distortions_cm = min(4, num_distortions_cm + 1)

            if steps % log_freq == 0 and self.csvlogfile is not None:
                elapsed = datetime.datetime.now() - starttime
                avg_cost = self.evaluate_avg()
                performance_r = np.mean(r_values[0:r_counter])
                self.csvlogfile.write(
                    str(steps)
                    + ";"
                    + str(avg_cost)
                    + ";"
                    + str(performance_r)
                    + ";"
                    + str(elapsed.total_seconds())
                    + "\n"
                )
                self.csvlogfile.flush()

        if self.logfile is not None:
            self.logfile.write("Total steps done: " + str(steps) + "\n")
            self.logfile.close()
        if self.csvlogfile is not None:
            elapsed = datetime.datetime.now() - starttime
            avg_cost = self.evaluate_avg()
            performance_r = np.mean(r_values[0:r_counter])
            self.csvlogfile.write(
                str(steps)
                + ";"
                + str(avg_cost)
                + ";"
                + str(performance_r)
                + ";"
                + str(elapsed.total_seconds())
                + "\n"
            )
            self.csvlogfile.flush()

    def replay(self, filename):
        self.load_tw(filename)
        if not os.path.exists("vid"):
            os.makedirs("vid")
        print("Average Reward: " + str(self.evaluate_avg()))
        print("Replay Return: " + str(self.run_multiple_episodes()))
        self.input_mins = None
        self.input_maxs = None
        self.input_sums = None
        self.input_count = 0
        try:
            self.run_one_episode(True)
        finally:
            if self.video_writer is not None:
                self.video_writer.close()
                self.video_writer = None
        if self.log_input_range and self.input_range_path is not None and self.input_mins is not None:
            input_means = self.input_sums / self.input_count
            with open(self.input_range_path, "w") as handle:
                handle.write("Projected ONC input ranges during replay\n")
                handle.write("sensory_range_scale: {}\n".format(float(self.sensory_range_scale)))
                handle.write("input_0_min: {}\n".format(float(self.input_mins[0])))
                handle.write("input_0_max: {}\n".format(float(self.input_maxs[0])))
                handle.write("input_0_mean: {}\n".format(float(input_means[0])))
                handle.write("input_1_min: {}\n".format(float(self.input_mins[1])))
                handle.write("input_1_max: {}\n".format(float(self.input_maxs[1])))
                handle.write("input_1_mean: {}\n".format(float(input_means[1])))
            print("Saved input range to " + self.input_range_path)

    def optimize_and_store(self, worker_id, in_file="tw_pure.bnn"):
        self.load_tw(in_file)

        if worker_id.isdigit():
            seed = int(worker_id) + 20 * datetime.datetime.now().microsecond + 23115
        else:
            seed = 20 * datetime.datetime.now().microsecond + 23115

        self.lif.SeedRandomNumberGenerator(seed)
        rng.seed(seed)

        root_path = "results/filter_" + str(self.filter_len) + "_" + str(self.mean_len)
        log_path = root_path + "/logs_csv"
        log_path_txt = root_path + "/logs_txt"
        store_path = root_path + "/final"

        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(log_path_txt):
            os.makedirs(log_path_txt)
        if not os.path.exists(store_path):
            os.makedirs(store_path)

        log_file = log_path_txt + "/textlog_" + worker_id + ".log"
        csv_log = log_path + "/csvlog_" + worker_id + ".log"
        self.logfile = open(log_file, "w")
        self.csvlogfile = open(csv_log, "w")

        print("Begin Return of " + worker_id + ": " + str(self.run_multiple_episodes()))
        self.optimize(ts=datetime.timedelta(hours=6), max_steps=20000)
        print("End Return: of " + worker_id + ": " + str(self.run_multiple_episodes()))

        outfile = store_path + "/tw-optimized_" + worker_id + ".bnn"
        self.store_tw(outfile)

    def optimize_and_store_experiment(self, seed, in_file="tw_pure.bnn", experiment_name="experiment"):
        self.load_tw(in_file)

        self.lif.SeedRandomNumberGenerator(seed)
        rng.seed(seed)
        np.random.seed(seed)
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)

        root_path = os.path.join(CURRENT_DIR, "results", experiment_name)
        os.makedirs(root_path, exist_ok=True)

        log_file = root_path + "/textlog_" + str(seed) + ".log"
        csv_log = root_path + "/csvlog_" + str(seed) + ".log"
        self.logfile = open(log_file, "w")
        self.csvlogfile = open(csv_log, "w")

        try:
            print("Begin Return of " + str(seed) + ": " + str(self.run_multiple_episodes()))
            self.optimize(ts=datetime.timedelta(hours=6), max_steps=25000)
            print("End Return of " + str(seed) + ": " + str(self.run_multiple_episodes()))

            outfile = root_path + "/tw-optimized_" + str(seed) + ".bnn"
            self.store_tw(outfile)
        finally:
            if self.logfile is not None:
                self.logfile.close()
                self.logfile = None
            if self.csvlogfile is not None:
                self.csvlogfile.close()
                self.csvlogfile = None


def experiment(partial_obs=False):
    seeds = [0, 1, 2, 3, 4]
    filter_len = 10
    mean_len = 5
    experiment_name = "experiment_partial" if partial_obs else "experiment_full"
    root_path = os.path.join(CURRENT_DIR, "results", experiment_name)

    os.makedirs(root_path, exist_ok=True)

    processes = []
    for seed in seeds:
        print("Starting run with seed: " + str(seed))
        cmd = [
            sys.executable,
            os.path.abspath(__file__),
            "--experiment-seed",
            str(seed),
            "--filter",
            str(filter_len),
            "--mean",
            str(mean_len),
        ]
        if partial_obs:
            cmd.append("--partial-obs")
        processes.append((seed, subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))))

    for seed, process in processes:
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError("Experiment run failed for seed {} with exit code {}".format(seed, return_code))

    curves = []
    steps = None
    for seed in seeds:
        csv_path = root_path + "/csvlog_" + str(seed) + ".log"
        current_steps = []
        eval_returns = []
        with open(csv_path, "r") as handle:
            for line in handle:
                parts = line.strip().split(";")
                if len(parts) < 2:
                    continue
                current_steps.append(int(parts[0]))
                eval_returns.append(float(parts[1]))
        if steps is None:
            steps = current_steps
        elif steps != current_steps:
            raise ValueError("CSV logs do not share the same checkpoints.")
        curves.append(eval_returns)

    curve_array = np.array(curves)
    mean_values = np.mean(curve_array, axis=0)
    std_values = np.std(curve_array, axis=0)
    output_path = root_path + "/experiment_mean_return.png"
    plt.figure(figsize=(8, 5))
    plt.plot(steps, mean_values, label="Average over {} seeds".format(len(seeds)))
    plt.fill_between(steps, mean_values - std_values, mean_values + std_values, alpha=0.25)
    plt.xlabel("Search iterations")
    plt.ylabel("Mean return over 50 eval episodes")
    if partial_obs:
        plt.title("Swimmer with Partial Obs. - ONC with ARS (Optimized Sensory Range)")
    else:
        plt.title("Swimmer - ONC with ARS (Optimized Sensory Range)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    final_returns = curve_array[:, -1]
    print("Final eval return: {:.1f} +- {:.1f}".format(np.mean(final_returns), np.std(final_returns)))
    print("Saved experiment plot to " + output_path)

    summary_path = root_path + "/final_return_summary.txt"
    with open(summary_path, "w") as handle:
        handle.write("Experiment: " + experiment_name + "\n")
        handle.write("Metric: final avg_cost from each csvlog (mean return over 50 eval episodes)\n\n")
        for seed, value in zip(seeds, final_returns):
            handle.write("Seed {}: {}\n".format(seed, value))
        handle.write("\n")
        handle.write("Final mean return: {:.6f}\n".format(np.mean(final_returns)))
        handle.write("Final std return: {:.6f}\n".format(np.std(final_returns)))
        handle.write("Final mean +- std: {:.1f} +- {:.1f}\n".format(np.mean(final_returns), np.std(final_returns)))


def demo_run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", default=10, type=int)
    parser.add_argument("--mean", default=5, type=int)
    parser.add_argument("--file", default="tw_pure.bnn")
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--id", default="0")
    parser.add_argument("--record-video")
    args = parser.parse_args()

    render_mode = None if args.optimize else "human"
    if args.record_video:
        render_mode = "rgb_array"
    env = gym.make("Swimmer-v5", render_mode=render_mode)

    twenv = TWsearchEnv(env, args.filter, args.mean, args.record_video)
    if args.optimize:
        print("Optimize")
        twenv.optimize_and_store(str(args.id), args.file)
    else:
        twenv.log_input_range = True
        twenv.input_range_path = os.path.join(CURRENT_DIR, "swimmer_opt_range_input_range.txt")
        print("Replay")
        twenv.replay(args.file)


if __name__ == "__main__":
    if "--experiment-seed" in sys.argv:
        seed_index = sys.argv.index("--experiment-seed")
        seed = int(sys.argv[seed_index + 1])
        partial_obs = "--partial-obs" in sys.argv
        filter_len = 10
        mean_len = 5
        if "--filter" in sys.argv:
            filter_index = sys.argv.index("--filter")
            filter_len = int(sys.argv[filter_index + 1])
        if "--mean" in sys.argv:
            mean_index = sys.argv.index("--mean")
            mean_len = int(sys.argv[mean_index + 1])

        env = gym.make("Swimmer-v5", render_mode=None)
        twenv = TWsearchEnv(env, filter_len, mean_len, partial_obs=partial_obs)
        try:
            experiment_name = "experiment_partial" if partial_obs else "experiment_full"
            twenv.optimize_and_store_experiment(seed, experiment_name=experiment_name)
        finally:
            env.close()
    elif "--experiment_full" in sys.argv:
        experiment()
    elif "--experiment_partial" in sys.argv:
        experiment(partial_obs=True)
    else:
        demo_run()
