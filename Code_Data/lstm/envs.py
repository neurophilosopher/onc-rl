"""Environment wrappers and factory for the MuJoCo control tasks."""
import gymnasium as gym


class PartialObservabilityWrapper(gym.ObservationWrapper):
    """Drops the given observation indices to make the task partially observable."""
    def __init__(self, env, masked_indices):
        super().__init__(env)
        self.masked_indices = sorted(masked_indices)
        full_obs = env.observation_space
        keep = [i for i in range(full_obs.shape[0]) if i not in self.masked_indices]
        self.keep_indices = keep
        low = full_obs.low[keep]; high = full_obs.high[keep]
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=full_obs.dtype)

    def observation(self, obs):
        return obs[self.keep_indices]


class InvertedPendulumRewardShaping(gym.Wrapper):
    """Adds a centered-pole bonus to the InvertedPendulum reward, matching Hasani et al.'s code.
    When reward > 0, adds up to +0.2 per step scaled by (1 - |cart_position|)."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward > 0.0:
            max_bonus = 200.0 / 1000.0
            bonus = (1.0 - abs(float(obs[0]))) * max_bonus
            reward = reward + bonus
        return obs, reward, terminated, truncated, info


PO_MASK = {
    "HalfCheetah-v4": list(range(8, 17)),
    "InvertedPendulum-v4": [1, 3],
}


def make_env(env_name, partial=False):
    env = gym.make(env_name)
    if env_name == "InvertedPendulum-v4":
        env = InvertedPendulumRewardShaping(env)
    if partial and env_name in PO_MASK:
        env = PartialObservabilityWrapper(env, PO_MASK[env_name])
    return env