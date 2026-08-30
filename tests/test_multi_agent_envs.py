import gym
import numpy as np
import torch

from training.networks.multi_agent_envs import (
    MultiAgentDummyVecEnv,
    VecMultiAgentPyTorch,
)


class FakeMultiAgentEnv:
    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                "robot_node": gym.spaces.Box(
                    low=-1,
                    high=1,
                    shape=(3, 1, 2),
                    dtype=np.float32,
                )
            }
        )
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2])
        self.step_count = 0
        self.received_actions = None

    def reset(self):
        self.step_count = 0
        return {"robot_node": np.zeros((3, 1, 2), dtype=np.float32)}

    def step(self, actions):
        self.received_actions = np.asarray(actions)
        self.step_count += 1
        done = self.step_count >= 2
        observation = {
            "robot_node": np.full(
                (3, 1, 2),
                self.step_count,
                dtype=np.float32,
            )
        }
        rewards = np.arange(3, dtype=np.float32).reshape(3, 1)
        return observation, rewards, done, {"step_count": self.step_count}

    def close(self):
        pass


def test_multi_agent_dummy_vec_env_preserves_agent_reward_axis():
    vec_env = MultiAgentDummyVecEnv([FakeMultiAgentEnv])

    observation = vec_env.reset()
    assert observation["robot_node"].shape == (1, 3, 1, 2)

    observation, rewards, dones, infos = vec_env.step(
        np.array([[0, 1, 0]], dtype=np.int64)
    )

    assert rewards.shape == (1, 3, 1)
    np.testing.assert_array_equal(vec_env.envs[0].received_actions, [0, 1, 0])
    assert dones.tolist() == [False]
    assert infos[0]["step_count"] == 1


def test_multi_agent_dummy_vec_env_resets_only_on_environment_done():
    vec_env = MultiAgentDummyVecEnv([FakeMultiAgentEnv])
    vec_env.reset()
    vec_env.step(np.array([[0, 0, 0]], dtype=np.int64))

    observation, _, dones, _ = vec_env.step(
        np.array([[0, 0, 0]], dtype=np.int64)
    )

    assert dones.tolist() == [True]
    np.testing.assert_array_equal(observation["robot_node"], 0.0)


def test_torch_wrapper_keeps_env_and_agent_dimensions():
    numpy_env = MultiAgentDummyVecEnv([FakeMultiAgentEnv])
    env = VecMultiAgentPyTorch(numpy_env, torch.device("cpu"))

    observation = env.reset()
    assert observation["robot_node"].shape == (1, 3, 1, 2)
    assert observation["robot_node"].dtype == torch.float32

    observation, rewards, dones, _ = env.step(
        torch.tensor([[[0], [1], [0]]], dtype=torch.long)
    )

    assert observation["robot_node"].shape == (1, 3, 1, 2)
    assert rewards.shape == (1, 3, 1)
    assert rewards.dtype == torch.float32
    assert dones.tolist() == [False]
