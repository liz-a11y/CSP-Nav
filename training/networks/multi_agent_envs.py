import gym
import numpy as np
import torch

from baselines.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info
from baselines.common.vec_env.vec_env import VecEnv, VecEnvWrapper
from crowd_sim.wrappers.lmte_csp_wrapper import LMTECSPWrapper
from training.networks.shmem_vec_env import ShmemVecEnv


class MultiAgentDummyVecEnv(VecEnv):
    """Sequential VecEnv that supports vector rewards per physical environment."""

    def __init__(self, env_fns):
        self.envs = [env_fn() if callable(env_fn) else env_fn for env_fn in env_fns]
        env = self.envs[0]
        super().__init__(len(self.envs), env.observation_space, env.action_space)
        self.keys, shapes, dtypes = obs_space_info(env.observation_space)
        self.buf_obs = {
            key: np.zeros((self.num_envs,) + tuple(shapes[key]), dtype=dtypes[key])
            for key in self.keys
        }
        if isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.agent_num = len(env.action_space.nvec)
        else:
            self.agent_num = env.action_space.shape[0]
        self.buf_rews = np.zeros(
            (self.num_envs, self.agent_num, 1),
            dtype=np.float32,
        )
        self.buf_dones = np.zeros(self.num_envs, dtype=np.bool_)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def _save_obs(self, env_id, observation):
        for key in self.keys:
            self.buf_obs[key][env_id] = (
                observation if key is None else observation[key]
            )

    def _obs_from_buf(self):
        return dict_to_obs(copy_obs_dict(self.buf_obs))

    def reset(self):
        for env_id, env in enumerate(self.envs):
            self._save_obs(env_id, env.reset())
        return self._obs_from_buf()

    def step_async(self, actions):
        actions = np.asarray(actions)
        if actions.shape[0] != self.num_envs:
            raise ValueError("actions must have one entry per physical environment")
        self.actions = actions

    def step_wait(self):
        for env_id, env in enumerate(self.envs):
            observation, reward, done, info = env.step(self.actions[env_id])
            self.buf_rews[env_id] = np.asarray(reward, dtype=np.float32).reshape(
                self.agent_num,
                1,
            )
            self.buf_dones[env_id] = done
            self.buf_infos[env_id] = info
            if done:
                observation = env.reset()
            self._save_obs(env_id, observation)
        return (
            self._obs_from_buf(),
            self.buf_rews.copy(),
            self.buf_dones.copy(),
            self.buf_infos.copy(),
        )

    def close_extras(self):
        for env in self.envs:
            env.close()

    def get_images(self):
        return [env.render(mode="rgb_array") for env in self.envs]


class VecMultiAgentPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super().__init__(venv)
        self.device = device

    def _to_torch(self, observation):
        if isinstance(observation, dict):
            return {
                key: torch.from_numpy(value).float().to(self.device)
                for key, value in observation.items()
            }
        return torch.from_numpy(observation).float().to(self.device)

    def reset(self):
        return self._to_torch(self.venv.reset())

    def step_async(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim >= 3 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        self.venv.step_async(actions)

    def step_wait(self):
        observation, rewards, dones, infos = self.venv.step_wait()
        return (
            self._to_torch(observation),
            torch.from_numpy(rewards).float().to(self.device),
            dones,
            infos,
        )


def make_multi_agent_env(
    env_id,
    seed,
    rank,
    config,
    env_num,
    phase=None,
    test_case=-1,
):
    def _thunk():
        env_seed = seed + rank if seed is not None else None
        base_env = gym.make(env_id)
        base_env.configure(config)
        base_env.thisSeed = env_seed
        base_env.nenv = env_num
        base_env.phase = phase or ("train" if env_num > 1 else "test")
        if test_case >= 0:
            base_env.test_case = test_case
        base_env.seed(env_seed)
        if getattr(getattr(config, "social", None), "enabled", False):
            return LMTECSPWrapper(base_env, config)
        return base_env

    return _thunk


def make_multi_agent_vec_envs(
    env_name,
    seed,
    num_processes,
    device,
    config,
    phase=None,
    test_case=-1,
):
    env_fns = [
        make_multi_agent_env(
            env_name,
            seed,
            rank,
            config,
            num_processes,
            phase=phase,
            test_case=test_case,
        )
        for rank in range(num_processes)
    ]
    if num_processes == 1:
        envs = MultiAgentDummyVecEnv(env_fns)
    else:
        envs = ShmemVecEnv(env_fns, context="spawn")
    return VecMultiAgentPyTorch(envs, device)
