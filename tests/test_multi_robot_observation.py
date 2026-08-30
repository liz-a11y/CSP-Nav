from types import SimpleNamespace

import numpy as np

from crowd_sim.envs.crowd_sim_tb2_obs_mappo import CrowdSim3DTbObsMAPPO
from crowd_sim.multi_robot_core import AgentStatus


class FakeAgent:
    def __init__(self, px, py, gx=0.0, gy=0.0, vx=0.0, vy=0.0, theta=0.0):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.radius = 0.2
        self.sensor_range = 10.0
        self.FOV = 2 * np.pi


def make_observation_env():
    env = CrowdSim3DTbObsMAPPO.__new__(CrowdSim3DTbObsMAPPO)
    env.robot_num = 3
    env.max_human_num = 4
    env.max_obs_num = 2
    env.ray_num = 8
    env.arena_size = 5.0
    env.config = SimpleNamespace(
        env=SimpleNamespace(action_space="discrete"),
        ob_space=SimpleNamespace(add_human_vel=True),
        robot=SimpleNamespace(v_max=0.5),
    )
    env.robots = [
        FakeAgent(0.0, 0.0, gx=4.0, gy=0.0, theta=0.0),
        FakeAgent(1.0, 0.0, gx=-4.0, gy=0.0, vx=0.1),
        FakeAgent(0.0, 2.0, gx=0.0, gy=-4.0, vy=-0.1),
    ]
    env.humans = [
        FakeAgent(2.0, 0.0, vx=-0.1),
        FakeAgent(20.0, 20.0),
    ]
    env.human_num = len(env.humans)
    env.robot_status = np.array(
        [AgentStatus.ACTIVE, AgentStatus.REACHED, AgentStatus.COLLIDED],
        dtype=object,
    )
    env.ray_test_for_robot = lambda robot_id, include_humans=False: np.full(
        env.ray_num, robot_id + 1.0, dtype=np.float32
    )
    return env


def test_observation_space_has_explicit_three_robot_axis():
    env = make_observation_env()

    env.set_observation_space()

    assert env.observation_space.spaces["robot_node"].shape == (3, 1, 5)
    assert env.observation_space.spaces["temporal_edges"].shape == (3, 1, 2)
    assert env.observation_space.spaces["spatial_edges"].shape == (3, 4, 4)
    assert env.observation_space.spaces["robot_robot_edges"].shape == (3, 2, 5)
    assert env.observation_space.spaces["detected_robot_num"].shape == (3, 1)
    assert env.observation_space.spaces["point_clouds"].shape == (3, 1, 8)
    assert env.observation_space.spaces["global_robot_states"].shape == (3, 10)
    assert env.observation_space.spaces["agent_active_mask"].shape == (3, 1)


def test_generate_observation_contains_local_peer_semantics_and_shared_state():
    env = make_observation_env()
    env.set_observation_space()

    observation = env.generate_ob(reset=True)

    for key, space in env.observation_space.spaces.items():
        assert observation[key].shape == space.shape
        assert np.isfinite(observation[key]).all()

    np.testing.assert_allclose(
        observation["agent_active_mask"][:, 0],
        [1.0, 0.0, 0.0],
    )
    assert observation["detected_robot_num"][0, 0] == 2
    assert observation["robot_robot_edges"][0, 0, 4] in (0.0, 1.0)
    np.testing.assert_allclose(observation["point_clouds"][2, 0], 3.0)


def test_world_to_robot_uses_observer_heading():
    env = make_observation_env()
    env.robots[0].theta = np.pi / 2

    transformed = env.world_to_robot_for(0, np.array([1.0, 0.0]))

    np.testing.assert_allclose(transformed, [0.0, -1.0], atol=1e-6)
