import gym
import numpy as np

from crowd_sim.envs.crowd_sim_tb2_mappo import CrowdSim3DTBMAPPO
from crowd_sim.multi_robot_core import AgentStatus


class CrowdSim3DTbObsMAPPO(CrowdSim3DTBMAPPO):
    """Three-robot MAPPO environment with HEIGHT-compatible local observations."""

    def set_observation_space(self):
        human_feature_size = 4 if self.config.ob_space.add_human_vel else 2
        spaces = {
            "robot_node": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.robot_num, 1, 5),
                dtype=np.float32,
            ),
            "temporal_edges": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.robot_num, 1, 2),
                dtype=np.float32,
            ),
            "spatial_edges": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.robot_num, max(1, self.max_human_num), human_feature_size),
                dtype=np.float32,
            ),
            "detected_human_num": gym.spaces.Box(
                low=0,
                high=max(1, self.max_human_num),
                shape=(self.robot_num, 1),
                dtype=np.float32,
            ),
            "robot_robot_edges": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.robot_num, self.robot_num - 1, 5),
                dtype=np.float32,
            ),
            "detected_robot_num": gym.spaces.Box(
                low=0,
                high=self.robot_num - 1,
                shape=(self.robot_num, 1),
                dtype=np.float32,
            ),
            "point_clouds": gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.robot_num, 1, self.ray_num),
                dtype=np.float32,
            ),
            "global_robot_states": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.robot_num, 10),
                dtype=np.float32,
            ),
            "agent_active_mask": gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.robot_num, 1),
                dtype=np.float32,
            ),
        }
        self.observation_space = gym.spaces.Dict(spaces)

    def set_action_space(self):
        if self.config.env.action_space == "continuous":
            self.action_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.robot_num, 2),
                dtype=np.float32,
            )
        elif self.config.env.action_space == "discrete":
            self.action_convert = {
                0: [0.05, 0.1],
                1: [0.05, 0.0],
                2: [0.05, -0.1],
                3: [0.0, 0.1],
                4: [0.0, 0.0],
                5: [0.0, -0.1],
                6: [-0.05, 0.1],
                7: [-0.05, 0.0],
                8: [-0.05, -0.1],
            }
            self.action_space = gym.spaces.MultiDiscrete(
                [len(self.action_convert)] * self.robot_num
            )
        else:
            raise ValueError("unknown action space {}".format(self.config.env.action_space))

    def world_to_robot_for(self, robot_id, vector):
        robot = self.robots[robot_id]
        vector = np.asarray(vector, dtype=np.float32)
        cos_theta = np.cos(robot.theta)
        sin_theta = np.sin(robot.theta)
        rotation = np.array(
            [[cos_theta, sin_theta], [-sin_theta, cos_theta]],
            dtype=np.float32,
        )
        return rotation.dot(vector)

    @staticmethod
    def _status_is(status, expected):
        return status == expected

    def _is_visible(self, observer, target):
        relative = np.array(
            [target.px - observer.px, target.py - observer.py],
            dtype=np.float32,
        )
        distance = np.linalg.norm(relative)
        if distance > observer.sensor_range:
            return False
        if observer.FOV >= 2 * np.pi - 1e-6 or distance <= 1e-8:
            return True
        relative_angle = np.arctan2(relative[1], relative[0])
        angular_error = np.arctan2(
            np.sin(relative_angle - observer.theta),
            np.cos(relative_angle - observer.theta),
        )
        return abs(angular_error) <= observer.FOV / 2

    def _robot_node(self, robot):
        velocity = float(getattr(robot, "v", np.linalg.norm([robot.vx, robot.vy])))
        angular_velocity = float(getattr(robot, "w", 0.0))
        return np.array(
            [
                robot.gx - robot.px,
                robot.gy - robot.py,
                robot.theta,
                velocity,
                angular_velocity,
            ],
            dtype=np.float32,
        )

    def _human_edges(self, robot_id):
        observer = self.robots[robot_id]
        feature_size = 4 if self.config.ob_space.add_human_vel else 2
        edges = []
        for human in self.humans:
            if not self._is_visible(observer, human):
                continue
            relative_position = self.world_to_robot_for(
                robot_id,
                np.array([human.px - observer.px, human.py - observer.py]),
            )
            feature = list(relative_position)
            if feature_size == 4:
                relative_velocity = self.world_to_robot_for(
                    robot_id,
                    np.array([human.vx - observer.vx, human.vy - observer.vy]),
                )
                feature.extend(relative_velocity)
            edges.append(np.asarray(feature, dtype=np.float32))

        edges.sort(key=lambda value: np.linalg.norm(value[:2]))
        padded = np.full(
            (max(1, self.max_human_num), feature_size),
            15.0,
            dtype=np.float32,
        )
        visible_count = min(len(edges), len(padded))
        if visible_count:
            padded[:visible_count] = np.stack(edges[:visible_count])
        return padded, max(1, visible_count)

    def _robot_edges(self, robot_id):
        observer = self.robots[robot_id]
        edges = []
        for other_id, other in enumerate(self.robots):
            if other_id == robot_id or not self._is_visible(observer, other):
                continue
            relative_position = self.world_to_robot_for(
                robot_id,
                np.array([other.px - observer.px, other.py - observer.py]),
            )
            relative_velocity = self.world_to_robot_for(
                robot_id,
                np.array([other.vx - observer.vx, other.vy - observer.vy]),
            )
            is_active = float(
                self._status_is(self.robot_status[other_id], AgentStatus.ACTIVE)
            )
            edges.append(
                np.concatenate(
                    [relative_position, relative_velocity, [is_active]]
                ).astype(np.float32)
            )

        edges.sort(key=lambda value: np.linalg.norm(value[:2]))
        padded = np.full((self.robot_num - 1, 5), 15.0, dtype=np.float32)
        padded[:, 2:4] = 0.0
        padded[:, 4] = 0.0
        visible_count = min(len(edges), len(padded))
        if visible_count:
            padded[:visible_count] = np.stack(edges[:visible_count])
        return padded, visible_count

    def _global_robot_states(self):
        position_scale = max(float(self.arena_size), 1.0)
        speed_scale = max(float(self.config.robot.v_max), 1e-6)
        states = np.zeros((self.robot_num, 10), dtype=np.float32)
        for robot_id, robot in enumerate(self.robots):
            status = self.robot_status[robot_id]
            states[robot_id] = [
                robot.px / position_scale,
                robot.py / position_scale,
                robot.vx / speed_scale,
                robot.vy / speed_scale,
                robot.gx / position_scale,
                robot.gy / position_scale,
                robot.theta / np.pi,
                float(self._status_is(status, AgentStatus.ACTIVE)),
                float(self._status_is(status, AgentStatus.REACHED)),
                float(self._status_is(status, AgentStatus.COLLIDED)),
            ]
        return states

    def generate_ob(self, reset):
        del reset
        robot_nodes = np.zeros((self.robot_num, 1, 5), dtype=np.float32)
        temporal_edges = np.zeros((self.robot_num, 1, 2), dtype=np.float32)
        spatial_edges = []
        detected_human_num = np.zeros((self.robot_num, 1), dtype=np.float32)
        robot_robot_edges = []
        detected_robot_num = np.zeros((self.robot_num, 1), dtype=np.float32)
        point_clouds = np.zeros(
            (self.robot_num, 1, self.ray_num),
            dtype=np.float32,
        )

        for robot_id, robot in enumerate(self.robots):
            robot_nodes[robot_id, 0] = self._robot_node(robot)
            temporal_edges[robot_id, 0] = [robot.vx, robot.vy]
            human_edges, human_count = self._human_edges(robot_id)
            peer_edges, peer_count = self._robot_edges(robot_id)
            spatial_edges.append(human_edges)
            robot_robot_edges.append(peer_edges)
            detected_human_num[robot_id, 0] = human_count
            detected_robot_num[robot_id, 0] = peer_count
            point_clouds[robot_id, 0] = self.ray_test_for_robot(
                robot_id,
                include_humans=self.config.ob_space.lidar_pc_include_humans
                if hasattr(self.config.ob_space, "lidar_pc_include_humans")
                else False,
            )

        active_mask = np.array(
            [
                [float(self._status_is(status, AgentStatus.ACTIVE))]
                for status in self.robot_status
            ],
            dtype=np.float32,
        )
        return {
            "robot_node": robot_nodes,
            "temporal_edges": temporal_edges,
            "spatial_edges": np.stack(spatial_edges).astype(np.float32),
            "detected_human_num": detected_human_num,
            "robot_robot_edges": np.stack(robot_robot_edges).astype(np.float32),
            "detected_robot_num": detected_robot_num,
            "point_clouds": point_clouds,
            "global_robot_states": self._global_robot_states(),
            "agent_active_mask": active_mask,
        }
