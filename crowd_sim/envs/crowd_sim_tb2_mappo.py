import copy
from collections import deque

import numpy as np
import pybullet as p
from numpy.linalg import norm

from crowd_sim.envs.crowd_sim_tb2 import CrowdSim3DTB
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.multi_robot_core import (
    AgentStatus,
    build_agent_masks,
    circle_clearance,
    count_crossing_pairs,
    mix_active_rewards,
)


class CrowdSim3DTBMAPPO(CrowdSim3DTB):
    """Three-robot PyBullet base environment for MAPPO."""

    def __init__(self):
        super().__init__()
        self.robot_num = 3
        self.robots = []
        self.robot_status = np.array([AgentStatus.ACTIVE] * self.robot_num, dtype=object)
        self.desired_velocities = np.zeros((self.robot_num, 2), dtype=np.float32)
        self.robot_potentials = np.zeros(self.robot_num, dtype=np.float32)
        self.robot_uids = []
        self.goal_uids = []
        self.human_timeout_len = 10
        self.capture_pybullet_keyframes = False

    def generate_circle_crossing_human(self, static=False):
        human = super().generate_circle_crossing_human(static=static)
        human.recent_speeds = deque(maxlen=self.human_timeout_len * 2)
        human.recent_speeds.append(norm([human.vx, human.vy]))
        return human

    def configure(self, config):
        super().configure(config)
        self.robot_num = config.mappo.robot_num
        if self.robot_num != 3:
            raise ValueError("CrowdSim3DTBMAPPO currently requires exactly 3 robots")

        first_robot = self.robot
        self.robots = [first_robot]
        for _ in range(1, self.robot_num):
            self.robots.append(Robot(config, "robot"))
        self.robot = self.robots[0]
        self.robot_status = np.array(
            [AgentStatus.ACTIVE] * self.robot_num,
            dtype=object,
        )
        self.desired_velocities = np.zeros((self.robot_num, 2), dtype=np.float32)
        self.robot_potentials = np.zeros(self.robot_num, dtype=np.float32)
        self.set_observation_space()
        self.set_action_space()

    def apply_terminal_events(self, reached, collision):
        reached = np.asarray(reached, dtype=bool)
        collision = np.asarray(collision, dtype=bool)
        expected_shape = (self.robot_num,)
        if reached.shape != expected_shape or collision.shape != expected_shape:
            raise ValueError("terminal event arrays must match robot_num")

        for robot_id in range(self.robot_num):
            if self.robot_status[robot_id] != AgentStatus.ACTIVE:
                continue
            if reached[robot_id]:
                self.robot_status[robot_id] = AgentStatus.REACHED
            elif collision[robot_id]:
                self.robot_status[robot_id] = AgentStatus.COLLIDED
        self.force_inactive_stop()

    def force_inactive_stop(self):
        for robot_id, status in enumerate(self.robot_status):
            if status != AgentStatus.ACTIVE:
                self.desired_velocities[robot_id] = 0.0

    def is_environment_done(self, timeout=False):
        if timeout:
            return True
        return all(status != AgentStatus.ACTIVE for status in self.robot_status)

    def _position_clear(self, position, radius, other_positions):
        if self.add_static_obs and self.circle_in_obstacles(
            position[0],
            position[1],
            radius + self.config.mappo.spawn_clearance,
        ):
            return False
        for human in self.humans:
            if circle_clearance(
                position,
                radius,
                (human.px, human.py),
                human.radius,
            ) < self.config.mappo.spawn_clearance:
                return False
        for other_position, other_radius in other_positions:
            if (
                circle_clearance(
                    position,
                    radius,
                    other_position,
                    other_radius,
                )
                < self.config.mappo.spawn_clearance
            ):
                return False
        return True

    @staticmethod
    def cross_pose_clear(
        start,
        goal,
        previous_starts,
        previous_goals,
        minimum_separation,
    ):
        return all(
            np.linalg.norm(start - previous_goal) >= minimum_separation
            for previous_goal in previous_goals
        ) and all(
            np.linalg.norm(goal - previous_start) >= minimum_separation
            for previous_start in previous_starts
        )

    def sample_robot_start_goals(self):
        for _ in range(self.config.mappo.max_spawn_attempts):
            starts = []
            goals = []
            headings = []
            valid = True
            for robot_id, robot in enumerate(self.robots):
                pose_found = False
                for _ in range(100):
                    if self.config.env.scenario == "circle_crossing":
                        start = np.array(
                            [
                                np.random.uniform(*self.config.robot.initX_range),
                                np.random.uniform(*self.config.robot.initY_range),
                            ],
                            dtype=np.float32,
                        )
                        goal = np.array(
                            [
                                np.random.uniform(*self.config.robot.goalX_range),
                                np.random.uniform(*self.config.robot.goalY_range),
                            ],
                            dtype=np.float32,
                        )
                    else:
                        route = self.config.robot.routes[
                            np.random.choice(len(self.config.robot.routes))
                        ]
                        start_region = self.config.robot.regions[route[0]]
                        goal_region = self.config.robot.regions[route[1]]
                        start = np.array(
                            [
                                np.random.uniform(start_region[0], start_region[1]),
                                np.random.uniform(start_region[2], start_region[3]),
                            ],
                            dtype=np.float32,
                        )
                        goal = np.array(
                            [
                                np.random.uniform(goal_region[0], goal_region[1]),
                                np.random.uniform(goal_region[2], goal_region[3]),
                            ],
                            dtype=np.float32,
                        )

                    goal_distance = np.linalg.norm(start - goal)
                    if not (
                        self.config.robot.min_goal_dist
                        <= goal_distance
                        <= self.config.robot.max_goal_dist
                    ):
                        continue
                    if not self._position_clear(
                        start,
                        robot.radius,
                        [
                            (value, self.robots[index].radius)
                            for index, value in enumerate(starts)
                        ],
                    ):
                        continue
                    if not self._position_clear(
                        goal,
                        robot.radius,
                        [
                            (value, self.robots[index].radius)
                            for index, value in enumerate(goals)
                        ],
                    ):
                        continue
                    if any(
                        np.linalg.norm(goal - previous_goal)
                        < self.config.mappo.goal_separation
                        for previous_goal in goals
                    ):
                        continue
                    if not self.cross_pose_clear(
                        start,
                        goal,
                        starts,
                        goals,
                        self.config.mappo.goal_separation,
                    ):
                        continue
                    pose_found = True
                    break
                if not pose_found:
                    valid = False
                    break
                starts.append(start)
                goals.append(goal)
                headings.append(float(np.random.uniform(0.0, 2.0 * np.pi)))

            if not valid or len(starts) != self.robot_num:
                continue
            starts = np.stack(starts)
            goals = np.stack(goals)
            if (
                count_crossing_pairs(
                    starts,
                    goals,
                    self.config.mappo.crossing_distance,
                )
                < self.config.mappo.min_crossing_pairs
            ):
                continue

            for robot, start, goal, heading in zip(
                self.robots,
                starts,
                goals,
                headings,
            ):
                robot.set(
                    float(start[0]),
                    float(start[1]),
                    float(goal[0]),
                    float(goal[1]),
                    0.0,
                    0.0,
                    heading,
                )
                robot.v = 0.0
                robot.w = 0.0
            return
        raise RuntimeError(
            "failed to sample a valid three-robot crossing scenario after {} attempts".format(
                self.config.mappo.max_spawn_attempts
            )
        )

    def _ensure_robot_bodies(self):
        if self.robot_uids:
            return
        self.robot_uids = [self.robots[0].uid]
        for robot_id in range(1, self.robot_num):
            uid = self._p.loadURDF(
                "crowd_sim/pybullet/media/turtlebot2/turtlebot.urdf",
                [20 + robot_id, 20, 0],
            )
            self.robots[robot_id].uid = uid
            self.robot_uids.append(uid)

    def _reset_robot_bodies(self):
        for robot in self.robots:
            self._p.resetBasePositionAndOrientation(
                robot.uid,
                [robot.px, robot.py, 0.0],
                self._p.getQuaternionFromEuler([0.0, 0.0, robot.theta]),
            )
            self._p.resetBaseVelocity(
                robot.uid,
                linearVelocity=[0.0, 0.0, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
            )

    def create_scenario(self, phase="train", test_case=None):
        super().create_scenario(phase=phase, test_case=test_case)
        self._ensure_robot_bodies()
        self.sample_robot_start_goals()
        self.robot = self.robots[0]
        self.robot_status = np.array(
            [AgentStatus.ACTIVE] * self.robot_num,
            dtype=object,
        )
        self.desired_velocities = np.zeros((self.robot_num, 2), dtype=np.float32)
        self.robot_potentials = np.array(
            [
                -norm(
                    np.array([robot.px, robot.py])
                    - np.array([robot.gx, robot.gy])
                )
                for robot in self.robots
            ],
            dtype=np.float32,
        )
        self._reset_robot_bodies()
        self._p.stepSimulation()

    def create_goal_objects(self):
        colors = (
            [1.0, 0.84, 0.0, 1.0],
            [0.0, 0.75, 1.0, 1.0],
            [0.75, 0.2, 1.0, 1.0],
        )
        if not self.goal_uids:
            for color in colors:
                visual_id = self._p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.25,
                    rgbaColor=color,
                )
                self.goal_uids.append(
                    self._p.createMultiBody(
                        baseMass=0,
                        baseCollisionShapeIndex=-1,
                        baseVisualShapeIndex=visual_id,
                        basePosition=[20.0, 20.0, 2.0],
                        baseOrientation=self._p.getQuaternionFromEuler(
                            [0.0, 0.0, 0.0]
                        ),
                    )
                )
        for goal_uid, robot in zip(self.goal_uids, self.robots):
            self._p.resetBasePositionAndOrientation(
                goal_uid,
                [robot.gx, robot.gy, 2.0],
                self._p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            )

    def reset(self, phase="train", test_case=None):
        if getattr(self, "used_human_uids", None):
            self._release_episode_bodies()
        self.create_scenario(phase=phase, test_case=test_case)
        self.create_goal_objects()
        return self.generate_ob(reset=True)

    def _human_can_see(self, human, target):
        relative = np.array([target.px - human.px, target.py - human.py])
        if np.linalg.norm(relative) <= 1e-8 or human.FOV >= 2 * np.pi - 1e-6:
            return True
        target_angle = np.arctan2(relative[1], relative[0])
        angular_error = np.arctan2(
            np.sin(target_angle - human.theta),
            np.cos(target_angle - human.theta),
        )
        return abs(angular_error) <= human.FOV / 2

    def get_human_actions(self):
        human_actions = []
        for human_id, human in enumerate(self.humans):
            if human.isObstacle:
                human_actions.append(ActionXY(0.0, 0.0))
                continue
            observation = []
            for other_id, other_human in enumerate(self.humans):
                if other_id == human_id:
                    continue
                if self._human_can_see(human, other_human):
                    observation.append(other_human.get_observable_state())
                else:
                    observation.append(self.dummy_human.get_observable_state())
            if self.config.robot.visible and human.react_to_robot:
                for robot in self.robots:
                    if self._human_can_see(human, robot):
                        observation.append(robot.get_observable_state())
                    else:
                        observation.append(self.dummy_robot.get_observable_state())
            human_actions.append(human.act(observation))
        return human_actions

    def ray_test_for_robot(self, robot_id, include_humans=False):
        robot = self.robots[robot_id]
        original_human_poses = []
        if not include_humans:
            for human in self.humans:
                original_human_poses.append((human.uid, human.px, human.py))
                self._p.resetBasePositionAndOrientation(
                    human.uid,
                    [30.0, 30.0, self.config.humans.height / 2.0],
                    self._p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                )

        ray_from = np.repeat(
            np.array([[robot.px, robot.py, self.lidar_height]], dtype=np.float32),
            self.ray_num,
            axis=0,
        )
        ray_vectors = np.stack(
            [
                robot.sensor_range * np.cos(robot.theta + self.ray_angles),
                robot.sensor_range * np.sin(robot.theta + self.ray_angles),
                np.zeros(self.ray_num),
            ],
            axis=1,
        )
        results = self._p.rayTestBatch(ray_from, ray_from + ray_vectors)
        distances = np.array(
            [result[2] * robot.sensor_range for result in results],
            dtype=np.float32,
        )

        for uid, px, py in original_human_poses:
            self._p.resetBasePositionAndOrientation(
                uid,
                [px, py, self.config.humans.height / 2.0],
                self._p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            )
        return distances

    def _wheel_targets(self, robot_id, action):
        if self.robot_status[robot_id] != AgentStatus.ACTIVE:
            self.desired_velocities[robot_id] = 0.0
            return 0.0, 0.0

        action_array = np.asarray(action)
        direct_control = action_array.shape == (2,) and not np.issubdtype(
            action_array.dtype,
            np.integer,
        )

        if self.config.env.action_space == "continuous" or direct_control:
            desired_v, desired_w = action
            self.desired_velocities[robot_id, 0] = np.clip(
                desired_v,
                self.config.robot.v_min,
                self.config.robot.v_max,
            )
            self.desired_velocities[robot_id, 1] = np.clip(
                desired_w,
                self.config.robot.w_min,
                self.config.robot.w_max,
            )
        else:
            action_index = int(np.asarray(action).reshape(-1)[0])
            delta_v, delta_w = self.action_convert[action_index]
            self.desired_velocities[robot_id, 0] = np.clip(
                self.desired_velocities[robot_id, 0] + delta_v,
                self.config.robot.v_min,
                self.config.robot.v_max,
            )
            self.desired_velocities[robot_id, 1] = np.clip(
                self.desired_velocities[robot_id, 1] + delta_w,
                self.config.robot.w_min,
                self.config.robot.w_max,
            )

        desired_v, desired_w = self.desired_velocities[robot_id]
        left = (2.0 * desired_v - 0.23 * desired_w) / (2.0 * 0.035)
        right = (2.0 * desired_v + 0.23 * desired_w) / (2.0 * 0.035)
        if self.step_counter < 2:
            return 0.0, 0.0
        noise = np.random.normal(0.0, 0.15, size=2)
        return (
            float(np.clip(left, -11.5, 11.5) + noise[0]),
            float(np.clip(right, -11.5, 11.5) + noise[1]),
        )

    def _set_robot_motor_targets(self, actions):
        actions = np.asarray(actions)
        if actions.shape[0] != self.robot_num:
            raise ValueError("actions must have one entry per robot")
        for robot_id, action in enumerate(actions):
            left, right = self._wheel_targets(robot_id, action)
            robot = self.robots[robot_id]
            self._p.setJointMotorControl2(
                robot.uid,
                0,
                p.VELOCITY_CONTROL,
                targetVelocity=left,
                force=10,
            )
            self._p.setJointMotorControl2(
                robot.uid,
                1,
                p.VELOCITY_CONTROL,
                targetVelocity=right,
                force=10,
            )

    def _refresh_robot_states(self):
        for robot in self.robots:
            position, orientation = self._p.getBasePositionAndOrientation(robot.uid)
            yaw = self._p.getEulerFromQuaternion(orientation)[2]
            linear_velocity, angular_velocity = self._p.getBaseVelocity(robot.uid)
            robot.set(
                position[0],
                position[1],
                robot.gx,
                robot.gy,
                linear_velocity[0],
                linear_velocity[1],
                yaw,
            )
            robot.w = angular_velocity[2]
            robot.v = float(np.linalg.norm(linear_velocity[:2]))
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            if np.dot(forward, np.asarray(linear_velocity[:2])) < 0:
                robot.v = -robot.v

    def _step_humans(self, human_actions):
        for human_id, human_action in enumerate(human_actions):
            human = self.humans[human_id]
            if human.isObstacle and human.isObstacle_period == np.inf:
                continue
            human.step(human_action)
            self.cur_human_states[human_id] = [
                human.px,
                human.py,
                human.radius,
            ]
            self._p.resetBasePositionAndOrientation(
                human.uid,
                [human.px, human.py, self.config.humans.height / 2.0],
                self._p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            )

    def _update_human_lifecycle(self):
        if self.config.sim.change_human_num_in_episode:
            self.change_human_num_periodically()

        if self.random_goal_changing and np.isclose(
            np.mod(self.global_time, 5.0),
            0.0,
            atol=1e-6,
        ):
            self.update_human_goals_randomly()

        if self.end_goal_changing:
            for human_id, human in list(enumerate(self.humans)):
                if human.isObstacle:
                    continue
                reach_distance = (
                    human.radius
                    if self.config.env.scenario == "circle_crossing"
                    else human.radius * 2.0
                )
                if norm((human.gx - human.px, human.gy - human.py)) >= reach_distance:
                    continue

                if self.robot.kinematics == "holonomic":
                    current_uid = human.uid
                    replacement = self.generate_circle_crossing_human()
                    replacement.id = human_id
                    replacement.uid = current_uid
                    self.humans[human_id] = replacement
                    self._p.resetBasePositionAndOrientation(
                        current_uid,
                        [
                            replacement.px,
                            replacement.py,
                            self.config.humans.height / 2.0,
                        ],
                        self._p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                    )
                else:
                    should_remove = self.update_human_goal(human)
                    if should_remove:
                        if (
                            self.config.env.scenario == "csl_workspace"
                            and self.config.env.mode == "sim2real"
                        ):
                            human.isObstacle = True
                        else:
                            self.remove_human(human_id)
                            self.add_human()

        for human in self.humans:
            if hasattr(human, "recent_speeds"):
                human.recent_speeds.append(norm([human.vx, human.vy]))

        if (
            self.config.env.scenario == "circle_crossing"
            and self.step_counter % self.human_timeout_len == 0
        ):
            for human in self.humans:
                if not hasattr(human, "recent_speeds") or not human.recent_speeds:
                    continue
                if (
                    sum(human.recent_speeds) / len(human.recent_speeds)
                    < 0.1
                ):
                    self.update_human_goal(human)

    def _collision_events(self):
        collision = np.zeros(self.robot_num, dtype=bool)
        collision_with = np.array(["none"] * self.robot_num, dtype=object)
        minimum_human_distance = np.full(self.robot_num, np.inf, dtype=np.float32)
        minimum_robot_distance = np.full(self.robot_num, np.inf, dtype=np.float32)
        minimum_obstacle_distance = np.full(self.robot_num, np.inf, dtype=np.float32)

        for robot_id, robot in enumerate(self.robots):
            for human in self.humans:
                closest = self._p.getClosestPoints(
                    robot.uid,
                    human.uid,
                    distance=1000.0,
                )
                if closest:
                    minimum_human_distance[robot_id] = min(
                        minimum_human_distance[robot_id],
                        closest[0][8],
                    )
                if self._p.getContactPoints(robot.uid, human.uid):
                    collision[robot_id] = True
                    collision_with[robot_id] = "human"
            if not collision[robot_id] and self.add_static_obs:
                for obstacle in self.cur_obstacles:
                    obstacle_uid = int(obstacle[-1])
                    closest = self._p.getClosestPoints(
                        robot.uid,
                        obstacle_uid,
                        distance=1000.0,
                    )
                    if closest:
                        minimum_obstacle_distance[robot_id] = min(
                            minimum_obstacle_distance[robot_id],
                            closest[0][8],
                        )
                    if self._p.getContactPoints(robot.uid, obstacle_uid):
                        collision[robot_id] = True
                        collision_with[robot_id] = "obstacle"
                        break
            if not collision[robot_id] and self.config.sim.borders:
                boundary = (
                    self.arena_size
                    + self.config.sim.human_pos_noise_range
                    - 0.5
                )
                if (
                    abs(robot.px) + robot.radius > boundary
                    or abs(robot.py) + robot.radius > boundary
                ):
                    collision[robot_id] = True
                    collision_with[robot_id] = "wall"

        for first in range(self.robot_num):
            for second in range(first + 1, self.robot_num):
                clearance = circle_clearance(
                    (self.robots[first].px, self.robots[first].py),
                    self.robots[first].radius,
                    (self.robots[second].px, self.robots[second].py),
                    self.robots[second].radius,
                )
                minimum_robot_distance[first] = min(
                    minimum_robot_distance[first],
                    clearance,
                )
                minimum_robot_distance[second] = min(
                    minimum_robot_distance[second],
                    clearance,
                )
                if clearance < 0.0 or self._p.getContactPoints(
                    self.robots[first].uid,
                    self.robots[second].uid,
                ):
                    if self.robot_status[first] == AgentStatus.ACTIVE:
                        collision[first] = True
                        collision_with[first] = "robot"
                    if self.robot_status[second] == AgentStatus.ACTIVE:
                        collision[second] = True
                        collision_with[second] = "robot"
        return (
            collision,
            collision_with,
            minimum_human_distance,
            minimum_robot_distance,
            minimum_obstacle_distance,
        )

    def _individual_rewards(
        self,
        active_at_step_start,
        reached,
        collision,
        minimum_human_distance,
        minimum_robot_distance,
        minimum_obstacle_distance,
    ):
        rewards = np.zeros(self.robot_num, dtype=np.float32)
        for robot_id, robot in enumerate(self.robots):
            if not active_at_step_start[robot_id]:
                continue
            if reached[robot_id]:
                reward = self.success_reward
            elif collision[robot_id]:
                reward = self.collision_penalty
            else:
                current_potential = -norm(
                    np.array([robot.px, robot.py])
                    - np.array([robot.gx, robot.gy])
                )
                reward = self.pot_factor * 2.0 * (
                    current_potential - self.robot_potentials[robot_id]
                )
                reward = np.clip(
                    reward,
                    -self.max_abs_pot_reward,
                    self.max_abs_pot_reward,
                )
                self.robot_potentials[robot_id] = current_potential
                if minimum_human_distance[robot_id] < self.discomfort_dist:
                    reward += (
                        minimum_human_distance[robot_id] - self.discomfort_dist
                    ) * self.discomfort_penalty_factor * self.time_step
                if (
                    minimum_robot_distance[robot_id]
                    < self.config.mappo.robot_discomfort_dist
                ):
                    reward += (
                        minimum_robot_distance[robot_id]
                        - self.config.mappo.robot_discomfort_dist
                    ) * self.config.mappo.robot_discomfort_penalty_factor * self.time_step
                if (
                    minimum_obstacle_distance[robot_id]
                    < self.config.mappo.obstacle_discomfort_dist
                ):
                    reward += (
                        minimum_obstacle_distance[robot_id]
                        - self.config.mappo.obstacle_discomfort_dist
                    ) * self.config.mappo.obstacle_discomfort_penalty_factor * self.time_step

            reward += -self.config.reward.spin_factor * robot.w ** 2
            if robot.v < 0:
                reward += -self.config.reward.back_factor * abs(robot.v)
            reward += self.config.reward.constant_penalty
            rewards[robot_id] = reward
        return rewards

    def _release_episode_bodies(self):
        if hasattr(self, "used_human_uids"):
            for uid in list(self.used_human_uids):
                self._p.resetBasePositionAndOrientation(
                    uid,
                    [20.0, 20.0, 0.0],
                    self._p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                )
            self.free_human_uids.extend(copy.deepcopy(self.used_human_uids))
            self.used_human_uids.clear()
        if hasattr(self, "used_obs_uids"):
            for uid in list(self.used_obs_uids):
                self._p.resetBasePositionAndOrientation(
                    uid,
                    [50.0, 50.0, 0.0],
                    self._p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
                )
            self.free_obs_uids.extend(copy.deepcopy(self.used_obs_uids))
            self.used_obs_uids.clear()

    def capture_topdown_rgb(self):
        view_matrix = self._p.computeViewMatrix(
            cameraEyePosition=[0.0, 0.0, 12.0],
            cameraTargetPosition=[0.0, 0.0, 0.0],
            cameraUpVector=[0.0, 1.0, 0.0],
        )
        projection_matrix = self._p.computeProjectionMatrixFOV(
            fov=float(self.render_fov),
            aspect=float(self.render_img_w) / float(self.render_img_h),
            nearVal=0.05,
            farVal=100.0,
        )
        _, _, pixels, _, _ = self._p.getCameraImage(
            int(self.render_img_w),
            int(self.render_img_h),
            view_matrix,
            projection_matrix,
            shadow=False,
            flags=self._p.ER_NO_SEGMENTATION_MASK,
            renderer=self._p.ER_TINY_RENDERER,
        )
        return np.asarray(pixels, dtype=np.uint8)[:, :, :3].copy()

    def step(self, actions, update=True):
        del update
        previous_status = self.robot_status.copy()
        active_at_step_start = np.array(
            [status == AgentStatus.ACTIVE for status in previous_status],
            dtype=bool,
        )
        human_actions = self.get_human_actions()
        self._set_robot_motor_targets(actions)
        self._step_humans(human_actions)
        self.scene.global_step()
        self._refresh_robot_states()
        self.global_time += self.time_step
        self.step_counter += 1
        self.envStepCounter += 1
        self._update_human_lifecycle()

        (
            collision,
            collision_with,
            minimum_human_distance,
            minimum_robot_distance,
            minimum_obstacle_distance,
        ) = self._collision_events()
        reached = np.array(
            [
                norm(
                    np.array([robot.px, robot.py])
                    - np.array([robot.gx, robot.gy])
                )
                < self.goal_reach_dist
                for robot in self.robots
            ],
            dtype=bool,
        )
        timeout = self.global_time >= self.time_limit - self.time_step
        individual_rewards = self._individual_rewards(
            active_at_step_start,
            reached,
            collision,
            minimum_human_distance,
            minimum_robot_distance,
            minimum_obstacle_distance,
        )
        self.apply_terminal_events(reached, collision)
        masks = build_agent_masks(
            previous_status,
            self.robot_status,
            timeout=timeout,
        )
        rewards, team_reward = mix_active_rewards(
            individual_rewards,
            active_at_step_start,
            self.config.mappo.individual_reward_coef,
            self.config.mappo.team_reward_coef,
        )

        observation = self.generate_ob(reset=False)
        terminal_reason = [status.value for status in self.robot_status]
        if timeout:
            terminal_reason = [
                "timeout" if reason == AgentStatus.ACTIVE.value else reason
                for reason in terminal_reason
            ]
        info = {
            "agent_dones": masks.agent_dones.copy(),
            "active_masks": masks.active_masks.copy(),
            "rnn_masks": masks.rnn_masks.copy(),
            "bad_masks": masks.bad_masks.copy(),
            "individual_rewards": individual_rewards.copy(),
            "team_reward": team_reward,
            "mixed_rewards": rewards.copy(),
            "terminal_reason": terminal_reason,
            "collision_with": collision_with.tolist(),
            "minimum_human_distance": minimum_human_distance.copy(),
            "minimum_robot_distance": minimum_robot_distance.copy(),
            "minimum_obstacle_distance": minimum_obstacle_distance.copy(),
            "robot_positions": np.array(
                [[robot.px, robot.py] for robot in self.robots],
                dtype=np.float32,
            ),
        }
        if self.capture_pybullet_keyframes:
            info["pybullet_frame"] = self.capture_topdown_rgb()
        if masks.env_done:
            self._release_episode_bodies()
        return observation, rewards.reshape(self.robot_num, 1), masks.env_done, info
