from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from crowd_sim.multi_robot_core import AgentStatus


def _finite_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


class MAPPOSocialAdapter:
    """Read-only semantic and action adapter for the multi-robot MAPPO env."""

    def __init__(self, env):
        self.env = env
        action_convert = self._env_attr("action_convert")
        if not isinstance(action_convert, Mapping) or not action_convert:
            raise ValueError("action_convert must be a non-empty mapping")

        self.action_index_by_delta = {}
        self._action_deltas = {}
        for raw_index, raw_delta in action_convert.items():
            if isinstance(raw_index, (bool, np.bool_)) or not isinstance(
                raw_index,
                (int, np.integer),
            ):
                raise ValueError("action indices must be integers")
            index = int(raw_index)

            try:
                delta = np.asarray(raw_delta, dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError("each action delta must contain two finite values")
            if delta.shape != (2,) or not np.all(np.isfinite(delta)):
                raise ValueError("each action delta must contain two finite values")

            quantized_delta = np.round(delta, 2)
            if np.any(np.abs(delta - quantized_delta) > 1.0e-9):
                raise ValueError(
                    "action deltas must be representable to two decimal places"
                )
            key = (
                float(quantized_delta[0]),
                float(quantized_delta[1]),
            )
            if key in self.action_index_by_delta:
                raise ValueError("action deltas must be unique after rounding")
            self.action_index_by_delta[key] = index
            self._action_deltas[index] = delta.copy()

        expected_indices = list(range(len(self._action_deltas)))
        if sorted(self._action_deltas) != expected_indices:
            raise ValueError("action indices must be contiguous from zero")
        if (0.0, 0.0) not in self.action_index_by_delta:
            raise ValueError("action table must include a zero-delta action")

        delta_v_values = {key[0] for key in self.action_index_by_delta}
        delta_w_values = {key[1] for key in self.action_index_by_delta}
        expected_deltas = {
            (delta_v, delta_w)
            for delta_v in delta_v_values
            for delta_w in delta_w_values
        }
        if set(self.action_index_by_delta) != expected_deltas:
            raise ValueError("action table must contain every delta combination")

        self.zero_action = self.action_index_by_delta[(0.0, 0.0)]

    def __getattr__(self, name):
        return self._env_attr(name)

    def _env_attr(self, name):
        current = object.__getattribute__(self, "env")
        visited = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            try:
                return getattr(current, name)
            except AttributeError:
                current = getattr(current, "env", None)
        raise AttributeError(name)

    @staticmethod
    def _is_active(status):
        return status == AgentStatus.ACTIVE or status == AgentStatus.ACTIVE.value

    def _robots_and_statuses(self):
        robots = list(self._env_attr("robots"))
        statuses = list(self._env_attr("robot_status"))
        if len(statuses) != len(robots):
            raise ValueError("robot_status must have one entry per robot")
        return robots, statuses

    def _robot_bounds(self):
        robot_config = self._env_attr("config").robot
        bounds = (
            _finite_float(robot_config.v_min),
            _finite_float(robot_config.v_max),
            _finite_float(robot_config.w_min),
            _finite_float(robot_config.w_max),
        )
        if bounds[0] > bounds[1] or bounds[2] > bounds[3]:
            raise ValueError("robot velocity bounds are invalid")
        return bounds

    def robot_states(self, world_velocities=None):
        robots, statuses = self._robots_and_statuses()
        if world_velocities is None:
            velocities = [
                [_finite_float(robot.vx), _finite_float(robot.vy)]
                for robot in robots
            ]
        else:
            try:
                velocity_array = np.asarray(world_velocities, dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError("world_velocities must have shape [robot_count, 2]")
            if velocity_array.shape != (len(robots), 2):
                raise ValueError("world_velocities must have shape [robot_count, 2]")
            velocity_array = np.nan_to_num(
                velocity_array,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            velocities = velocity_array.tolist()

        states = []
        for index, (robot, status, velocity) in enumerate(
            zip(robots, statuses, velocities)
        ):
            states.append(
                {
                    "robot_id": index,
                    "index": index,
                    "position": [
                        _finite_float(robot.px),
                        _finite_float(robot.py),
                    ],
                    "velocity": [
                        _finite_float(velocity[0]),
                        _finite_float(velocity[1]),
                    ],
                    "heading": _finite_float(robot.theta),
                    "goal": [
                        _finite_float(robot.gx),
                        _finite_float(robot.gy),
                    ],
                    "radius": max(0.0, _finite_float(robot.radius)),
                    "active": self._is_active(status),
                }
            )
        return states

    def human_states(self):
        states = []
        for human in self._env_attr("humans"):
            if bool(getattr(human, "isObstacle", False)):
                continue
            vx = _finite_float(human.vx)
            vy = _finite_float(human.vy)
            heading = (
                math.atan2(vy, vx)
                if math.hypot(vx, vy) > 1.0e-9
                else _finite_float(human.theta)
            )
            states.append(
                {
                    "human_id": f"{getattr(human, 'uid', None)}:{id(human)}",
                    "position": [
                        _finite_float(human.px),
                        _finite_float(human.py),
                    ],
                    "velocity": [
                        vx,
                        vy,
                    ],
                    "heading": heading,
                    "radius": max(0.0, _finite_float(human.radius)),
                }
            )
        return states

    def _validated_actions(self, actions, robot_count):
        try:
            raw_actions = np.asarray(actions, dtype=object)
        except (TypeError, ValueError):
            raise ValueError("actions must have one index per robot")
        if raw_actions.shape != (robot_count,):
            raise ValueError("actions must have one index per robot")

        validated = np.empty(robot_count, dtype=np.int64)
        for index, raw_action in enumerate(raw_actions):
            if isinstance(raw_action, (bool, np.bool_)) or not isinstance(
                raw_action,
                (int, np.integer),
            ):
                raise ValueError("action indices must be integers")
            try:
                action = int(raw_action)
                validated[index] = action
            except (OverflowError, ValueError):
                raise ValueError("action indices must fit in int64")
            if action not in self._action_deltas:
                raise ValueError(f"illegal action index {action}")
        return validated

    def preview_controls(self, actions):
        robots, statuses = self._robots_and_statuses()
        validated_actions = self._validated_actions(actions, len(robots))
        desired = np.array(
            self._env_attr("desired_velocities"),
            dtype=np.float64,
            copy=True,
        )
        if desired.shape != (len(robots), 2):
            raise ValueError(
                "desired_velocities must have shape [robot_count, 2]"
            )
        desired = np.nan_to_num(desired, nan=0.0, posinf=0.0, neginf=0.0)
        v_min, v_max, w_min, w_max = self._robot_bounds()

        controls = np.zeros((len(robots), 2), dtype=np.float64)
        for index, (status, action) in enumerate(
            zip(statuses, validated_actions)
        ):
            if not self._is_active(status):
                continue
            delta_v, delta_w = self._action_deltas[action]
            controls[index, 0] = np.clip(
                desired[index, 0] + delta_v,
                v_min,
                v_max,
            )
            controls[index, 1] = np.clip(
                desired[index, 1] + delta_w,
                w_min,
                w_max,
            )
        return controls

    def preview_world_velocities(self, actions):
        controls = self.preview_controls(actions)
        robots, statuses = self._robots_and_statuses()
        dt = _finite_float(self._env_attr("time_step"))
        velocities = np.zeros_like(controls)
        for index, (robot, status) in enumerate(zip(robots, statuses)):
            if not self._is_active(status):
                continue
            speed, predicted_w = controls[index]
            heading = _finite_float(robot.theta) + predicted_w * dt
            velocities[index] = [
                speed * math.cos(heading),
                speed * math.sin(heading),
            ]
        return velocities

    def world_velocities_to_controls(self, world_velocities):
        robots, statuses = self._robots_and_statuses()
        velocities = np.asarray(world_velocities, dtype=np.float64)
        if velocities.shape != (len(robots), 2):
            raise ValueError("world_velocities must have shape [robot_count, 2]")
        velocities = np.nan_to_num(velocities, nan=0.0, posinf=0.0, neginf=0.0)
        v_min, v_max, w_min, w_max = self._robot_bounds()
        dt = max(_finite_float(self._env_attr("time_step")), 1.0e-6)

        controls = np.zeros_like(velocities)
        for index, (robot, status) in enumerate(zip(robots, statuses)):
            if not self._is_active(status):
                continue
            velocity = velocities[index]
            speed = float(np.linalg.norm(velocity))
            if speed <= 1.0e-9:
                continue
            heading = _finite_float(robot.theta)
            target = math.atan2(float(velocity[1]), float(velocity[0]))
            delta = math.atan2(math.sin(target - heading), math.cos(target - heading))
            if abs(delta) > math.pi / 2.0 and v_min < 0.0:
                speed = -speed
                delta = math.atan2(
                    math.sin(target + math.pi - heading),
                    math.cos(target + math.pi - heading),
                )
            controls[index, 0] = np.clip(speed, v_min, v_max)
            controls[index, 1] = np.clip(delta / dt, w_min, w_max)
        return controls.astype(np.float32)

    def controls_to_actions(self, controls):
        robots, statuses = self._robots_and_statuses()
        targets = np.asarray(controls, dtype=np.float64)
        if targets.shape != (len(robots), 2):
            raise ValueError("controls must have shape [robot_count, 2]")
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        v_min, v_max, w_min, w_max = self._robot_bounds()
        desired = np.asarray(self._env_attr("desired_velocities"), dtype=np.float64)
        if desired.shape != targets.shape:
            raise ValueError(
                "desired_velocities must have shape [robot_count, 2]"
            )
        scales = np.asarray(
            [self._minimum_nonzero_delta(0), self._minimum_nonzero_delta(1)]
        )

        actions = np.full(len(robots), self.zero_action, dtype=np.int64)
        for index, status in enumerate(statuses):
            if not self._is_active(status):
                continue
            candidates = []
            for action, delta in self._action_deltas.items():
                predicted = np.clip(
                    desired[index] + delta,
                    [v_min, w_min],
                    [v_max, w_max],
                )
                error = float(np.sum(((predicted - targets[index]) / scales) ** 2))
                magnitude = float(np.sum(np.abs(delta) / scales))
                candidates.append((error, magnitude, action))
            actions[index] = min(candidates)[-1]
        return actions

    def braking_actions(self):
        robots, statuses = self._robots_and_statuses()
        desired = np.array(
            self._env_attr("desired_velocities"),
            dtype=np.float64,
            copy=True,
        )
        if desired.shape != (len(robots), 2):
            raise ValueError(
                "desired_velocities must have shape [robot_count, 2]"
        )
        desired = np.nan_to_num(desired, nan=0.0, posinf=0.0, neginf=0.0)
        v_min, v_max, w_min, w_max = self._robot_bounds()
        linear_step = self._minimum_nonzero_delta(axis=0)
        angular_step = self._minimum_nonzero_delta(axis=1)
        scales = np.asarray([linear_step, angular_step], dtype=np.float64)

        actions = np.full(len(robots), self.zero_action, dtype=np.int64)
        for index, status in enumerate(statuses):
            if not self._is_active(status):
                continue
            current = desired[index]
            candidates = []
            for action, (delta_v, delta_w) in self._action_deltas.items():
                predicted = np.asarray(
                    [
                        np.clip(current[0] + delta_v, v_min, v_max),
                        np.clip(current[1] + delta_w, w_min, w_max),
                    ],
                    dtype=np.float64,
                )
                normalized_drop = self._safe_braking_drop(
                    current,
                    predicted,
                    scales,
                )
                if normalized_drop is None:
                    continue
                normalized_remaining = float(
                    np.sum(np.abs(predicted) / scales)
                )
                normalized_magnitude = (
                    abs(delta_v) / linear_step
                    + abs(delta_w) / angular_step
                )
                candidates.append(
                    (
                        -normalized_drop,
                        normalized_remaining,
                        normalized_magnitude,
                        action,
                    )
                )
            if candidates:
                actions[index] = min(candidates)[-1]
        return actions

    def _minimum_nonzero_delta(self, axis):
        tolerance = 1.0e-12
        magnitudes = [
            abs(delta[axis])
            for delta in self._action_deltas.values()
            if abs(delta[axis]) > tolerance
        ]
        return min(magnitudes) if magnitudes else 1.0

    @staticmethod
    def _safe_braking_drop(current, predicted, scales):
        tolerance = 1.0e-12
        for before, after in zip(current, predicted):
            if before * after < -tolerance:
                return None
            decrease = abs(before) - abs(after)
            if decrease < -tolerance:
                return None
        normalized_drop = float(
            np.sum((np.abs(current) - np.abs(predicted)) / scales)
        )
        return normalized_drop if normalized_drop > tolerance else None
