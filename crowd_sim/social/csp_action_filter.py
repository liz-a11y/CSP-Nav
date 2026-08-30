from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .csp_metrics import CollectiveSocialPressureMetrics


SCORE_LIMIT = 1.0e30


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _finite_score(value: Any) -> float:
    result = _finite_float(value, SCORE_LIMIT)
    return float(np.clip(result, -SCORE_LIMIT, SCORE_LIMIT))


def _vec2(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        for key in ("position", "velocity", "v_hat", "goal", "pos", "xy", "center"):
            if key in value:
                value = value[key]
                break
        else:
            if "px" in value and "py" in value:
                value = [value["px"], value["py"]]
            elif "vx" in value and "vy" in value:
                value = [value["vx"], value["vy"]]
            elif "gx" in value and "gy" in value:
                value = [value["gx"], value["gy"]]
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        array = np.zeros(0, dtype=np.float64)
    result = np.zeros(2, dtype=np.float64)
    result[: min(2, array.size)] = array[:2]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _section(config: Dict[str, Any]) -> Dict[str, Any]:
    values = {
        "horizon_steps": 1,
        "dt": 0.25,
        "csp_threshold": 1.0,
        "task_cost_weight": 1.0,
        "csp_cost_weight": 0.25,
        "threshold_violation_weight": 5.0,
        "deviation_weight": 0.02,
        "max_action_norm": 1.0,
    }
    values.update(config.get("csp_action_filter", {}) or {})
    return _validate_config(values)


def _validated_number(
    values: Dict[str, Any],
    name: str,
    *,
    positive: bool = False,
) -> float:
    value = values[name]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    if not positive and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_config(values: Dict[str, Any]) -> Dict[str, Any]:
    horizon = values["horizon_steps"]
    if (
        isinstance(horizon, (bool, np.bool_))
        or not isinstance(horizon, (int, np.integer))
        or not 1 <= int(horizon) <= 8
    ):
        raise ValueError("horizon_steps must be an integer between 1 and 8")

    validated = dict(values)
    validated["horizon_steps"] = int(horizon)
    validated["dt"] = _validated_number(values, "dt", positive=True)
    for name in (
        "csp_threshold",
        "task_cost_weight",
        "csp_cost_weight",
        "threshold_violation_weight",
        "deviation_weight",
        "max_action_norm",
    ):
        validated[name] = _validated_number(values, name)
    return validated


class CSPActionFilter:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        self.features = self.config.get("features", {}) or {}
        self.cfg = _section(self.config)
        self.metrics = CollectiveSocialPressureMetrics(self.config)

    @property
    def enabled(self) -> bool:
        return bool(self.features.get("enable_csp_action_filter", True))

    def select_action(
        self,
        candidate_actions: Any,
        robot_states: Iterable[Dict[str, Any]],
        human_states: Iterable[Dict[str, Any]],
        lmte_outputs: Optional[Any] = None,
        nominal_action: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        candidates = self._candidate_array(candidate_actions)
        candidate_count, robot_count, _ = candidates.shape
        input_robots = [] if robot_states is None else list(robot_states)
        if robot_count != len(input_robots):
            raise ValueError(
                "candidate_actions robot dimension "
                f"{robot_count} does not match len(robot_states) "
                f"{len(input_robots)}"
            )

        if not self.enabled or candidate_count == 1:
            return candidates[0].astype(np.float32), self._compact_info(
                used=0,
                candidate_count=candidate_count,
                selected_index=0,
                selected_cvar=0.0,
                selected_score=0.0,
                feasible_count=candidate_count,
            )

        nominal = (
            candidates[0]
            if nominal_action is None
            else self._action_array(nominal_action, robot_count)
        )
        robots = input_robots
        input_humans = [] if human_states is None else list(human_states)
        humans = input_humans
        records = [
            self._score_candidate(
                index,
                action,
                nominal,
                robots,
                humans,
                lmte_outputs,
            )
            for index, action in enumerate(candidates)
        ]

        threshold = self.cfg["csp_threshold"]
        feasible = [
            record
            for record in records
            if record["CSP_scene_CVaR"] <= threshold
        ]
        if feasible:
            selected_record = min(
                feasible,
                key=lambda record: (
                    record["feasible_score"],
                    record["candidate_index"],
                ),
            )
        else:
            selected_record = min(
                records,
                key=lambda record: (
                    record["penalized_score"],
                    record["candidate_index"],
                ),
            )

        selected_index = int(selected_record["candidate_index"])
        info = self._compact_info(
            used=1,
            candidate_count=candidate_count,
            selected_index=selected_index,
            selected_cvar=selected_record["CSP_scene_CVaR"],
            selected_score=(
                selected_record["feasible_score"]
                if feasible
                else selected_record["penalized_score"]
            ),
            feasible_count=len(feasible),
        )
        return candidates[selected_index].astype(np.float32), info

    def _candidate_array(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 3 or array.shape[-1] != 2:
            raise ValueError(
                f"candidate_actions must have shape [K,R,2], got {array.shape}"
            )
        if array.shape[0] < 1 or array.shape[1] < 1:
            raise ValueError("candidate_actions must contain a candidate and robot")

        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        max_norm = self.cfg["max_action_norm"]
        if max_norm == 0.0:
            array = np.zeros_like(array)
        else:
            norms = np.linalg.norm(array, axis=-1, keepdims=True)
            scale = np.minimum(1.0, max_norm / np.maximum(norms, 1.0e-12))
            array = array * scale
        return array

    def _action_array(self, value: Any, robot_count: int) -> np.ndarray:
        action = np.asarray(value, dtype=np.float64)
        if action.shape != (robot_count, 2):
            raise ValueError(
                f"nominal_action must have shape [{robot_count},2], got {action.shape}"
            )
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        max_norm = self.cfg["max_action_norm"]
        if max_norm == 0.0:
            action = np.zeros_like(action)
        else:
            norms = np.linalg.norm(action, axis=-1, keepdims=True)
            scale = np.minimum(1.0, max_norm / np.maximum(norms, 1.0e-12))
            action = action * scale
        return action

    def _score_candidate(
        self,
        index: int,
        action: np.ndarray,
        nominal_action: np.ndarray,
        robot_states: List[Dict[str, Any]],
        human_states: List[Dict[str, Any]],
        lmte_outputs: Optional[Any],
    ) -> Dict[str, float]:
        rolled_robots, rolled_humans = self._rollout_states(
            action,
            robot_states,
            human_states,
            lmte_outputs,
        )
        scene = self.metrics.compute_scene_csp(
            rolled_robots,
            rolled_humans,
            lmte_outputs,
            include_details=False,
        )
        threshold = self.cfg["csp_threshold"]
        csp_cvar = _finite_float(
            scene.get("CSP_scene_CVaR"),
            threshold + 1.0,
        )
        _, task_cost = self._task_progress_cost(robot_states, rolled_robots)
        action_deviation = _finite_float(
            np.mean(np.linalg.norm(action - nominal_action, axis=-1))
        )

        feasible_score = _finite_score(
            self.cfg["task_cost_weight"] * task_cost
            + self.cfg["csp_cost_weight"] * csp_cvar
            + self.cfg["deviation_weight"] * action_deviation
        )
        violation = max(0.0, csp_cvar - threshold)
        penalized_score = _finite_score(
            feasible_score
            + self.cfg["threshold_violation_weight"] * violation
        )
        return {
            "candidate_index": int(index),
            "CSP_scene_CVaR": csp_cvar,
            "feasible_score": feasible_score,
            "penalized_score": penalized_score,
        }

    def _rollout_states(
        self,
        action: np.ndarray,
        robot_states: List[Dict[str, Any]],
        human_states: List[Dict[str, Any]],
        lmte_outputs: Optional[Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        robots = [
            self._minimal_robot_state(state, index)
            for index, state in enumerate(robot_states)
        ]
        humans = [
            self._minimal_human_state(state, index)
            for index, state in enumerate(human_states)
        ]
        dt = self.cfg["dt"]
        horizon = self.cfg["horizon_steps"]

        for _ in range(horizon):
            for index, robot in enumerate(robots):
                if not robot["active"]:
                    self._set_velocity(robot, np.zeros(2, dtype=np.float64))
                    continue
                velocity = action[index]
                position = self._position(robot) + dt * velocity
                self._set_motion(robot, position, velocity)
            for human in humans:
                velocity = self._human_velocity(human, lmte_outputs)
                position = self._position(human) + dt * velocity
                self._set_motion(human, position, velocity)
        return robots, humans

    def _minimal_robot_state(
        self,
        state: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any]:
        position = self._position(state)
        velocity = self._velocity(state)
        active = bool(state.get("active", True))
        if not active:
            velocity = np.zeros(2, dtype=np.float64)
        return {
            "robot_id": state.get("robot_id", state.get("id", index)),
            "px": float(position[0]),
            "py": float(position[1]),
            "vx": float(velocity[0]),
            "vy": float(velocity[1]),
            "position": [float(position[0]), float(position[1])],
            "velocity": [float(velocity[0]), float(velocity[1])],
            "radius": max(0.0, _finite_float(state.get("radius"), 0.2)),
            "active": active,
        }

    def _minimal_human_state(
        self,
        state: Dict[str, Any],
        index: int,
    ) -> Dict[str, Any]:
        position = self._position(state)
        velocity = self._velocity(state)
        minimal = {
            "human_id": state.get(
                "human_id",
                state.get("ped_id", state.get("id", index)),
            ),
            "px": float(position[0]),
            "py": float(position[1]),
            "vx": float(velocity[0]),
            "vy": float(velocity[1]),
            "position": [float(position[0]), float(position[1])],
            "velocity": [float(velocity[0]), float(velocity[1])],
            "radius": max(0.0, _finite_float(state.get("radius"), 0.3)),
        }
        if "heading" in state:
            minimal["heading"] = _finite_float(state["heading"])
        elif "theta" in state:
            minimal["theta"] = _finite_float(state["theta"])
        return minimal

    def _human_velocity(
        self,
        human: Dict[str, Any],
        lmte_outputs: Optional[Any],
    ) -> np.ndarray:
        human_id = human.get("human_id", human.get("ped_id", human.get("id", "")))
        if isinstance(lmte_outputs, dict):
            trend = lmte_outputs.get(human_id, lmte_outputs.get(str(human_id), {}))
            if isinstance(trend, dict) and "v_hat" in trend:
                return _vec2(trend["v_hat"])
        if "velocity" in human:
            return _vec2(human["velocity"])
        return _vec2({"vx": human.get("vx", 0.0), "vy": human.get("vy", 0.0)})

    @staticmethod
    def _position(state: Dict[str, Any]) -> np.ndarray:
        if "position" in state:
            return _vec2(state["position"])
        if "center" in state:
            return _vec2(state["center"])
        return _vec2({"px": state.get("px", 0.0), "py": state.get("py", 0.0)})

    @staticmethod
    def _velocity(state: Dict[str, Any]) -> np.ndarray:
        if "velocity" in state:
            return _vec2(state["velocity"])
        return _vec2({"vx": state.get("vx", 0.0), "vy": state.get("vy", 0.0)})

    @staticmethod
    def _set_motion(
        state: Dict[str, Any],
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        state["position"] = [float(position[0]), float(position[1])]
        state["velocity"] = [float(velocity[0]), float(velocity[1])]
        state["px"], state["py"] = float(position[0]), float(position[1])
        state["vx"], state["vy"] = float(velocity[0]), float(velocity[1])
        if float(np.linalg.norm(velocity)) > 1.0e-12:
            state["heading"] = float(math.atan2(velocity[1], velocity[0]))

    @staticmethod
    def _set_velocity(
        state: Dict[str, Any],
        velocity: np.ndarray,
    ) -> None:
        state["velocity"] = [float(velocity[0]), float(velocity[1])]
        state["vx"], state["vy"] = float(velocity[0]), float(velocity[1])

    def _task_progress_cost(
        self,
        start_robots: List[Dict[str, Any]],
        rolled_robots: List[Dict[str, Any]],
    ) -> Tuple[float, float]:
        progress_values = []
        for before, after in zip(start_robots, rolled_robots):
            if not bool(before.get("active", True)):
                continue
            if "goal" in before:
                goal = _vec2(before["goal"])
            elif "gx" in before and "gy" in before:
                goal = _vec2({"gx": before["gx"], "gy": before["gy"]})
            else:
                continue
            before_distance = _finite_float(np.linalg.norm(self._position(before) - goal))
            after_distance = _finite_float(np.linalg.norm(self._position(after) - goal))
            progress_values.append(before_distance - after_distance)
        if not progress_values:
            return 0.0, 0.0
        progress = _finite_float(np.mean(progress_values))
        return progress, -progress

    @staticmethod
    def _compact_info(
        *,
        used: int,
        candidate_count: int,
        selected_index: int,
        selected_cvar: float,
        selected_score: float,
        feasible_count: int,
    ) -> Dict[str, Any]:
        return {
            "csp_action_filter_used": int(used),
            "candidate_count": int(candidate_count),
            "selected_index": int(selected_index),
            "selected_CSP_scene_CVaR": _finite_float(selected_cvar),
            "selected_score": _finite_score(selected_score),
            "filtered_by_csp_action_filter": int(selected_index != 0),
            "feasible_candidate_count": int(feasible_count),
        }
