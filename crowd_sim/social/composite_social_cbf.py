from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover - fallback path is tested by info fields.
    minimize = None


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _vec2(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        for key in ("position", "velocity", "v_hat", "goal", "center"):
            if key in value:
                value = value[key]
                break
        else:
            if "px" in value and "py" in value:
                value = [value["px"], value["py"]]
            elif "vx" in value and "vy" in value:
                value = [value["vx"], value["vy"]]
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        array = np.zeros(0, dtype=np.float64)
    result = np.zeros(2, dtype=np.float64)
    result[: min(2, array.size)] = array[:2]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    return dict((config or {}).get(name, {}) or {})


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class LMTEAwareCompositeCBFSafetyShield:
    """Last-mile CBF-QP shield over per-robot world velocities."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        self.features = _section(self.config, "features")
        self.cfg = {
            "alpha": 0.5,
            "dt": 0.25,
            "control_limit": 1.0,
            "robot_human_margin": 0.10,
            "robot_margin": 0.10,
            "obstacle_margin": 0.10,
            "wall_margin": 0.10,
            "lmte_axis_margin": 0.05,
            "lmte_uncertainty_margin": 0.25,
            "csp_threshold": 1.0,
            "csp_weight": 1.0,
            "min_h_clip": 20.0,
            "fallback": "brake",
        }
        self.cfg.update(_section(self.config, "social_cbf"))
        self.previous_safe_action: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        return bool(
            self.features.get(
                "enable_cbf_filter",
                self.features.get("enable_social_cbf_filter", False),
            )
        )

    @property
    def use_qp(self) -> bool:
        return bool(self.features.get("use_cbf_qp", True))

    def filter(
        self,
        nominal_action: Any,
        robot_states: Iterable[Dict[str, Any]],
        human_states: Iterable[Dict[str, Any]],
        lmte_outputs: Optional[Any] = None,
        *,
        clearance_context: Optional[Dict[str, Any]] = None,
        fallback_action: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        nominal = self._candidate_array(nominal_action)
        robots = [self._robot_state(state, i) for i, state in enumerate(robot_states or [])]
        humans = [self._human_state(state, i) for i, state in enumerate(human_states or [])]
        if nominal.shape != (len(robots), 2):
            raise ValueError(
                f"nominal_action must have shape [{len(robots)},2], got {nominal.shape}"
            )

        if not self.enabled or not self.use_qp:
            info = self._info(
                used=0,
                nominal=nominal,
                safe=nominal,
                before=self._barrier(robots, humans, lmte_outputs, clearance_context),
                after=self._barrier(robots, humans, lmte_outputs, clearance_context),
                qp_success=False,
                fallback_used=False,
            )
            self.previous_safe_action = nominal.copy()
            return nominal.astype(np.float32), info

        before = self._barrier(robots, humans, lmte_outputs, clearance_context)
        rows = self._linear_constraints(robots, humans, lmte_outputs, clearance_context)
        safe, qp_success = self._solve_qp(nominal, rows)
        fallback_used = False
        if not qp_success:
            safe = self._fallback(nominal, fallback_action)
            fallback_used = True

        after = self._barrier(
            self._rolled_robots(robots, safe),
            self._rolled_humans(humans, lmte_outputs),
            lmte_outputs,
            clearance_context,
        )
        info = self._info(
            used=1,
            nominal=nominal,
            safe=safe,
            before=before,
            after=after,
            alpha=_finite_float(self.cfg.get("alpha"), 0.5),
            qp_success=qp_success,
            fallback_used=fallback_used,
        )
        self.previous_safe_action = safe.copy()
        return safe.astype(np.float32), info

    def select_action(
        self,
        candidate_actions: Any,
        robot_states: Iterable[Dict[str, Any]],
        human_states: Iterable[Dict[str, Any]],
        lmte_outputs: Optional[Any] = None,
        nominal_action: Optional[Any] = None,
        clearance_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        candidates = np.asarray(candidate_actions, dtype=np.float64)
        if candidates.ndim != 3 or candidates.shape[-1] != 2:
            raise ValueError(
                f"candidate_actions must have shape [K,R,2], got {candidates.shape}"
            )
        nominal = candidates[0] if nominal_action is None else nominal_action
        fallback = candidates[1] if len(candidates) > 1 else None
        return self.filter(
            nominal,
            robot_states,
            human_states,
            lmte_outputs,
            clearance_context=clearance_context,
            fallback_action=fallback,
        )

    def _solve_qp(
        self,
        nominal: np.ndarray,
        rows: List[Tuple[np.ndarray, float, str, float]],
    ) -> Tuple[np.ndarray, bool]:
        flat_nominal = nominal.reshape(-1)
        if not rows:
            return nominal.copy(), True
        if minimize is None:
            return nominal.copy(), False

        limit = max(0.0, _finite_float(self.cfg.get("control_limit"), 1.0))
        bounds = [(-limit, limit)] * flat_nominal.size

        def objective(flat):
            diff = flat - flat_nominal
            return 0.5 * float(np.dot(diff, diff))

        def jac(flat):
            return flat - flat_nominal

        constraints = [
            {
                "type": "ineq",
                "fun": lambda flat, row=row, lower=lower: float(np.dot(row, flat) - lower),
                "jac": lambda flat, row=row, lower=lower: row,
            }
            for row, lower, _source, _h in rows
        ]
        result = minimize(
            objective,
            flat_nominal,
            jac=jac,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
            options={"ftol": 1e-6, "maxiter": 50, "disp": False},
        )
        if not bool(result.success) or not np.all(np.isfinite(result.x)):
            return nominal.copy(), False
        return np.asarray(result.x, dtype=np.float64).reshape(nominal.shape), True

    def _linear_constraints(
        self,
        robots: List[Dict[str, Any]],
        humans: List[Dict[str, Any]],
        lmte_outputs: Optional[Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Tuple[np.ndarray, float, str, float]]:
        rows: List[Tuple[np.ndarray, float, str, float]] = []
        dim = 2 * len(robots)
        alpha = max(0.0, _finite_float(self.cfg.get("alpha"), 0.5))

        def add(robot_id: int, grad: np.ndarray, lower: float, source: str, h: float) -> None:
            row = np.zeros(dim, dtype=np.float64)
            row[2 * robot_id : 2 * robot_id + 2] = grad
            rows.append((row, lower, source, h))

        for i, robot in enumerate(robots):
            if not robot["active"]:
                continue
            for human in humans:
                human_velocity = self._human_velocity(human, lmte_outputs)
                margin = self._human_margin(human, lmte_outputs)
                h, grad = self._clearance_grad(
                    robot["position"],
                    human["position"],
                    robot["radius"] + human["radius"] + margin,
                )
                add(i, grad, float(np.dot(grad, human_velocity) - alpha * h), "robot_human", h)
            for h, grad, source in self._lmte_terms(robot, lmte_outputs):
                add(i, grad, -alpha * h, source, h)
            for h, grad, source in self._wall_terms(robot, context):
                add(i, grad, -alpha * h, source, h)
            for h, grad, source in self._obstacle_terms(robot, context):
                add(i, grad, -alpha * h, source, h)

        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                if not robots[i]["active"] and not robots[j]["active"]:
                    continue
                h, grad = self._clearance_grad(
                    robots[i]["position"],
                    robots[j]["position"],
                    robots[i]["radius"] + robots[j]["radius"] + _finite_float(self.cfg["robot_margin"]),
                )
                row = np.zeros(dim, dtype=np.float64)
                row[2 * i : 2 * i + 2] = grad
                row[2 * j : 2 * j + 2] = -grad
                rows.append((row, -alpha * h, "robot", h))

        return rows

    def _fallback(self, nominal: np.ndarray, fallback_action: Optional[Any]) -> np.ndarray:
        if fallback_action is not None and str(self.cfg.get("fallback", "brake")) == "brake":
            fallback = self._candidate_array(fallback_action)
            if fallback.shape == nominal.shape:
                return fallback.copy()
        if self.previous_safe_action is not None and self.previous_safe_action.shape == nominal.shape:
            return self.previous_safe_action.copy()
        return np.zeros_like(nominal)

    def _barrier(
        self,
        robots: List[Dict[str, Any]],
        humans: List[Dict[str, Any]],
        lmte_outputs: Optional[Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        terms: List[Tuple[float, str]] = []
        for robot in robots:
            if not robot["active"]:
                continue
            for human in humans:
                margin = self._human_margin(human, lmte_outputs)
                h, _ = self._clearance_grad(
                    robot["position"],
                    human["position"],
                    robot["radius"] + human["radius"] + margin,
                )
                terms.append((h, "robot_human"))
            terms.extend((h, source) for h, _grad, source in self._lmte_terms(robot, lmte_outputs))
            terms.extend((h, source) for h, _grad, source in self._wall_terms(robot, context))
            terms.extend((h, source) for h, _grad, source in self._obstacle_terms(robot, context))
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                h, _ = self._clearance_grad(
                    robots[i]["position"],
                    robots[j]["position"],
                    robots[i]["radius"] + robots[j]["radius"] + _finite_float(self.cfg["robot_margin"]),
                )
                terms.append((h, "robot"))
        if not terms:
            return {"min_h": 0.0, "source": "none"}
        min_h, source = min(terms, key=lambda item: item[0])
        clip = _finite_float(self.cfg.get("min_h_clip"), 20.0)
        return {"min_h": float(np.clip(min_h, -clip, clip)), "source": source}

    def _rolled_robots(self, robots: List[Dict[str, Any]], action: np.ndarray) -> List[Dict[str, Any]]:
        dt = _finite_float(self.cfg.get("dt"), 0.25)
        rolled = []
        for robot, velocity in zip(robots, action):
            clone = dict(robot)
            clone["position"] = robot["position"] + dt * velocity
            clone["velocity"] = np.asarray(velocity, dtype=np.float64)
            rolled.append(clone)
        return rolled

    def _rolled_humans(
        self,
        humans: List[Dict[str, Any]],
        lmte_outputs: Optional[Any],
    ) -> List[Dict[str, Any]]:
        dt = _finite_float(self.cfg.get("dt"), 0.25)
        rolled = []
        for human in humans:
            velocity = self._human_velocity(human, lmte_outputs)
            clone = dict(human)
            clone["position"] = human["position"] + dt * velocity
            clone["velocity"] = np.asarray(velocity, dtype=np.float64)
            rolled.append(clone)
        return rolled

    def _lmte_terms(
        self,
        robot: Dict[str, Any],
        lmte_outputs: Optional[Any],
    ) -> List[Tuple[float, np.ndarray, str]]:
        terms = []
        for trend in self._lmte_items(lmte_outputs):
            for region in (trend.get("reachable_regions") or {}).values():
                h, grad = self._ellipse_clearance(robot["position"], region)
                terms.append((h - _finite_float(self.cfg["lmte_axis_margin"]), grad, "lmte"))
        return terms

    def _wall_terms(
        self,
        robot: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Tuple[float, np.ndarray, str]]:
        if not context or not bool(context.get("borders", False)):
            return []
        arena = _finite_float(context.get("arena_size"), 0.0)
        if arena <= 0.0:
            return []
        margin = robot["radius"] + _finite_float(self.cfg["wall_margin"])
        x, y = robot["position"]
        return [
            (arena - margin - x, np.array([-1.0, 0.0]), "wall"),
            (x + arena - margin, np.array([1.0, 0.0]), "wall"),
            (arena - margin - y, np.array([0.0, -1.0]), "wall"),
            (y + arena - margin, np.array([0.0, 1.0]), "wall"),
        ]

    def _obstacle_terms(
        self,
        robot: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> List[Tuple[float, np.ndarray, str]]:
        if not context or context.get("obstacles") is None:
            return []
        obstacles = np.asarray(context["obstacles"], dtype=np.float64)
        if obstacles.size == 0:
            return []
        obstacles = obstacles.reshape(-1, obstacles.shape[-1])
        terms = []
        for obstacle in obstacles:
            if obstacle.size < 4:
                continue
            x, y, width, height = obstacle[:4]
            low = np.asarray([x, y], dtype=np.float64)
            high = low + np.asarray([width, height], dtype=np.float64)
            point = robot["position"]
            closest = np.minimum(np.maximum(point, low), high)
            diff = point - closest
            dist = float(np.linalg.norm(diff))
            if dist <= 1e-9:
                distances = np.asarray(
                    [point[0] - low[0], high[0] - point[0], point[1] - low[1], high[1] - point[1]]
                )
                side = int(np.argmin(np.abs(distances)))
                normals = [np.array([-1.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, -1.0]), np.array([0.0, 1.0])]
                grad = normals[side]
                clearance = -abs(float(distances[side]))
            else:
                grad = diff / dist
                clearance = dist
            h = clearance - robot["radius"] - _finite_float(self.cfg["obstacle_margin"])
            terms.append((h, grad, "obstacle"))
        return terms

    def _human_margin(self, human: Dict[str, Any], lmte_outputs: Optional[Any]) -> float:
        margin = _finite_float(self.cfg["robot_human_margin"])
        trend = self._lmte_for_human(lmte_outputs, human.get("human_id"))
        uncertainty = _finite_float(trend.get("uncertainty"), 0.0)
        margin += _finite_float(self.cfg.get("lmte_uncertainty_margin"), 0.25) * uncertainty
        axes = [
            max(_finite_float(region.get("a")), _finite_float(region.get("b")))
            for region in (trend.get("reachable_regions") or {}).values()
            if isinstance(region, dict)
        ]
        if axes:
            margin += _finite_float(self.cfg.get("lmte_axis_margin"), 0.05) * max(axes)
        return margin

    def _human_velocity(self, human: Dict[str, Any], lmte_outputs: Optional[Any]) -> np.ndarray:
        trend = self._lmte_for_human(lmte_outputs, human.get("human_id"))
        if "v_hat" in trend:
            return _vec2(trend["v_hat"])
        return human["velocity"]

    def _lmte_for_human(self, lmte_outputs: Optional[Any], human_id: Any) -> Dict[str, Any]:
        if isinstance(lmte_outputs, dict):
            return lmte_outputs.get(human_id, lmte_outputs.get(str(human_id), {})) or {}
        return {}

    def _lmte_items(self, lmte_outputs: Optional[Any]) -> List[Dict[str, Any]]:
        if isinstance(lmte_outputs, dict):
            return [item for item in lmte_outputs.values() if isinstance(item, dict)]
        return []

    @staticmethod
    def _clearance_grad(point_a: np.ndarray, point_b: np.ndarray, safe_distance: float) -> Tuple[float, np.ndarray]:
        diff = point_a - point_b
        dist = float(np.linalg.norm(diff))
        if dist <= 1e-9:
            return -safe_distance, np.array([1.0, 0.0], dtype=np.float64)
        return dist - safe_distance, diff / dist

    @staticmethod
    def _ellipse_clearance(point: np.ndarray, region: Dict[str, Any]) -> Tuple[float, np.ndarray]:
        center = _vec2(region.get("center", [0.0, 0.0]))
        a = max(1e-6, _finite_float(region.get("a"), 1.0))
        b = max(1e-6, _finite_float(region.get("b"), 1.0))
        theta = _finite_float(region.get("theta"), 0.0)
        c, s = math.cos(theta), math.sin(theta)
        rot_t = np.asarray([[c, s], [-s, c]], dtype=np.float64)
        rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
        local = rot_t.dot(point - center)
        scaled = np.asarray([local[0] / a, local[1] / b], dtype=np.float64)
        norm = float(np.linalg.norm(scaled))
        if norm <= 1e-9:
            return -1.0, rot.dot(np.array([1.0, 0.0], dtype=np.float64))
        grad_local = np.asarray([local[0] / (a * a), local[1] / (b * b)], dtype=np.float64) / norm
        return norm - 1.0, rot.dot(grad_local)

    @staticmethod
    def _candidate_array(value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 2 or array.shape[-1] != 2:
            raise ValueError(f"action must have shape [R,2], got {array.shape}")
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _robot_state(state: Dict[str, Any], index: int) -> Dict[str, Any]:
        return {
            "robot_id": state.get("robot_id", state.get("id", index)),
            "position": _vec2(state),
            "velocity": _vec2(state.get("velocity", state)),
            "heading": _finite_float(state.get("heading", state.get("theta", 0.0))),
            "radius": max(0.0, _finite_float(state.get("radius"), 0.2)),
            "active": bool(state.get("active", True)),
        }

    @staticmethod
    def _human_state(state: Dict[str, Any], index: int) -> Dict[str, Any]:
        return {
            "human_id": state.get("human_id", state.get("ped_id", state.get("id", index))),
            "position": _vec2(state),
            "velocity": _vec2(state.get("velocity", state)),
            "radius": max(0.0, _finite_float(state.get("radius"), 0.3)),
        }

    @staticmethod
    def _info(
        *,
        used: int,
        nominal: np.ndarray,
        safe: np.ndarray,
        before: Dict[str, Any],
        after: Dict[str, Any],
        alpha: float = 0.5,
        qp_success: bool,
        fallback_used: bool,
    ) -> Dict[str, Any]:
        deviation = float(np.mean(np.linalg.norm(safe - nominal, axis=-1))) if nominal.size else 0.0
        intervened = bool(deviation > 1e-5)
        min_before = _finite_float(before.get("min_h"), 0.0)
        min_after = _finite_float(after.get("min_h"), 0.0)
        condition = min_after - min_before + max(0.0, float(alpha)) * min_before
        return {
            "cbf_used": int(used),
            "cbf_intervened": int(intervened),
            "min_h_before": min_before,
            "min_h_after": min_after,
            "cbf_condition": condition,
            "qp_success": int(qp_success),
            "qp_infeasible": int(used and not qp_success),
            "fallback_used": int(fallback_used),
            "action_deviation": deviation,
            "min_h_source": str(before.get("source", "none")),
            "social_cbf_used": int(used),
            "social_cbf_filtered": int(intervened),
            "social_cbf_violation": int(_finite_float(before.get("min_h"), 0.0) < 0.0),
            "social_cbf_min_h": _finite_float(before.get("min_h"), 0.0),
            "social_cbf_next_h": _finite_float(after.get("min_h"), 0.0),
            "social_cbf_condition": condition,
            "social_cbf_min_source": str(before.get("source", "none")),
        }


CompositeSocialCBFFilter = LMTEAwareCompositeCBFSafetyShield