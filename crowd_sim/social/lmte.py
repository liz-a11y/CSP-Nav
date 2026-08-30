from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterable, Optional

import numpy as np


def _vec2(value: Optional[Iterable[float]], default: Optional[np.ndarray] = None) -> np.ndarray:
    if default is None:
        default = np.zeros(2, dtype=np.float64)
    if value is None:
        return np.asarray(default, dtype=np.float64).copy()
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        out = np.asarray(default, dtype=np.float64).copy()
        out[: arr.size] = arr
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(arr[:2], nan=0.0, posinf=0.0, neginf=0.0)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _validate_tau_values(values: Iterable[float], parameter_name: str) -> list[float]:
    try:
        taus = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter_name} must contain numeric values.") from exc
    if not taus:
        raise ValueError(f"{parameter_name} must not be empty.")
    if any(not math.isfinite(tau) or tau <= 0.0 for tau in taus):
        raise ValueError(
            f"Each {parameter_name} value must be finite and greater than zero."
        )
    return taus


@dataclass
class LMTERecord:
    ped_id: Any
    timestamp: Optional[float]
    position: np.ndarray
    velocity: np.ndarray
    valid: bool

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "ped_id": self.ped_id,
            "timestamp": self.timestamp,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "valid": int(self.valid),
        }


class LightweightMotionTrendEstimator:
    """Lightweight short-horizon trend estimator for pedestrian motion.

    The estimator keeps a fixed K-step cache per pedestrian and produces a
    weighted recent velocity, uncertainty, and simple reachable regions.
    """

    def __init__(
        self,
        K: int = 5,
        alpha_decay: float = 0.6,
        base_uncertainty: float = 0.05,
        missing_uncertainty_gain: float = 0.2,
        velocity_variance_gain: float = 0.5,
        position_variance_gain: float = 0.0,
        recent_change_gain: float = 0.25,
        max_uncertainty: float = 2.0,
        tau_list: Optional[Iterable[float]] = None,
        a_base: float = 0.3,
        b_base: float = 0.25,
        rho_uncertainty: float = 1.0,
        rho_side: float = 0.5,
        min_axis: float = 0.1,
        max_axis: float = 3.0,
        min_heading_speed: float = 1e-3,
        default_dt: float = 0.25,
    ) -> None:
        if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K <= 0:
            raise ValueError("K must be a positive integer.")
        self.K = int(K)
        self.alpha_decay = float(alpha_decay)
        self.base_uncertainty = float(base_uncertainty)
        self.missing_uncertainty_gain = float(missing_uncertainty_gain)
        self.velocity_variance_gain = float(velocity_variance_gain)
        self.position_variance_gain = float(position_variance_gain)
        self.recent_change_gain = float(recent_change_gain)
        self.max_uncertainty = float(max_uncertainty)
        self.tau_list = _validate_tau_values(
            tau_list if tau_list is not None else [0.5, 1.0, 1.5, 2.0],
            "tau_list",
        )
        self.a_base = float(a_base)
        self.b_base = float(b_base)
        self.rho_uncertainty = float(rho_uncertainty)
        self.rho_side = float(rho_side)
        self.min_axis = float(min_axis)
        self.max_axis = float(max_axis)
        self.min_heading_speed = float(min_heading_speed)
        self.default_dt = float(default_dt)
        self._validate_parameters()

        self._history: Dict[Any, Deque[LMTERecord]] = defaultdict(lambda: deque(maxlen=self.K))
        self._last_stable_heading: Dict[Any, float] = {}

    def _validate_parameters(self) -> None:
        finite_parameters = {
            "alpha_decay": self.alpha_decay,
            "base_uncertainty": self.base_uncertainty,
            "missing_uncertainty_gain": self.missing_uncertainty_gain,
            "velocity_variance_gain": self.velocity_variance_gain,
            "position_variance_gain": self.position_variance_gain,
            "recent_change_gain": self.recent_change_gain,
            "max_uncertainty": self.max_uncertainty,
            "a_base": self.a_base,
            "b_base": self.b_base,
            "rho_uncertainty": self.rho_uncertainty,
            "rho_side": self.rho_side,
            "min_axis": self.min_axis,
            "max_axis": self.max_axis,
            "min_heading_speed": self.min_heading_speed,
            "default_dt": self.default_dt,
        }
        for name, value in finite_parameters.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not 0.0 < self.alpha_decay <= 1.0:
            raise ValueError("alpha_decay must be greater than zero and at most one.")

        nonnegative_parameters = {
            "base_uncertainty": self.base_uncertainty,
            "missing_uncertainty_gain": self.missing_uncertainty_gain,
            "velocity_variance_gain": self.velocity_variance_gain,
            "position_variance_gain": self.position_variance_gain,
            "recent_change_gain": self.recent_change_gain,
            "max_uncertainty": self.max_uncertainty,
            "a_base": self.a_base,
            "b_base": self.b_base,
            "rho_uncertainty": self.rho_uncertainty,
            "rho_side": self.rho_side,
            "min_heading_speed": self.min_heading_speed,
        }
        for name, value in nonnegative_parameters.items():
            if value < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
        if self.base_uncertainty > self.max_uncertainty:
            raise ValueError("base_uncertainty must not exceed max_uncertainty.")
        if self.default_dt <= 0.0:
            raise ValueError("default_dt must be greater than zero.")
        if self.min_axis <= 0.0:
            raise ValueError("min_axis must be greater than zero.")
        if self.max_axis < self.min_axis:
            raise ValueError("max_axis must be greater than or equal to min_axis.")

    def reset(self, ped_id: Optional[Any] = None) -> None:
        if ped_id is None:
            self._history.clear()
            self._last_stable_heading.clear()
            return
        self._history.pop(ped_id, None)
        self._last_stable_heading.pop(ped_id, None)

    def update(
        self,
        ped_id: Any,
        position: Optional[Iterable[float]],
        velocity: Optional[Iterable[float]] = None,
        timestamp: Optional[float] = None,
        valid: bool = True,
    ) -> Dict[str, Any]:
        history = self._history[ped_id]
        valid = bool(valid)

        if valid:
            fallback_pos = self._last_position(history)
            pos = _vec2(position, fallback_pos)
            vel = self._resolve_velocity(history, pos, velocity, timestamp)
        else:
            pos = self._last_position(history)
            vel = np.zeros(2, dtype=np.float64)

        record = LMTERecord(
            ped_id=ped_id,
            timestamp=None if timestamp is None else _safe_float(timestamp),
            position=pos,
            velocity=vel,
            valid=valid,
        )
        history.append(record)
        return self.estimate(ped_id)

    def estimate(self, ped_id: Any) -> Dict[str, Any]:
        history = list(self._history.get(ped_id, []))
        if not history:
            return self._empty_estimate(ped_id)

        valid_records = [item for item in history if item.valid]
        valid_count = len(valid_records)
        missing_ratio = float(np.clip(1.0 - valid_count / float(self.K), 0.0, 1.0))

        if valid_records:
            weighted_vx = 0.0
            weighted_vy = 0.0
            weight_sum = 0.0
            for age, record in enumerate(reversed(history)):
                if not record.valid:
                    continue
                weight = self.alpha_decay**age
                weighted_vx += weight * float(record.velocity[0])
                weighted_vy += weight * float(record.velocity[1])
                weight_sum += weight
            if weight_sum > 0.0:
                vx_hat = weighted_vx / weight_sum
                vy_hat = weighted_vy / weight_sum
            else:
                vx_hat = float(valid_records[-1].velocity[0])
                vy_hat = float(valid_records[-1].velocity[1])
        else:
            vx_hat = 0.0
            vy_hat = 0.0

        vx_hat = vx_hat if math.isfinite(vx_hat) else 0.0
        vy_hat = vy_hat if math.isfinite(vy_hat) else 0.0
        speed = math.hypot(vx_hat, vy_hat)
        if speed > self.min_heading_speed:
            heading = float(math.atan2(vy_hat, vx_hat))
            self._last_stable_heading[ped_id] = heading
        else:
            heading = float(self._last_stable_heading.get(ped_id, 0.0))

        def pair_variance(records, attribute):
            if len(records) <= 1:
                return 0.0
            count = float(len(records))
            sum_x = sum(float(getattr(item, attribute)[0]) for item in records)
            sum_y = sum(float(getattr(item, attribute)[1]) for item in records)
            mean_x = sum_x / count
            mean_y = sum_y / count
            var_x = sum(
                (float(getattr(item, attribute)[0]) - mean_x) ** 2
                for item in records
            ) / count
            var_y = sum(
                (float(getattr(item, attribute)[1]) - mean_y) ** 2
                for item in records
            ) / count
            return 0.5 * (var_x + var_y)

        pos_var = pair_variance(valid_records, "position")
        vel_var = pair_variance(valid_records, "velocity")

        latest_valid = valid_records[-1] if valid_records else None
        latest_record = history[-1]
        latest_vx = (
            float(latest_valid.velocity[0]) if latest_valid is not None else 0.0
        )
        latest_vy = (
            float(latest_valid.velocity[1]) if latest_valid is not None else 0.0
        )
        recent_change = math.hypot(
            latest_vx - vx_hat, latest_vy - vy_hat
        )
        shortage = float(max(0, min(2, 2 - valid_count))) * 0.15
        uncertainty = (
            self.base_uncertainty
            + self.missing_uncertainty_gain * missing_ratio
            + self.velocity_variance_gain * vel_var
            + self.position_variance_gain * pos_var
            + self.recent_change_gain * recent_change
            + shortage
        )
        uncertainty = float(
            min(self.max_uncertainty, max(0.0, uncertainty))
        )

        return {
            "ped_id": ped_id,
            "position": latest_record.position.tolist(),
            "last_observed_position": (
                latest_valid.position.tolist() if latest_valid is not None else latest_record.position.tolist()
            ),
            "v_hat": [vx_hat, vy_hat],
            "speed": speed,
            "heading": heading,
            "uncertainty": uncertainty,
            "pos_var": pos_var,
            "vel_var": vel_var,
            "missing_ratio": missing_ratio,
            "valid_count": int(valid_count),
            "history_count": int(len(history)),
            "valid": bool(valid_count > 0),
        }

    def batch_estimate(self) -> Dict[Any, Dict[str, Any]]:
        return {ped_id: self.estimate(ped_id) for ped_id in self._history}

    def reachable_region(self, ped_id: Any, tau: float = 1.0, shape: str = "ellipse") -> Dict[str, Any]:
        if shape != "ellipse":
            raise ValueError("Only ellipse reachable regions are implemented.")

        tau = _validate_tau_values([tau], "tau")[0]
        estimate = self.estimate(ped_id)
        return self._reachable_region_from_estimate(ped_id, estimate, tau)

    def _reachable_region_from_estimate(
        self,
        ped_id: Any,
        estimate: Dict[str, Any],
        tau: float,
    ) -> Dict[str, Any]:
        position = (
            estimate.get("last_observed_position")
            or estimate.get("position")
            or [0.0, 0.0]
        )
        velocity = estimate.get("v_hat") or [0.0, 0.0]
        px = _safe_float(position[0] if len(position) > 0 else 0.0)
        py = _safe_float(position[1] if len(position) > 1 else 0.0)
        vx = _safe_float(velocity[0] if len(velocity) > 0 else 0.0)
        vy = _safe_float(velocity[1] if len(velocity) > 1 else 0.0)
        speed = _safe_float(estimate.get("speed"))
        heading = _safe_float(estimate.get("heading"))
        uncertainty = _safe_float(estimate.get("uncertainty"))

        center = [px + tau * vx, py + tau * vy]
        a = self.a_base + tau * speed + self.rho_uncertainty * uncertainty
        b = self.b_base + self.rho_side * uncertainty

        if speed <= self.min_heading_speed:
            theta = float(self._last_stable_heading.get(ped_id, heading))
            radius = max(a, b)
            a = radius
            b = radius
        else:
            theta = heading

        a = float(min(self.max_axis, max(self.min_axis, a)))
        b = float(min(self.max_axis, max(self.min_axis, b)))
        center = [
            value if math.isfinite(value) else 0.0
            for value in center
        ]

        return {
            "type": "ellipse",
            "center": center,
            "a": a,
            "b": b,
            "theta": float(theta),
            "tau": tau,
            "uncertainty": uncertainty,
            "speed": speed,
            "heading": heading,
            "valid": bool(estimate.get("valid", False)),
        }

    def reachable_regions(self, ped_id: Any, tau_list: Optional[Iterable[float]] = None) -> Dict[float, Dict[str, Any]]:
        estimate = self.estimate(ped_id)
        return self.reachable_regions_from_estimate(
            ped_id, estimate, tau_list=tau_list
        )

    def reachable_regions_from_estimate(
        self,
        ped_id: Any,
        estimate: Dict[str, Any],
        tau_list: Optional[Iterable[float]] = None,
    ) -> Dict[float, Dict[str, Any]]:
        taus = (
            self.tau_list
            if tau_list is None
            else _validate_tau_values(tau_list, "tau_list")
        )
        return {
            float(tau): self._reachable_region_from_estimate(ped_id, estimate, tau=float(tau))
            for tau in taus
        }

    def export_history(self) -> Dict[Any, Any]:
        return {ped_id: [record.to_debug_dict() for record in history] for ped_id, history in self._history.items()}

    def get_debug_state(self) -> Dict[str, Any]:
        return {
            "K": self.K,
            "alpha_decay": self.alpha_decay,
            "pedestrian_count": len(self._history),
            "history": self.export_history(),
            "last_stable_heading": dict(self._last_stable_heading),
        }

    def _empty_estimate(self, ped_id: Any) -> Dict[str, Any]:
        return {
            "ped_id": ped_id,
            "position": [0.0, 0.0],
            "last_observed_position": [0.0, 0.0],
            "v_hat": [0.0, 0.0],
            "speed": 0.0,
            "heading": float(self._last_stable_heading.get(ped_id, 0.0)),
            "uncertainty": self.max_uncertainty,
            "pos_var": 0.0,
            "vel_var": 0.0,
            "missing_ratio": 1.0,
            "valid_count": 0,
            "history_count": 0,
            "valid": False,
        }

    def _last_position(self, history: Deque[LMTERecord]) -> np.ndarray:
        for record in reversed(history):
            if record.valid:
                return record.position.copy()
        if history:
            return history[-1].position.copy()
        return np.zeros(2, dtype=np.float64)

    def _resolve_velocity(
        self,
        history: Deque[LMTERecord],
        position: np.ndarray,
        velocity: Optional[Iterable[float]],
        timestamp: Optional[float],
    ) -> np.ndarray:
        if velocity is not None:
            return _vec2(velocity)

        last_valid = None
        for record in reversed(history):
            if record.valid:
                last_valid = record
                break
        if last_valid is None:
            return np.zeros(2, dtype=np.float64)

        if timestamp is not None and last_valid.timestamp is not None:
            dt = _safe_float(timestamp, 0.0) - _safe_float(last_valid.timestamp, 0.0)
            if dt <= 1e-6:
                dt = self.default_dt
        else:
            dt = self.default_dt
        if dt <= 1e-6:
            dt = self.default_dt
        return np.nan_to_num((position - last_valid.position) / dt, nan=0.0, posinf=0.0, neginf=0.0)
