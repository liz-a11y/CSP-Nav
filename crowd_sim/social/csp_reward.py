from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

from .csp_metrics import CollectiveSocialPressureMetrics


INFO_KEYS = (
    "CSP_scene_mean",
    "CSP_scene_max",
    "CSP_scene_CVaR",
    "mean_P_ps",
    "mean_P_enc",
    "mean_P_blk",
    "mean_P_reach",
    "max_P_reach",
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


class CSPRewardWrapper:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config or {}
        self.features = self.config.get("features", {}) or {}
        self.reward_cfg = self.config.get("reward", {}) or {}
        self.csp_metrics = CollectiveSocialPressureMetrics(self.config)

    def compute(
        self,
        env_reward: np.ndarray,
        active_at_step_start: Iterable[bool],
        robot_states: Iterable[Any],
        human_states: Iterable[Any],
        lmte_outputs: Optional[Any] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        reward = np.asarray(env_reward)
        active = np.asarray(active_at_step_start, dtype=bool).reshape(-1)
        if reward.ndim == 0 or reward.shape[0] != active.size:
            raise ValueError(
                "env_reward first dimension must match active_at_step_start"
            )

        scene = {key: 0.0 for key in INFO_KEYS}
        enabled = bool(self.features.get("enable_csp_reward", True))
        reporting = bool(self.features.get("enable_csp_reporting", False))
        if enabled or reporting:
            compact = self.csp_metrics.compute_scene_csp(
                robot_states,
                human_states,
                lmte_outputs,
                include_details=False,
            )
            for key in INFO_KEYS:
                scene[key] = _finite_float(compact.get(key))

        eta_csp = max(0.0, _finite_float(self.reward_cfg.get("eta_csp", 0.02), 0.02))
        csp_clip = max(
            0.0,
            _finite_float(self.reward_cfg.get("csp_clip", 1.5), 1.5),
        )
        r_csp = (
            -eta_csp
            * float(np.clip(scene["CSP_scene_CVaR"], 0.0, csp_clip))
            if enabled
            else 0.0
        )

        output_dtype = (
            reward.dtype
            if np.issubdtype(reward.dtype, np.floating)
            else np.dtype(np.float32)
        )
        modified = reward.astype(np.float64, copy=True)
        modified[active] += r_csp
        modified = modified.astype(output_dtype, copy=False)

        info = {"r_csp": float(r_csp)}
        info.update(scene)
        return modified, info
