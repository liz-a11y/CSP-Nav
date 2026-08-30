from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


EPS = 1e-9
FLOAT_MAX = float(np.finfo(np.float64).max)


DEFAULT_PARAMS = {
    "csp": {
        "lambda_ps": 1.0,
        "lambda_enc": 1.0,
        "lambda_blk": 1.0,
        "lambda_reach": 0.75,
        "cvar_beta": 0.8,
        "top_k": 2,
    },
    "personal_space": {
        "a_front": 1.2,
        "a_back": 0.6,
        "b_side": 0.8,
        "d_ps_threshold": 1.5,
        "ps_gain": 1.0,
    },
    "enclosure": {
        "sigma_enc": 1.2,
        "enc_range": 3.0,
        "enc_gain": 1.0,
    },
    "blocking": {
        "corridor_base_length": 0.8,
        "corridor_tau": 1.5,
        "corridor_width": 0.8,
        "min_speed_for_heading": 0.05,
        "blk_gain": 1.0,
    },
    "reachable": {
        "reach_threshold": 1.5,
        "reach_gain": 1.0,
        "reach_axis_margin": 0.10,
        "tau_weights": {0.5: 0.4, 1.0: 0.3, 1.5: 0.2, 2.0: 0.1},
    },
}


def _section(params: Optional[Dict[str, Any]], name: str) -> Dict[str, Any]:
    merged = dict(DEFAULT_PARAMS[name])
    if params:
        if name in params and isinstance(params[name], dict):
            merged.update(params[name])
        else:
            merged.update({key: value for key, value in params.items() if key in merged})
    return merged


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    if math.isinf(out):
        return math.copysign(FLOAT_MAX, out)
    return out


def _vec2(value: Any, default: Optional[np.ndarray] = None) -> np.ndarray:
    if default is None:
        default = np.zeros(2, dtype=np.float64)
    if isinstance(value, dict):
        for key in ("position", "pos", "p", "xy", "center"):
            if key in value:
                value = value[key]
                break
        else:
            if "px" in value and "py" in value:
                value = [value["px"], value["py"]]
            elif "x" in value and "y" in value:
                value = [value["x"], value["y"]]
    if hasattr(value, "get_position"):
        value = value.get_position()
    elif hasattr(value, "px") and hasattr(value, "py"):
        value = [getattr(value, "px"), getattr(value, "py")]
    if value is None:
        return np.asarray(default, dtype=np.float64).copy()
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray(default, dtype=np.float64).copy()
    out = np.asarray(default, dtype=np.float64).copy()
    out[: min(2, arr.size)] = arr[:2]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _bounded_vec2(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    out = np.zeros(2, dtype=np.float64)
    out[: min(2, arr.size)] = arr[:2]
    return np.nan_to_num(
        out,
        nan=0.0,
        posinf=FLOAT_MAX,
        neginf=-FLOAT_MAX,
    )


def _saturating_add_scaled(
    position: np.ndarray,
    velocity: np.ndarray,
    scale: float,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = position + _safe_float(scale) * velocity
    return _bounded_vec2(result)


def _saturating_sub(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = first - second
    return _bounded_vec2(result)


def _safe_product(first: float, second: float) -> float:
    first = _bounded_float(first)
    second = _bounded_float(second)
    if first == 0.0 or second == 0.0:
        return 0.0
    if abs(first) > FLOAT_MAX / abs(second):
        sign = -1.0 if (first < 0.0) != (second < 0.0) else 1.0
        return sign * FLOAT_MAX
    return first * second


def _safe_add(first: float, second: float) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        return _bounded_float(np.add(first, second))


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = max(abs(_safe_float(denominator, EPS)), EPS)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _bounded_float(np.divide(numerator, denominator))


def _safe_hypot(first: float, second: float) -> float:
    return _bounded_float(math.hypot(_bounded_float(first), _bounded_float(second)))


def _safe_norm(vec: np.ndarray) -> float:
    arr = _bounded_vec2(vec)
    return _safe_hypot(arr[0], arr[1])


def _safe_dot2(first: np.ndarray, second: np.ndarray) -> float:
    first = _bounded_vec2(first)
    second = _bounded_vec2(second)
    scale = max(abs(float(first[0])), abs(float(first[1])))
    if scale <= EPS:
        return 0.0
    scaled_dot = float(
        (first[0] / scale) * second[0]
        + (first[1] / scale) * second[1]
    )
    return _safe_product(scale, scaled_dot)


def _safe_unit2(vec: np.ndarray) -> np.ndarray:
    arr = _bounded_vec2(vec)
    scale = max(abs(float(arr[0])), abs(float(arr[1])))
    if scale <= EPS:
        return np.zeros(2, dtype=np.float64)
    scaled = arr / scale
    norm = math.hypot(float(scaled[0]), float(scaled[1]))
    if norm <= EPS:
        return np.zeros(2, dtype=np.float64)
    return scaled / norm


def _gaussian_decay(ratio: float) -> float:
    ratio = abs(_bounded_float(ratio))
    if ratio >= 27.0:
        return 0.0
    return math.exp(-(ratio * ratio))


def _state_id(state: Any, *keys: str, default: Any = "") -> Any:
    if isinstance(state, dict):
        for key in keys:
            if key in state:
                return state[key]
    for key in keys:
        if hasattr(state, key):
            return getattr(state, key)
    return default


def _velocity(state: Any) -> np.ndarray:
    if isinstance(state, dict):
        if "velocity" in state:
            return _vec2(state["velocity"])
        if "v_hat" in state:
            return _vec2(state["v_hat"])
        if "vx" in state and "vy" in state:
            return _vec2([state["vx"], state["vy"]])
    if hasattr(state, "velocity"):
        return _vec2(getattr(state, "velocity"))
    if hasattr(state, "vx") and hasattr(state, "vy"):
        return _vec2([getattr(state, "vx"), getattr(state, "vy")])
    return np.zeros(2, dtype=np.float64)


def _heading(state: Any, fallback: float = 0.0) -> float:
    if isinstance(state, dict):
        for key in ("heading", "theta"):
            if key in state:
                return _safe_float(state[key], fallback)
    else:
        for key in ("heading", "theta"):
            if hasattr(state, key):
                return _safe_float(getattr(state, key), fallback)
    vel = _velocity(state)
    if _safe_norm(vel) > EPS:
        return float(math.atan2(vel[1], vel[0]))
    return fallback


def _radius(state: Any, default: float = 0.2) -> float:
    if isinstance(state, dict):
        radius = _safe_float(state.get("radius", default), default)
    else:
        radius = _safe_float(getattr(state, "radius", default), default)
    return max(radius, 0.0)


def _is_active(state: Any) -> bool:
    if isinstance(state, dict):
        return bool(state.get("active", True))
    return bool(getattr(state, "active", True))


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, _safe_float(value)))


def _as_list(states: Optional[Iterable[Any]]) -> List[Any]:
    if states is None:
        return []
    if isinstance(states, dict):
        return list(states.values())
    return list(states)


def _active_states(states: Optional[Iterable[Any]]) -> List[Any]:
    return [state for state in _as_list(states) if _is_active(state)]


def _lmte_for_human(lmte_outputs: Any, human_id: Any) -> Dict[str, Any]:
    if lmte_outputs is None:
        return {}
    if isinstance(lmte_outputs, dict):
        if human_id in lmte_outputs:
            return lmte_outputs[human_id] or {}
        if str(human_id) in lmte_outputs:
            return lmte_outputs[str(human_id)] or {}
        if any(
            key in lmte_outputs
            for key in ("v_hat", "heading", "reachable_region", "reachable_regions")
        ):
            return lmte_outputs
        return {}
    for item in lmte_outputs:
        item_id = _state_id(item, "human_id", "ped_id", "id", default=None)
        if str(item_id) == str(human_id):
            return item or {}
    return {}


def elliptical_distance(
    robot_pos: Iterable[float],
    human_pos: Iterable[float],
    human_heading: float,
    a_front: float,
    a_back: float,
    b_side: float,
) -> float:
    robot_pos = _vec2(robot_pos)
    human_pos = _vec2(human_pos)
    rel = _saturating_sub(robot_pos, human_pos)
    heading = _safe_float(human_heading)
    forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
    side = np.asarray([-math.sin(heading), math.cos(heading)], dtype=np.float64)
    x_parallel = _safe_dot2(rel, forward)
    y_perp = _safe_dot2(rel, side)
    a = max(_safe_float(a_front if x_parallel >= 0.0 else a_back, 1.0), EPS)
    b = max(_safe_float(b_side, 1.0), EPS)
    return _safe_hypot(_safe_ratio(x_parallel, a), _safe_ratio(y_perp, b))


def aggregate_personal_space_pressure(
    s_ps_list: Iterable[float],
    mode: str = "soft_union",
) -> float:
    values = np.asarray([_clip01(value) for value in s_ps_list], dtype=np.float64)
    if values.size == 0:
        return 0.0
    if mode == "sum":
        return float(np.sum(values))
    if mode == "max":
        return float(np.max(values))
    if mode == "soft_union":
        return _clip01(1.0 - float(np.prod(1.0 - values)))
    raise ValueError(f"Unknown personal-space aggregation mode: {mode}")


def _personal_space_components(
    robot_state: Any,
    human_state: Any,
    cfg: Dict[str, Any],
) -> Tuple[float, float, int]:
    active = _is_active(robot_state)
    d_ell = elliptical_distance(
        _vec2(robot_state),
        _vec2(human_state),
        _heading(human_state),
        cfg["a_front"],
        cfg["a_back"],
        cfg["b_side"],
    )
    threshold = max(_safe_float(cfg["d_ps_threshold"], 1.5), 0.0)
    s_ps = (
        _clip01(_safe_float(cfg["ps_gain"], 1.0) * math.exp(-d_ell))
        if active and d_ell < threshold
        else 0.0
    )
    return d_ell, s_ps, int(active and d_ell < 1.0)


def compute_personal_space_pressure(
    robot_state: Any,
    human_state: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _section(params, "personal_space")
    d_ell, s_ps, inside_personal_space = _personal_space_components(
        robot_state,
        human_state,
        cfg,
    )
    threshold = max(_safe_float(cfg["d_ps_threshold"], 1.5), 0.0)
    return {
        "human_id": _state_id(human_state, "human_id", "ped_id", "id", default="human"),
        "robot_id": _state_id(robot_state, "robot_id", "id", default="robot"),
        "d_ell": d_ell,
        "s_ps": s_ps,
        "inside_personal_space": inside_personal_space,
        "comfort_boundary_radius": 1.0,
        "ellipse_params": {
            "a_front": _safe_float(cfg["a_front"], 1.2),
            "a_back": _safe_float(cfg["a_back"], 0.6),
            "b_side": _safe_float(cfg["b_side"], 0.8),
            "d_ps_threshold": threshold,
        },
    }


def _enclosure_robot_components(
    robot: Any,
    human_pos: np.ndarray,
    sigma: float,
    enc_range: float,
) -> Tuple[np.ndarray, float, float]:
    rel = _saturating_sub(_vec2(robot), human_pos)
    dist = _safe_norm(rel)
    direction = _safe_unit2(rel)
    range_gate = max(0.0, 1.0 - _safe_ratio(dist, enc_range))
    weight = _clip01(math.exp(-_safe_ratio(dist, sigma)) * range_gate)
    return direction, dist, weight


def compute_enclosure_pressure(
    robot_states: Iterable[Any],
    human_state: Any,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _section(params, "enclosure")
    robots = _active_states(robot_states)
    human_pos = _vec2(human_state)
    human_id = _state_id(human_state, "human_id", "ped_id", "id", default="human")
    if len(robots) < 2:
        return {
            "human_id": human_id,
            "P_enc": 0.0,
            "pair_terms": [],
            "max_pair_term": 0.0,
            "pair_angle_deg": 0.0,
            "closest_pair_robot_ids": [],
        }

    sigma = max(_safe_float(cfg["sigma_enc"], 1.2), EPS)
    enc_range = max(_safe_float(cfg["enc_range"], 3.0), EPS)
    enc_gain = _safe_float(cfg["enc_gain"], 1.0)
    robot_infos = []
    for robot in robots:
        direction, dist, weight = _enclosure_robot_components(
            robot,
            human_pos,
            sigma,
            enc_range,
        )
        robot_infos.append(
            {
                "robot_id": _state_id(robot, "robot_id", "id", default=len(robot_infos)),
                "direction": direction,
                "dist": dist,
                "weight": weight,
            }
        )

    terms = []
    for first, second in combinations(robot_infos, 2):
        dot = float(np.clip(np.dot(first["direction"], second["direction"]), -1.0, 1.0))
        angle = math.degrees(math.acos(dot))
        term = enc_gain * first["weight"] * second["weight"] * max(0.0, -dot)
        terms.append(
            {
                "robot_i": first["robot_id"],
                "robot_l": second["robot_id"],
                "term": _clip01(term),
                "dot": dot,
                "angle_deg": angle,
                "dist_i": first["dist"],
                "dist_l": second["dist"],
            }
        )
    raw = _safe_float(np.sum([item["term"] for item in terms]), 0.0) if terms else 0.0
    max_item = max(terms, key=lambda item: item["term"], default=None)
    return {
        "human_id": human_id,
        "P_enc": _clip01(raw),
        "P_enc_raw": raw,
        "pair_terms": terms,
        "max_pair_term": 0.0 if max_item is None else float(max_item["term"]),
        "pair_angle_deg": 0.0 if max_item is None else float(max_item["angle_deg"]),
        "closest_pair_robot_ids": (
            [] if max_item is None else [max_item["robot_i"], max_item["robot_l"]]
        ),
    }


def _blocking_context(
    human_state: Any,
    lmte_output: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, float]:
    human_pos = _vec2(human_state)
    speed = _safe_float(
        lmte_output.get("speed"),
        _safe_norm(_velocity(human_state)),
    )
    heading_confidence = 1.0
    if "heading" in lmte_output:
        heading = _safe_float(lmte_output.get("heading"))
    else:
        heading = _heading(human_state)
        heading_confidence = 0.5
    human_has_heading = isinstance(human_state, dict) and any(
        key in human_state for key in ("heading", "theta")
    )
    min_heading_speed = max(_safe_float(cfg["min_speed_for_heading"], 0.05), 0.0)
    if speed < min_heading_speed and "heading" not in lmte_output and not human_has_heading:
        heading_confidence = 0.0
    forward = np.asarray([math.cos(heading), math.sin(heading)], dtype=np.float64)
    corridor_length = max(
        _safe_add(
            _safe_float(cfg["corridor_base_length"], 0.8),
            _safe_product(
                _safe_float(cfg["corridor_tau"], 1.5),
                max(speed, min_heading_speed),
            ),
        ),
        EPS,
    )
    corridor_width = max(_safe_float(cfg["corridor_width"], 0.8), EPS)
    return (
        human_pos,
        forward,
        corridor_length,
        corridor_width,
        _safe_float(cfg["blk_gain"], 1.0),
        heading,
        heading_confidence,
    )


def _blocking_robot_components(
    robot: Any,
    human_pos: np.ndarray,
    forward: np.ndarray,
    corridor_length: float,
    corridor_width: float,
    blk_gain: float,
    heading_confidence: float,
) -> Tuple[float, float, float]:
    rel = _saturating_sub(_vec2(robot), human_pos)
    proj = _safe_dot2(rel, forward)
    lateral_vec = _saturating_sub(
        rel,
        _saturating_add_scaled(
            np.zeros(2, dtype=np.float64),
            forward,
            proj,
        ),
    )
    lateral_dist = _safe_norm(lateral_vec)
    forward_weight = 1.0 if 0.0 < proj < corridor_length else 0.0
    lateral_scale = _safe_add(corridor_width, _radius(robot, 0.2))
    lateral_weight = _gaussian_decay(
        _safe_ratio(lateral_dist, max(lateral_scale, EPS))
    )
    distance_weight = (
        math.exp(-_safe_ratio(proj, corridor_length))
        if proj > 0.0
        else 0.0
    )
    score = _clip01(
        blk_gain
        * forward_weight
        * lateral_weight
        * distance_weight
        * heading_confidence
    )
    return score, proj, lateral_dist


def compute_blocking_pressure(
    robot_states: Iterable[Any],
    human_state: Any,
    lmte_output: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _section(params, "blocking")
    robots = _active_states(robot_states)
    human_id = _state_id(human_state, "human_id", "ped_id", "id", default="human")
    lmte_output = lmte_output or {}
    (
        human_pos,
        forward,
        corridor_length,
        corridor_width,
        blk_gain,
        heading,
        heading_confidence,
    ) = _blocking_context(human_state, lmte_output, cfg)

    block_scores = []
    for robot in robots:
        score, proj, lateral_dist = _blocking_robot_components(
            robot,
            human_pos,
            forward,
            corridor_length,
            corridor_width,
            blk_gain,
            heading_confidence,
        )
        block_scores.append(
            {
                "robot_id": _state_id(robot, "robot_id", "id", default=len(block_scores)),
                "block_score": score,
                "proj": proj,
                "lateral_dist": lateral_dist,
            }
        )

    P_blk = aggregate_personal_space_pressure(
        [item["block_score"] for item in block_scores],
        mode="soft_union",
    )
    blocking_robot_ids = [
        item["robot_id"] for item in block_scores if item["block_score"] > 0.2
    ]
    blocking_rate = len(blocking_robot_ids) / len(robots) if robots else 0.0
    return {
        "human_id": human_id,
        "P_blk": P_blk,
        "blocking_rate": float(blocking_rate),
        "blocked": int(P_blk > 0.25),
        "max_block_score": max(
            [item["block_score"] for item in block_scores],
            default=0.0,
        ),
        "blocking_robot_ids": blocking_robot_ids,
        "block_terms": block_scores,
        "corridor_length": corridor_length,
        "corridor_width": corridor_width,
        "corridor_heading": float(heading),
        "heading_confidence": float(heading_confidence),
    }


def _parsed_reachable_regions(lmte_output: Dict[str, Any]) -> Dict[float, Dict[str, Any]]:
    regions = lmte_output.get("reachable_regions", {})
    parsed = {}
    if isinstance(regions, dict):
        for tau, region in regions.items():
            tau_value = _safe_float(tau, -1.0)
            if tau_value > 0.0 and isinstance(region, dict):
                parsed[tau_value] = region
    if not parsed:
        region = lmte_output.get("reachable_region")
        if isinstance(region, dict):
            tau_value = _safe_float(region.get("tau"), 1.0)
            if tau_value > 0.0:
                parsed[tau_value] = region
    return parsed


def _iter_tau_weights(cfg: Dict[str, Any]) -> Iterable[Tuple[float, float]]:
    raw_weights = cfg.get("tau_weights", {})
    if not isinstance(raw_weights, dict):
        return
    for tau, weight in raw_weights.items():
        tau_value = _safe_float(tau, -1.0)
        if tau_value > 0.0:
            yield tau_value, max(_safe_float(weight, 0.0), 0.0)


def _reachable_robot_components(
    robot: Any,
    tau: float,
    region: Dict[str, Any],
    axis_margin: float,
    threshold: float,
    gain: float,
) -> Tuple[float, np.ndarray, float]:
    center = _vec2(region.get("center"))
    a = max(_safe_float(region.get("a"), 0.0), 0.0)
    b = max(_safe_float(region.get("b"), 0.0), 0.0)
    theta = _safe_float(region.get("theta"), 0.0)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    predicted = _saturating_add_scaled(
        _vec2(robot),
        _velocity(robot),
        tau,
    )
    rel = _saturating_sub(predicted, center)
    local_x = _safe_dot2(rel, [cos_theta, sin_theta])
    local_y = _safe_dot2(rel, [-sin_theta, cos_theta])
    radius_margin = _safe_add(_radius(robot, 0.2), axis_margin)
    a_eff = max(_safe_add(a, radius_margin), EPS)
    b_eff = max(_safe_add(b, radius_margin), EPS)
    distance = _safe_hypot(
        _safe_ratio(local_x, a_eff),
        _safe_ratio(local_y, b_eff),
    )
    pressure = (
        _clip01(gain * math.exp(-distance))
        if distance < threshold
        else 0.0
    )
    return pressure, predicted, distance


def compute_reachable_pressure(
    robot_states: Iterable[Any],
    human_state: Any,
    lmte_output: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = _section(params, "reachable")
    robots = _active_states(robot_states)
    human_id = _state_id(human_state, "human_id", "ped_id", "id", default="human")
    lmte_output = lmte_output or {}
    regions_by_tau = _parsed_reachable_regions(lmte_output)

    threshold = max(_safe_float(cfg["reach_threshold"], 1.5), 0.0)
    gain = max(_safe_float(cfg["reach_gain"], 1.0), 0.0)
    axis_margin = max(_safe_float(cfg["reach_axis_margin"], 0.10), 0.0)
    horizon_terms = []

    for tau, weight in _iter_tau_weights(cfg):
        region = regions_by_tau.get(tau)
        robot_terms = []
        if region is not None and bool(region.get("valid", True)):
            for robot in robots:
                pressure, predicted, distance = _reachable_robot_components(
                    robot,
                    tau,
                    region,
                    axis_margin,
                    threshold,
                    gain,
                )
                robot_terms.append(
                    {
                        "robot_id": _state_id(
                            robot,
                            "robot_id",
                            "id",
                            default=len(robot_terms),
                        ),
                        "predicted_position": predicted.tolist(),
                        "elliptical_distance": distance,
                        "pressure": pressure,
                    }
                )
        horizon_pressure = aggregate_personal_space_pressure(
            [item["pressure"] for item in robot_terms],
            mode="soft_union",
        )
        horizon_terms.append(
            {
                "tau": tau,
                "weight": weight,
                "pressure": horizon_pressure,
                "robot_terms": robot_terms,
            }
        )

    weight_sum = sum(item["weight"] for item in horizon_terms)
    if weight_sum > EPS:
        P_reach = sum(
            item["weight"] * item["pressure"] for item in horizon_terms
        ) / weight_sum
    else:
        P_reach = 0.0
    return {
        "human_id": human_id,
        "P_reach": _clip01(P_reach),
        "horizon_terms": horizon_terms,
        "max_reach_pressure": max(
            [item["pressure"] for item in horizon_terms],
            default=0.0,
        ),
    }


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if math.isfinite(out):
        return out
    return default if math.isnan(out) else math.copysign(FLOAT_MAX, out)


def _compact_clip01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def _scalar_add(first: float, second: float) -> float:
    result = first + second
    if math.isfinite(result):
        return result
    if math.isnan(result):
        return 0.0
    return math.copysign(FLOAT_MAX, result)


def _scalar_sub(first: float, second: float) -> float:
    result = first - second
    if math.isfinite(result):
        return result
    if math.isnan(result):
        return 0.0
    return math.copysign(FLOAT_MAX, result)


def _scalar_mul(first: float, second: float) -> float:
    if first == 0.0 or second == 0.0:
        return 0.0
    if abs(first) > FLOAT_MAX / abs(second):
        sign = -1.0 if (first < 0.0) != (second < 0.0) else 1.0
        return sign * FLOAT_MAX
    return first * second


def _scalar_ratio(numerator: float, denominator: float) -> float:
    denominator = max(abs(denominator), EPS)
    result = numerator / denominator
    if math.isfinite(result):
        return result
    return math.copysign(FLOAT_MAX, numerator)


def _scalar_hypot(first: float, second: float) -> float:
    result = math.hypot(first, second)
    return result if math.isfinite(result) else FLOAT_MAX


def _scalar_dot(
    first_x: float,
    first_y: float,
    second_x: float,
    second_y: float,
) -> float:
    result = first_x * second_x + first_y * second_y
    if math.isfinite(result):
        return result
    return _scalar_add(
        _scalar_mul(first_x, second_x),
        _scalar_mul(first_y, second_y),
    )


def _scalar_unit(x_value: float, y_value: float) -> Tuple[float, float]:
    scale = max(abs(x_value), abs(y_value))
    if scale <= EPS:
        return 0.0, 0.0
    scaled_x = x_value / scale
    scaled_y = y_value / scale
    norm = math.hypot(scaled_x, scaled_y)
    if norm <= EPS:
        return 0.0, 0.0
    return scaled_x / norm, scaled_y / norm


def _pair_from_value(value: Any) -> Tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    try:
        return _finite_float(value[0]), _finite_float(value[1])
    except (KeyError, IndexError, TypeError):
        return 0.0, 0.0


def _compact_position(state: Any) -> Tuple[float, float]:
    if isinstance(state, dict):
        for key in ("position", "pos", "p", "xy", "center"):
            if key in state:
                return _pair_from_value(state[key])
        if "px" in state and "py" in state:
            return _finite_float(state["px"]), _finite_float(state["py"])
        if "x" in state and "y" in state:
            return _finite_float(state["x"]), _finite_float(state["y"])
        return 0.0, 0.0
    if hasattr(state, "get_position"):
        return _pair_from_value(state.get_position())
    if hasattr(state, "px") and hasattr(state, "py"):
        return _finite_float(state.px), _finite_float(state.py)
    return 0.0, 0.0


def _compact_velocity(state: Any) -> Tuple[float, float]:
    if isinstance(state, dict):
        if "velocity" in state:
            return _pair_from_value(state["velocity"])
        if "v_hat" in state:
            return _pair_from_value(state["v_hat"])
        if "vx" in state and "vy" in state:
            return _finite_float(state["vx"]), _finite_float(state["vy"])
        return 0.0, 0.0
    if hasattr(state, "velocity"):
        return _pair_from_value(state.velocity)
    if hasattr(state, "vx") and hasattr(state, "vy"):
        return _finite_float(state.vx), _finite_float(state.vy)
    return 0.0, 0.0


def _compact_radius(state: Any, default: float = 0.2) -> float:
    value = state.get("radius", default) if isinstance(state, dict) else getattr(state, "radius", default)
    return max(_finite_float(value, default), 0.0)


def _compact_heading(
    state: Any,
    velocity_x: float,
    velocity_y: float,
) -> Tuple[float, bool]:
    if isinstance(state, dict):
        for key in ("heading", "theta"):
            if key in state:
                return _finite_float(state[key]), True
    else:
        for key in ("heading", "theta"):
            if hasattr(state, key):
                return _finite_float(getattr(state, key)), False
    if _scalar_hypot(velocity_x, velocity_y) > EPS:
        return math.atan2(velocity_y, velocity_x), False
    return 0.0, False


def _compact_robot_states(states: Iterable[Any]) -> List[Tuple[Any, float, float, float, float, float]]:
    parsed = []
    for state in _as_list(states):
        if not _is_active(state):
            continue
        position_x, position_y = _compact_position(state)
        velocity_x, velocity_y = _compact_velocity(state)
        parsed.append(
            (
                _state_id(state, "robot_id", "id", default=len(parsed)),
                position_x,
                position_y,
                velocity_x,
                velocity_y,
                _compact_radius(state, 0.2),
            )
        )
    return parsed


def _compact_human_state(
    state: Any,
) -> Tuple[Any, float, float, float, float, float, float, bool]:
    position_x, position_y = _compact_position(state)
    velocity_x, velocity_y = _compact_velocity(state)
    heading, has_dict_heading = _compact_heading(state, velocity_x, velocity_y)
    return (
        _state_id(state, "human_id", "ped_id", "id", default="human"),
        position_x,
        position_y,
        velocity_x,
        velocity_y,
        _compact_radius(state, 0.3),
        heading,
        has_dict_heading,
    )


def _compact_parameter_sets(
    params: Optional[Dict[str, Any]],
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...], Tuple[Any, ...], Tuple[float, ...]]:
    personal_space = _section(params, "personal_space")
    enclosure = _section(params, "enclosure")
    blocking = _section(params, "blocking")
    reachable = _section(params, "reachable")
    csp = _section(params, "csp")
    tau_weights = tuple(
        (tau, weight)
        for tau, weight in _iter_tau_weights(reachable)
    )
    return (
        (
            max(_finite_float(personal_space["a_front"], 1.2), EPS),
            max(_finite_float(personal_space["a_back"], 0.6), EPS),
            max(_finite_float(personal_space["b_side"], 0.8), EPS),
            max(_finite_float(personal_space["d_ps_threshold"], 1.5), 0.0),
            _finite_float(personal_space["ps_gain"], 1.0),
        ),
        (
            max(_finite_float(enclosure["sigma_enc"], 1.2), EPS),
            max(_finite_float(enclosure["enc_range"], 3.0), EPS),
            _finite_float(enclosure["enc_gain"], 1.0),
        ),
        (
            _finite_float(blocking["corridor_base_length"], 0.8),
            _finite_float(blocking["corridor_tau"], 1.5),
            max(_finite_float(blocking["corridor_width"], 0.8), EPS),
            max(_finite_float(blocking["min_speed_for_heading"], 0.05), 0.0),
            _finite_float(blocking["blk_gain"], 1.0),
        ),
        (
            max(_finite_float(reachable["reach_threshold"], 1.5), 0.0),
            max(_finite_float(reachable["reach_gain"], 1.0), 0.0),
            max(_finite_float(reachable["reach_axis_margin"], 0.10), 0.0),
            tau_weights,
        ),
        (
            _safe_float(csp["lambda_ps"], 1.0),
            _safe_float(csp["lambda_enc"], 1.0),
            _safe_float(csp["lambda_blk"], 1.0),
            _safe_float(csp["lambda_reach"], 0.75),
            min(0.99, max(0.0, _safe_float(csp["cvar_beta"], 0.8))),
            max(1, int(_safe_float(csp["top_k"], 2))),
        ),
    )


def _compact_lmte_state(
    lmte_output: Dict[str, Any],
    human: Tuple[Any, float, float, float, float, float, float, bool],
    blocking_cfg: Tuple[float, ...],
    reachable_cfg: Tuple[Any, ...],
) -> Tuple[float, float, float, Tuple[Tuple[Any, ...], ...]]:
    lmte_output = lmte_output or {}
    speed = _finite_float(
        lmte_output.get("speed"),
        _scalar_hypot(human[3], human[4]),
    )
    if "heading" in lmte_output:
        heading = _finite_float(lmte_output["heading"])
        heading_confidence = 1.0
    else:
        heading = human[6]
        heading_confidence = 0.5
        if speed < blocking_cfg[3] and not human[7]:
            heading_confidence = 0.0

    raw_regions = lmte_output.get("reachable_regions", {})
    regions_by_tau = {}
    if isinstance(raw_regions, dict):
        for raw_tau, raw_region in raw_regions.items():
            tau = _finite_float(raw_tau, -1.0)
            if tau > 0.0 and isinstance(raw_region, dict):
                regions_by_tau[tau] = raw_region
    if not regions_by_tau:
        singular = lmte_output.get("reachable_region")
        if isinstance(singular, dict):
            tau = _finite_float(singular.get("tau"), 1.0)
            if tau > 0.0:
                regions_by_tau[tau] = singular

    parsed_regions = []
    for tau, weight in reachable_cfg[3]:
        region = regions_by_tau.get(tau)
        if region is None or not bool(region.get("valid", True)):
            parsed_regions.append((tau, weight, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, False))
            continue
        center_x, center_y = _pair_from_value(region.get("center"))
        axis_a = max(_finite_float(region.get("a"), 0.0), 0.0)
        axis_b = max(_finite_float(region.get("b"), 0.0), 0.0)
        theta = _finite_float(region.get("theta"), 0.0)
        parsed_regions.append(
            (
                tau,
                weight,
                center_x,
                center_y,
                axis_a,
                axis_b,
                math.cos(theta),
                math.sin(theta),
                True,
            )
        )
    return speed, heading, heading_confidence, tuple(parsed_regions)


def _compact_personal_space(
    robots: List[Tuple[Any, float, float, float, float, float]],
    human: Tuple[Any, float, float, float, float, float, float, bool],
    cfg: Tuple[float, ...],
) -> float:
    forward_x = math.cos(human[6])
    forward_y = math.sin(human[6])
    complement_product = 1.0
    for _, robot_x, robot_y, _, _, _ in robots:
        rel_x = _scalar_sub(robot_x, human[1])
        rel_y = _scalar_sub(robot_y, human[2])
        parallel = _scalar_dot(rel_x, rel_y, forward_x, forward_y)
        perpendicular = _scalar_dot(rel_x, rel_y, -forward_y, forward_x)
        axis_a = cfg[0] if parallel >= 0.0 else cfg[1]
        distance = _scalar_hypot(
            _scalar_ratio(parallel, axis_a),
            _scalar_ratio(perpendicular, cfg[2]),
        )
        pressure = (
            _compact_clip01(cfg[4] * math.exp(-distance))
            if distance < cfg[3]
            else 0.0
        )
        complement_product *= 1.0 - pressure
    return _compact_clip01(1.0 - complement_product)


def _compact_enclosure(
    robots: List[Tuple[Any, float, float, float, float, float]],
    human: Tuple[Any, float, float, float, float, float, float, bool],
    cfg: Tuple[float, ...],
) -> float:
    if len(robots) < 2:
        return 0.0
    robot_terms = []
    for _, robot_x, robot_y, _, _, _ in robots:
        rel_x = _scalar_sub(robot_x, human[1])
        rel_y = _scalar_sub(robot_y, human[2])
        distance = _scalar_hypot(rel_x, rel_y)
        direction_x, direction_y = _scalar_unit(rel_x, rel_y)
        range_gate = max(0.0, 1.0 - _scalar_ratio(distance, cfg[1]))
        weight = _compact_clip01(math.exp(-_scalar_ratio(distance, cfg[0])) * range_gate)
        robot_terms.append((direction_x, direction_y, weight))
    raw = 0.0
    for first in range(len(robot_terms) - 1):
        first_x, first_y, first_weight = robot_terms[first]
        for second in range(first + 1, len(robot_terms)):
            second_x, second_y, second_weight = robot_terms[second]
            dot = min(
                1.0,
                max(-1.0, first_x * second_x + first_y * second_y),
            )
            raw += _compact_clip01(
                cfg[2]
                * first_weight
                * second_weight
                * max(0.0, -dot)
            )
    return _compact_clip01(raw)


def _compact_blocking(
    robots: List[Tuple[Any, float, float, float, float, float]],
    human: Tuple[Any, float, float, float, float, float, float, bool],
    lmte: Tuple[float, float, float, Tuple[Tuple[Any, ...], ...]],
    cfg: Tuple[float, ...],
) -> float:
    forward_x = math.cos(lmte[1])
    forward_y = math.sin(lmte[1])
    corridor_length = max(
        _scalar_add(cfg[0], _scalar_mul(cfg[1], max(lmte[0], cfg[3]))),
        EPS,
    )
    complement_product = 1.0
    for _, robot_x, robot_y, _, _, radius in robots:
        rel_x = _scalar_sub(robot_x, human[1])
        rel_y = _scalar_sub(robot_y, human[2])
        projection = _scalar_dot(rel_x, rel_y, forward_x, forward_y)
        lateral_x = _scalar_sub(rel_x, _scalar_mul(projection, forward_x))
        lateral_y = _scalar_sub(rel_y, _scalar_mul(projection, forward_y))
        lateral_distance = _scalar_hypot(lateral_x, lateral_y)
        forward_weight = 1.0 if 0.0 < projection < corridor_length else 0.0
        lateral_scale = max(_scalar_add(cfg[2], radius), EPS)
        lateral_weight = _gaussian_decay(
            _scalar_ratio(lateral_distance, lateral_scale)
        )
        distance_weight = (
            math.exp(-_scalar_ratio(projection, corridor_length))
            if projection > 0.0
            else 0.0
        )
        pressure = _compact_clip01(
            cfg[4]
            * forward_weight
            * lateral_weight
            * distance_weight
            * lmte[2]
        )
        complement_product *= 1.0 - pressure
    return _compact_clip01(1.0 - complement_product)


def _compact_reachable(
    robots: List[Tuple[Any, float, float, float, float, float]],
    lmte: Tuple[float, float, float, Tuple[Tuple[Any, ...], ...]],
    cfg: Tuple[Any, ...],
) -> float:
    weighted_pressure = 0.0
    weight_sum = 0.0
    for (
        tau,
        weight,
        center_x,
        center_y,
        axis_a,
        axis_b,
        cos_theta,
        sin_theta,
        valid,
    ) in lmte[3]:
        complement_product = 1.0
        if valid:
            for _, robot_x, robot_y, velocity_x, velocity_y, radius in robots:
                predicted_x = _scalar_add(robot_x, _scalar_mul(tau, velocity_x))
                predicted_y = _scalar_add(robot_y, _scalar_mul(tau, velocity_y))
                rel_x = _scalar_sub(predicted_x, center_x)
                rel_y = _scalar_sub(predicted_y, center_y)
                local_x = _scalar_dot(rel_x, rel_y, cos_theta, sin_theta)
                local_y = _scalar_dot(rel_x, rel_y, -sin_theta, cos_theta)
                margin = _scalar_add(radius, cfg[2])
                axis_a_eff = max(_scalar_add(axis_a, margin), EPS)
                axis_b_eff = max(_scalar_add(axis_b, margin), EPS)
                distance = _scalar_hypot(
                    _scalar_ratio(local_x, axis_a_eff),
                    _scalar_ratio(local_y, axis_b_eff),
                )
                pressure = (
                    _compact_clip01(cfg[1] * math.exp(-distance))
                    if distance < cfg[0]
                    else 0.0
                )
                complement_product *= 1.0 - pressure
        horizon_pressure = _compact_clip01(1.0 - complement_product)
        weighted_pressure += weight * horizon_pressure
        weight_sum += weight
    if weight_sum <= EPS:
        return 0.0
    return _compact_clip01(weighted_pressure / weight_sum)


def _compact_csp_values(
    robots: List[Tuple[Any, float, float, float, float, float]],
    human: Tuple[Any, float, float, float, float, float, float, bool],
    lmte: Tuple[float, float, float, Tuple[Tuple[Any, ...], ...]],
    personal_space_cfg: Tuple[float, ...],
    enclosure_cfg: Tuple[float, ...],
    blocking_cfg: Tuple[float, ...],
    reachable_cfg: Tuple[Any, ...],
    csp_cfg: Tuple[float, ...],
) -> Tuple[float, float, float, float, float]:
    P_ps = _compact_personal_space(robots, human, personal_space_cfg)
    P_enc = _compact_enclosure(robots, human, enclosure_cfg)
    P_blk = _compact_blocking(robots, human, lmte, blocking_cfg)
    P_reach = _compact_reachable(robots, lmte, reachable_cfg)
    CSP_j = 0.0
    for weight, pressure in (
        (csp_cfg[0], P_ps),
        (csp_cfg[1], P_enc),
        (csp_cfg[2], P_blk),
        (csp_cfg[3], P_reach),
    ):
        term = weight * pressure
        if not math.isfinite(term):
            CSP_j = 0.0
            break
        CSP_j += term
        if not math.isfinite(CSP_j):
            CSP_j = 0.0
            break
    return P_ps, P_enc, P_blk, P_reach, CSP_j


def _compact_csp_for_human(
    robot_states: Iterable[Any],
    human_state: Any,
    lmte_output: Dict[str, Any],
    params: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    robots = _compact_robot_states(robot_states)
    human = _compact_human_state(human_state)
    (
        personal_space_cfg,
        enclosure_cfg,
        blocking_cfg,
        reachable_cfg,
        csp_cfg,
    ) = _compact_parameter_sets(params)
    lmte = _compact_lmte_state(
        lmte_output,
        human,
        blocking_cfg,
        reachable_cfg,
    )
    P_ps, P_enc, P_blk, P_reach, CSP_j = _compact_csp_values(
        robots,
        human,
        lmte,
        personal_space_cfg,
        enclosure_cfg,
        blocking_cfg,
        reachable_cfg,
        csp_cfg,
    )
    return {
        "P_ps": P_ps,
        "P_enc": P_enc,
        "P_blk": P_blk,
        "P_reach": P_reach,
        "CSP_j": CSP_j,
    }


def compute_csp_for_human(
    robot_states: Iterable[Any],
    human_state: Any,
    lmte_output: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    include_details: bool = True,
) -> Dict[str, Any]:
    lmte_output = lmte_output or {}
    if not include_details:
        return _compact_csp_for_human(
            robot_states,
            human_state,
            lmte_output,
            params,
        )
    robots = _active_states(robot_states)
    csp_cfg = _section(params, "csp")
    ps_items = [
        compute_personal_space_pressure(robot, human_state, params=params)
        for robot in robots
    ]
    s_values = [item["s_ps"] for item in ps_items]
    P_ps_sum = aggregate_personal_space_pressure(s_values, mode="sum")
    P_ps_max = aggregate_personal_space_pressure(s_values, mode="max")
    P_ps_soft_union = aggregate_personal_space_pressure(s_values, mode="soft_union")
    enc = compute_enclosure_pressure(robots, human_state, params=params)
    blk = compute_blocking_pressure(
        robots,
        human_state,
        lmte_output=lmte_output,
        params=params,
    )
    reachable = compute_reachable_pressure(
        robots,
        human_state,
        lmte_output=lmte_output,
        params=params,
    )
    P_ps = _clip01(P_ps_soft_union)
    P_enc = _clip01(enc["P_enc"])
    P_blk = _clip01(blk["P_blk"])
    P_reach = _clip01(reachable["P_reach"])
    CSP_j = _safe_float(
        _safe_float(csp_cfg["lambda_ps"], 1.0) * P_ps
        + _safe_float(csp_cfg["lambda_enc"], 1.0) * P_enc
        + _safe_float(csp_cfg["lambda_blk"], 1.0) * P_blk
        + _safe_float(csp_cfg["lambda_reach"], 0.75) * P_reach,
        0.0,
    )
    pressures = {
        "P_ps": P_ps,
        "P_enc": P_enc,
        "P_blk": P_blk,
        "P_reach": P_reach,
    }
    dominant = max(pressures, key=pressures.get)
    return {
        "human_id": _state_id(human_state, "human_id", "ped_id", "id", default="human"),
        "P_ps": P_ps,
        "P_enc": P_enc,
        "P_blk": P_blk,
        "P_reach": P_reach,
        "CSP_j": CSP_j,
        "P_ps_sum": float(P_ps_sum),
        "P_ps_max": float(P_ps_max),
        "P_ps_soft_union": float(P_ps_soft_union),
        "personal_space_terms": ps_items,
        "enclosure": enc,
        "blocking": blk,
        "reachable": reachable,
        "enclosure_count": len(enc.get("pair_terms", [])),
        "enclosure_flag": int(P_enc > 0.2),
        "blocking_rate": float(blk["blocking_rate"]),
        "blocking_flag": int(P_blk > 0.25),
        "personal_space_violation": int(
            any(item["inside_personal_space"] for item in ps_items)
        ),
        "ps_violation_flag": int(
            any(item["inside_personal_space"] for item in ps_items)
        ),
        "dominant_pressure_type": dominant,
        "d_ell_min": min(
            [item["d_ell"] for item in ps_items],
            default=0.0,
        ),
        "s_ps_max": max([item["s_ps"] for item in ps_items], default=0.0),
    }


def _tail_mean(csp_values: np.ndarray, csp_cfg: Dict[str, Any]) -> float:
    beta = min(0.99, max(0.0, _safe_float(csp_cfg["cvar_beta"], 0.8)))
    top_k_cfg = max(1, int(_safe_float(csp_cfg["top_k"], 2)))
    tail_k = max(top_k_cfg, int(math.ceil((1.0 - beta) * len(csp_values))), 1)
    tail_k = min(tail_k, len(csp_values))
    return float(np.mean(np.sort(csp_values)[-tail_k:]))


def _compact_tail_mean(
    csp_values: List[float],
    csp_cfg: Tuple[float, ...],
) -> float:
    tail_k = max(
        int(csp_cfg[5]),
        int(math.ceil((1.0 - csp_cfg[4]) * len(csp_values))),
        1,
    )
    tail_k = min(tail_k, len(csp_values))
    top_values = sorted(csp_values)[-tail_k:]
    return sum(top_values) / tail_k


def _empty_compact_scene() -> Dict[str, Any]:
    return {
        "CSP_scene_mean": 0.0,
        "CSP_scene_max": 0.0,
        "CSP_scene_CVaR": 0.0,
        "mean_P_ps": 0.0,
        "mean_P_enc": 0.0,
        "mean_P_blk": 0.0,
        "mean_P_reach": 0.0,
        "max_P_reach": 0.0,
        "human_count": 0,
    }


def _compact_scene_csp(
    robot_states: Iterable[Any],
    humans: List[Any],
    lmte_outputs: Optional[Any],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not humans:
        return _empty_compact_scene()
    robots = _compact_robot_states(robot_states)
    human_count = len(humans)
    csp_values = []
    ps_values = []
    enc_values = []
    blk_values = []
    reach_values = []
    (
        personal_space_cfg,
        enclosure_cfg,
        blocking_cfg,
        reachable_cfg,
        csp_cfg,
    ) = _compact_parameter_sets(params)
    for human_state in humans:
        human = _compact_human_state(human_state)
        lmte = _compact_lmte_state(
            _lmte_for_human(lmte_outputs, human[0]),
            human,
            blocking_cfg,
            reachable_cfg,
        )
        P_ps, P_enc, P_blk, P_reach, CSP_j = _compact_csp_values(
            robots,
            human,
            lmte,
            personal_space_cfg,
            enclosure_cfg,
            blocking_cfg,
            reachable_cfg,
            csp_cfg,
        )
        csp_values.append(CSP_j)
        ps_values.append(P_ps)
        enc_values.append(P_enc)
        blk_values.append(P_blk)
        reach_values.append(P_reach)
    return {
        "CSP_scene_mean": sum(csp_values) / human_count,
        "CSP_scene_max": max(csp_values),
        "CSP_scene_CVaR": _compact_tail_mean(csp_values, csp_cfg),
        "mean_P_ps": sum(ps_values) / human_count,
        "mean_P_enc": sum(enc_values) / human_count,
        "mean_P_blk": sum(blk_values) / human_count,
        "mean_P_reach": sum(reach_values) / human_count,
        "max_P_reach": max(reach_values),
        "human_count": human_count,
    }


def compute_scene_csp(
    robot_states: Iterable[Any],
    human_states: Iterable[Any],
    lmte_outputs: Optional[Any] = None,
    params: Optional[Dict[str, Any]] = None,
    include_details: bool = True,
) -> Dict[str, Any]:
    humans = _as_list(human_states)
    if not include_details:
        return _compact_scene_csp(
            robot_states,
            humans,
            lmte_outputs,
            params,
        )
    robots = _active_states(robot_states)
    if not humans:
        return {
            "CSP_scene_mean": 0.0,
            "CSP_scene_max": 0.0,
            "CSP_scene_CVaR": 0.0,
            "CSP_scene_topk": 0.0,
            "CSP_scene": 0.0,
            "CSP_mean": 0.0,
            "CSP_max": 0.0,
            "CSP_CVaR": 0.0,
            "CSP_per_human": [],
            "worst_human_id": "",
            "worst_pressure_type": "",
            "mean_P_ps": 0.0,
            "mean_P_enc": 0.0,
            "mean_P_blk": 0.0,
            "mean_P_reach": 0.0,
            "max_P_ps": 0.0,
            "max_P_enc": 0.0,
            "max_P_blk": 0.0,
            "max_P_reach": 0.0,
        }

    per_human = []
    for human in humans:
        human_id = _state_id(human, "human_id", "ped_id", "id", default="human")
        lmte = _lmte_for_human(lmte_outputs, human_id)
        per_human.append(
            compute_csp_for_human(
                robots,
                human,
                lmte_output=lmte,
                params=params,
            )
        )

    csp_values = np.asarray([item["CSP_j"] for item in per_human], dtype=np.float64)
    ps_values = np.asarray([item["P_ps"] for item in per_human], dtype=np.float64)
    enc_values = np.asarray([item["P_enc"] for item in per_human], dtype=np.float64)
    blk_values = np.asarray([item["P_blk"] for item in per_human], dtype=np.float64)
    reach_values = np.asarray([item["P_reach"] for item in per_human], dtype=np.float64)
    csp_cfg = _section(params, "csp")
    cvar_value = _tail_mean(csp_values, csp_cfg)
    worst = max(per_human, key=lambda item: item["CSP_j"])
    return {
        "CSP_scene_mean": float(np.mean(csp_values)),
        "CSP_scene_max": float(np.max(csp_values)),
        "CSP_scene_CVaR": cvar_value,
        "CSP_scene_topk": cvar_value,
        "CSP_scene": float(np.mean(csp_values)),
        "CSP_mean": float(np.mean(csp_values)),
        "CSP_max": float(np.max(csp_values)),
        "CSP_CVaR": cvar_value,
        "CSP_per_human": per_human,
        "worst_human_id": worst["human_id"],
        "worst_pressure_type": worst["dominant_pressure_type"],
        "mean_P_ps": float(np.mean(ps_values)),
        "mean_P_enc": float(np.mean(enc_values)),
        "mean_P_blk": float(np.mean(blk_values)),
        "mean_P_reach": float(np.mean(reach_values)),
        "max_P_ps": float(np.max(ps_values)),
        "max_P_enc": float(np.max(enc_values)),
        "max_P_blk": float(np.max(blk_values)),
        "max_P_reach": float(np.max(reach_values)),
    }


class CollectiveSocialPressureMetrics:
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}

    def elliptical_distance(
        self,
        robot_pos: Iterable[float],
        human_pos: Iterable[float],
        human_heading: float,
    ) -> float:
        cfg = _section(self.params, "personal_space")
        return elliptical_distance(
            robot_pos,
            human_pos,
            human_heading,
            cfg["a_front"],
            cfg["a_back"],
            cfg["b_side"],
        )

    def compute_personal_space_pressure(
        self,
        robot_state: Any,
        human_state: Any,
    ) -> Dict[str, Any]:
        return compute_personal_space_pressure(robot_state, human_state, self.params)

    def aggregate_personal_space_pressure(
        self,
        s_ps_list: Iterable[float],
        mode: str = "soft_union",
    ) -> float:
        return aggregate_personal_space_pressure(s_ps_list, mode)

    def compute_enclosure_pressure(
        self,
        robot_states: Iterable[Any],
        human_state: Any,
    ) -> Dict[str, Any]:
        return compute_enclosure_pressure(robot_states, human_state, self.params)

    def compute_blocking_pressure(
        self,
        robot_states: Iterable[Any],
        human_state: Any,
        lmte_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return compute_blocking_pressure(
            robot_states,
            human_state,
            lmte_output,
            self.params,
        )

    def compute_reachable_pressure(
        self,
        robot_states: Iterable[Any],
        human_state: Any,
        lmte_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return compute_reachable_pressure(
            robot_states,
            human_state,
            lmte_output,
            self.params,
        )

    def compute_csp_for_human(
        self,
        robot_states: Iterable[Any],
        human_state: Any,
        lmte_output: Optional[Dict[str, Any]] = None,
        include_details: bool = True,
    ) -> Dict[str, Any]:
        return compute_csp_for_human(
            robot_states,
            human_state,
            lmte_output,
            self.params,
            include_details,
        )

    def compute_scene_csp(
        self,
        robot_states: Iterable[Any],
        human_states: Iterable[Any],
        lmte_outputs: Optional[Any] = None,
        include_details: bool = True,
    ) -> Dict[str, Any]:
        return compute_scene_csp(
            robot_states,
            human_states,
            lmte_outputs,
            self.params,
            include_details,
        )
