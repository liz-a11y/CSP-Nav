from dataclasses import dataclass
from enum import Enum

import numpy as np


class AgentStatus(str, Enum):
    ACTIVE = "active"
    REACHED = "reached"
    COLLIDED = "collided"


@dataclass(frozen=True)
class AgentMasks:
    agent_dones: np.ndarray
    active_masks: np.ndarray
    rnn_masks: np.ndarray
    bad_masks: np.ndarray
    env_done: bool


def _status_array(statuses):
    statuses = np.asarray(statuses, dtype=object)
    if statuses.ndim != 1:
        raise ValueError("statuses must be a one-dimensional array")
    return statuses


def mix_active_rewards(
    individual_rewards,
    active_at_step_start,
    individual_coef=0.8,
    team_coef=0.2,
):
    if not np.isclose(individual_coef + team_coef, 1.0):
        raise ValueError("individual_coef and team_coef must sum to 1")

    individual_rewards = np.asarray(individual_rewards, dtype=np.float32)
    active_at_step_start = np.asarray(active_at_step_start, dtype=bool)
    if individual_rewards.shape != active_at_step_start.shape:
        raise ValueError("rewards and active mask must have identical shapes")
    if individual_rewards.ndim != 1:
        raise ValueError("rewards must be a one-dimensional array")

    mixed_rewards = np.zeros_like(individual_rewards)
    if not np.any(active_at_step_start):
        return mixed_rewards, 0.0

    team_reward = float(individual_rewards[active_at_step_start].mean())
    mixed_rewards[active_at_step_start] = (
        individual_coef * individual_rewards[active_at_step_start]
        + team_coef * team_reward
    )
    return mixed_rewards, team_reward


def build_agent_masks(previous_status, current_status, timeout=False):
    previous_status = _status_array(previous_status)
    current_status = _status_array(current_status)
    if previous_status.shape != current_status.shape:
        raise ValueError("previous and current status arrays must match")

    was_active = np.fromiter(
        (status == AgentStatus.ACTIVE for status in previous_status),
        dtype=bool,
        count=len(previous_status),
    )
    is_active = np.fromiter(
        (status == AgentStatus.ACTIVE for status in current_status),
        dtype=bool,
        count=len(current_status),
    )
    newly_terminal = np.logical_and(was_active, np.logical_not(is_active))

    if timeout:
        agent_dones = was_active.copy()
        bad_masks = np.ones((len(current_status), 1), dtype=np.float32)
        still_active = np.logical_and(was_active, is_active)
        bad_masks[still_active, 0] = 0.0
        env_done = True
    else:
        agent_dones = newly_terminal
        bad_masks = np.ones((len(current_status), 1), dtype=np.float32)
        env_done = bool(np.all(np.logical_not(is_active)))

    active_masks = is_active.astype(np.float32).reshape(-1, 1)
    rnn_masks = np.logical_not(agent_dones).astype(np.float32).reshape(-1, 1)
    return AgentMasks(
        agent_dones=agent_dones,
        active_masks=active_masks,
        rnn_masks=rnn_masks,
        bad_masks=bad_masks,
        env_done=env_done,
    )


def circle_clearance(center_a, radius_a, center_b, radius_b):
    center_a = np.asarray(center_a, dtype=np.float64)
    center_b = np.asarray(center_b, dtype=np.float64)
    return float(np.linalg.norm(center_a - center_b) - radius_a - radius_b)


def _orientation(a, b, c, epsilon=1e-12):
    value = np.cross(np.asarray(b) - np.asarray(a), np.asarray(c) - np.asarray(a))
    if abs(value) <= epsilon:
        return 0
    return 1 if value > 0 else -1


def _on_segment(a, b, point, epsilon=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    return bool(
        np.all(point >= np.minimum(a, b) - epsilon)
        and np.all(point <= np.maximum(a, b) + epsilon)
    )


def _segments_intersect(a0, a1, b0, b1):
    o1 = _orientation(a0, a1, b0)
    o2 = _orientation(a0, a1, b1)
    o3 = _orientation(b0, b1, a0)
    o4 = _orientation(b0, b1, a1)
    if o1 != o2 and o3 != o4:
        return True
    return (
        (o1 == 0 and _on_segment(a0, a1, b0))
        or (o2 == 0 and _on_segment(a0, a1, b1))
        or (o3 == 0 and _on_segment(b0, b1, a0))
        or (o4 == 0 and _on_segment(b0, b1, a1))
    )


def _point_segment_distance(point, start, end):
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0.0:
        return float(np.linalg.norm(point - start))
    projection = np.clip(np.dot(point - start, segment) / length_squared, 0.0, 1.0)
    return float(np.linalg.norm(point - (start + projection * segment)))


def segment_distance(a0, a1, b0, b1):
    if _segments_intersect(a0, a1, b0, b1):
        return 0.0
    return min(
        _point_segment_distance(a0, b0, b1),
        _point_segment_distance(a1, b0, b1),
        _point_segment_distance(b0, a0, a1),
        _point_segment_distance(b1, a0, a1),
    )


def count_crossing_pairs(starts, goals, threshold):
    starts = np.asarray(starts, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    if starts.shape != goals.shape or starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("starts and goals must both have shape [agent_count, 2]")

    pair_count = 0
    for first in range(len(starts)):
        for second in range(first + 1, len(starts)):
            if (
                segment_distance(
                    starts[first],
                    goals[first],
                    starts[second],
                    goals[second],
                )
                <= threshold
            ):
                pair_count += 1
    return pair_count
